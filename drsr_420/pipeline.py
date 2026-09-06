# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Implementation of the LLMSR pipeline. """
from __future__ import annotations

# from collections.abc import Sequence
import json
import os
import threading
from typing import Any, Tuple, Sequence
import numpy as np

from drsr_420 import code_manipulation
from drsr_420 import config as config_lib
from drsr_420 import evaluator
from drsr_420 import buffer
from drsr_420 import sampler
from drsr_420 import profile
from drsr_420 import data_analyse_real
from drsr_420.console import print_block
from drsr_420.find_best_eq import find_best_eq


# from drsr_420.db_helper import get_context


def _extract_function_names(specification: str) -> Tuple[str, str]:
    """ Return the name of the function to evolve and of the function to run.

    The so-called specification refers to the boilerplate code template for a task.
    The template MUST have two important functions decorated with '@evaluate.run', '@equation.evolve' respectively.
    The function labeled with '@evaluate.run' is going to evaluate the generated code (like data-diven fitness evaluation).
    The function labeled with '@equation.evolve' is the function to be searched (like 'equation' structure).
    """
    run_functions = list(code_manipulation.yield_decorated(specification, 'evaluate', 'run'))
    if len(run_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@evaluate.run`.')
    evolve_functions = list(code_manipulation.yield_decorated(specification, 'equation', 'evolve'))
    
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@equation.evolve`.')
    
    return evolve_functions[0], run_functions[0]



def main(
        specification: str,
        inputs: Sequence[Any],
        config: config_lib.Config,
        max_sample_nums: int | None,
        class_config: config_lib.ClassConfig,
        **kwargs
):
    """ Launch a LLMSR experiment.
    Args:
        specification: the boilerplate code for the problem.
        inputs       : the data instances for the problem.
        config       : config file.
        max_sample_nums: the maximum samples nums from LLM. 'None' refers to no stop.
    """
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)
    database = buffer.ExperienceBuffer(config.experience_buffer, template, function_to_evolve)

    # Profiler：直接基于 results_root（不再使用 logs 子目录）
    results_root = kwargs.get('results_root', None) or config.results_root
    llm_config = kwargs.get('llm_config', None)
    llm_client = kwargs.get('llm_client', None)

    # 工程加固：断点续跑——若存在 checkpoint，恢复经验缓冲与全局采样数
    if results_root:
        checkpoint_path = os.path.join(results_root, "checkpoint.json")
        if os.path.exists(checkpoint_path):
            try:
                database.load_checkpoint(checkpoint_path)
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                sampler.SamplingOrchestrator.set_global_sample_nums(int(ckpt.get("global_sample_nums", 1)))
                print(f"[INFO] 已从 checkpoint 恢复：全局采样数={ckpt.get('global_sample_nums')}")
            except Exception as e:
                print(f"[WARN] 恢复 checkpoint 失败（从头开始）: {e}")
    target_variance = None
    try:
        if hasattr(inputs, 'values'):
            any_dataset = next(iter(inputs.values()))
            if isinstance(any_dataset, dict) and 'outputs' in any_dataset:
                y = np.asarray(any_dataset['outputs'])
                if y.size > 0:
                    target_variance = float(np.var(y))
    except Exception:
        target_variance = None
    # Profiler：记录样本与中间结果（包括 Top-K、历史最优与逐 iteration 进度）
    profiler = profile.Profiler(
        results_root,
        samples_per_iteration=config.samples_per_prompt,
        target_variance=target_variance,
        persist_all_samples=bool(kwargs.get('persist_all_samples', False)),
    ) if results_root else None

    seed = kwargs.get('seed', None)

    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(evaluator.Evaluator(
            database,
            template,
            function_to_evolve,
            function_to_run,
            inputs,     # the data instances for the problem.
            timeout_seconds=config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class
        ))

    initial = template.get_function(function_to_evolve).body
    ini_score, error_msg ,res= evaluators[0].analyse(initial, island_id=None, version_generated=None, profiler=profiler)

    # 创建DataAnalyzer实例
    # DataAnalyzer 也写入统一结果目录（直接使用 results_root）
    # 将 LLM API 配置传递给 DataAnalyzer（通过模块级常量覆盖，最小侵入）
    if llm_config and not llm_client:
        try:
            import drsr_420.data_analyse_real as _dar
            if llm_config.get('host') or llm_config.get('base_url'):
                _dar.API_HOST = llm_config.get('host') or llm_config.get('base_url')
            if 'api_key' in llm_config: _dar.API_KEY = llm_config['api_key']
            if 'model' in llm_config: _dar.API_MODEL = llm_config['model']
            if 'max_tokens' in llm_config: _dar.MAX_TOKENS = llm_config['max_tokens']
        except Exception:
            pass
    analyzer = data_analyse_real.DataAnalyzer(timeout=600, base_dir=results_root, llm_client=llm_client, seed=seed)  # 可以自定义参数

    # PromptContext 用于动态渲染初次数据分析/残差分析提示（变量名、因变量、输出格式均动态化）
    prompt_ctx = kwargs.get('prompt_ctx', None)
    # 分析指定的CSV文件
    # {np.concatenate(
    #     (inputs.get('data', None).get('inputs', None), inputs.get('data', None).get('outputs', None).reshape(-1, 1)),
    #     axis=1)}

    # 初次数据分析：提示模板由 PromptContext 动态渲染（变量名/因变量/输出格式动态化），避免领域与变量名硬编码
    initial_analysis_prompt = prompt_ctx.render_initial_analysis_prompt() if prompt_ctx else None
    # RAG 检索增强：从文献知识库检索物理背景并注入初次分析提示（检索失败或库为空时静默跳过，不影响主流程）
    if initial_analysis_prompt:
        try:
            from drsr_420 import prompt_config as _pc
            from drsr_420.rag_kb import get_kb, load_config
            _rag_cfg = load_config()
            _rag_query = (
                kwargs.get('rag_query', None)
                or (prompt_ctx.background_text if prompt_ctx else None)
                or _rag_cfg.get('default_query', '')
            )
            _kb = get_kb()
            _lit_ctx = _kb.get_context(_rag_query, k=_rag_cfg.get('k', 5)) if _kb.count() > 0 else ""
            _block = f"{_pc.literature_block_title}{_lit_ctx}\n" if _lit_ctx else ""
            initial_analysis_prompt = initial_analysis_prompt.replace("{literature_context}", _block)
            if _lit_ctx:
                print(f"[RAG] 已注入 {_lit_ctx.count(chr(10))} 行文献上下文（{_rag_query}）")
        except Exception as _e:
            initial_analysis_prompt = initial_analysis_prompt.replace("{literature_context}", "")
            print(f"[RAG] 文献上下文注入失败（跳过）: {_e}")
    result = analyzer.analyze(
        inputs,
        initial_analysis_prompt,
        # max_rows=1000,  # 可选：限制行数
        verbose=True    # 可选：显示详细信息
    )

    # 打印分析结果（截断 + 带线程前缀，避免无前缀长文本刷屏）
    print_block("\n===== 分析结果 =====")
    print_block(result)

    # Set global max sample nums.
    # 每个 sampler 独享一份 Evaluator 列表：避免多线程并发调用同一 Evaluator/Sandbox，
    # 防止 sandbox 上的 _last_params 等实例可变状态竞态。
    samplers = []
    for _ in range(config.num_samplers):
        evals = [
            evaluator.Evaluator(
                database,
                template,
                function_to_evolve,
                function_to_run,
                inputs,     # the data instances for the problem.
                timeout_seconds=config.evaluate_timeout_seconds,
                sandbox_class=class_config.sandbox_class,
            ) for _ in range(config.num_evaluators)
        ]
        samplers.append(sampler.SamplingOrchestrator(
            database, evals,
            config.samples_per_prompt,
            max_sample_nums=max_sample_nums,
            llm_class=class_config.llm_class,
            config=config,
            prompt_ctx=prompt_ctx,
            llm_client=llm_client,
            llm_api=None,
        ))

    # 多线程并行启动多个 sampler：共享经验缓冲（内部加锁），并行调用 LLM 提升吞吐。
    # 每个 sampler 进入无限采样循环，直到全局采样数达到上限或实验超时。
    # 线程命名 Sampler-<i>，日志带前缀以体现并行执行。
    threads = []
    for i, s in enumerate(samplers):
        t = threading.Thread(target=s.sample, kwargs={'profiler': profiler}, daemon=True, name=f"Sampler-{i}")
        print(f"[Sampler-{i}] 采样线程启动，开始并行采样", flush=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    find_best_eq(results_root)
