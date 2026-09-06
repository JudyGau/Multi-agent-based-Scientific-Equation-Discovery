"""LLM 采样与编排：生成方程（Sampler）与采样主循环（SamplingOrchestrator）。

职责拆分（各组件可独立单测）：
- Sampler（生成方程）：调用 LLM 生成方程程序骨架，含提示词构造（指令/任务头/经验注入）。
- ToolCaller（MCP 工具循环）：与 LLM 多轮对话并自动执行工具调用。见 tool_caller.py
- ExperienceSummarizer（经验总结）：分析方程与得分给出改进建议。见 experience_summarizer.py
- ResidualAnalyzer（残差分析）：根据残差分析方程。见 residual_analyzer.py
- SamplingOrchestrator（采样编排）：sample() 主循环，编排上述组件与评估/经验缓冲/断点续跑。
"""
from __future__ import annotations

import re
import copy
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type, Any
import numpy as np
import time

import random
import threading
from drsr_420.console import LineStreamPrinter, print_block
from drsr_420 import evaluator
from drsr_420 import buffer
from drsr_420 import config as config_lib
import json
import os
import traceback
import csv
from drsr_420 import prompt_config as pc
from llm import LLMClient

from drsr_420.tool_caller import ToolCaller
from drsr_420.experience_summarizer import ExperienceSummarizer
from drsr_420.residual_analyzer import ResidualAnalyzer

Port = '5000'

# API配置
API_HOST = "api.bltcy.ai"
API_KEY = "sk-1zejrP7CKGPUXASwGpow3vOQ1Pjl5QzeU8xCjMrOEMSbqFQd"
API_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 1024

# 多 sampler 并行时保护共享文件读写与全局采样计数。
# 使用 RLock：内部方法 _get_global_sample_nums 会在 with _SAMPLER_LOCK 块内被再次调用，
# 普通 Lock 不可重入会导致同线程自锁死锁（Sampler 全部挂起）。
_SAMPLER_LOCK = threading.RLock()


def _atomic_write_json(path: str, data) -> None:
    """原子写 JSON：先写临时文件再替换，避免并发读方读到半成品。"""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _clone_llm_client(client, **kwargs_overrides):
    """基于现有 LLMClient 复制一份独立实例，使各实例 kwargs 互不影响。

    原先对同一个 llm_client 连续执行 kwargs.update，导致采样/经验/残差三个用途
    的生成参数互相覆盖（最终全部变成 temperature=0.4），并覆盖了 llm_summary.config 中
    用户配置的温度。这里通过浅拷贝 + 独立 kwargs 字典修复。
    """
    if client is None:
        return None
    new_client = copy.copy(client)
    new_client.kwargs = dict(client.kwargs)
    new_client.kwargs.update(kwargs_overrides)
    # 重置独立实例的累计统计，避免计数重复累加
    new_client._call_index = 0
    new_client.tokens = {'prompt': 0, 'content': 0, 'reasoning': 0, 'total': 0}
    new_client._cum_tokens = {
        'prompt': 0, 'thinking': 0, 'content': 0, 'total': 0,
    }
    new_client._cum_time_seconds = 0.0
    return new_client


class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]
    # self._samples_per_prompt = 4 每一次prompt都生成四个相互独立的回答


class Sampler(LLM):
    """生成方程：调用 LLM 生成方程程序骨架。

    提示词构造（指令、任务头、历史经验/残差注入）与 MCP 工具循环都封装在此，
    工具循环委托给 ToolCaller。
    """

    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True,
                 prompt_ctx: pc.PromptContext | None = None,
                 llm_client: LLMClient | None = None) -> None:
        """
        Args:
            batch_inference: Use batch inference when sample equation program skeletons. The batch size equals to the samples_per_prompt.
        """
        super().__init__(samples_per_prompt)

        self._prompt_ctx = prompt_ctx
        self._llm_client = llm_client
        instruction_prompt = (self._prompt_ctx.render_instruction() if self._prompt_ctx else pc.instruction_prompt)
        self._batch_inference = batch_inference
        self._instruction_prompt = instruction_prompt
        self._trim = trim
        # MCP 工具循环组件
        self._tool_caller = ToolCaller(llm_client)
        # 本地文件目录（用于加载经验/残差），由 draw_samples 时设置
        self._base_dir = "."

        ####################################
        # 添加会话ID存储
        self._conversation_ids = {}  # 用于存储每个样本的对话ID

    def draw_samples(self, prompt: str, config: config_lib.Config) -> tuple[list[Any] | list[str], list[Any]] | None:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        # 记录统一结果目录供本地路径引用
        try:
            self._base_dir = config.results_root or "."
        except Exception:
            self._base_dir = "."

        # 统一走本地批量请求（已使用注入的 llm_client）
        return self._draw_samples_local(prompt, config)

    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> tuple[list[Any] | list[str], list[
        Any]] | None:
        # instruction
        prompt = '\n'.join([self._instruction_prompt, prompt])
        while True:
            try:
                all_samples = []
                all_thinking_contents = []
                # response from llm server
                if self._batch_inference:
                    print("运行了_draw_samples_local的_batch_inference分支")
                    content = self._build_request_content(prompt)
                    first_responses, thinking_contents = self._tool_caller.complete(
                        content, self._samples_per_prompt)
                    print("成功运行first_responses = ToolCaller.complete")
                    print_block(first_responses)
                    all_samples = list(first_responses)
                    all_thinking_contents = list(thinking_contents)
                else:
                    for _ in range(self._samples_per_prompt):
                        content = self._build_request_content(prompt)
                        first_responses, _second_responses = self._tool_caller.complete(content, 1)
                        all_samples.append(first_responses)

                # trim equation program skeleton body from samples
                if self._trim:
                    all_samples = [_extract_body(sample, config) for sample in all_samples]

                return all_samples, all_thinking_contents
            except Exception:
                print(Exception)
                continue

    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        all_samples = []
        prompt = '\n'.join([self._instruction_prompt, prompt])
        for _ in range(self._samples_per_prompt):
            try:
                stream = LineStreamPrinter()
                shown = 0
                think_label_printed = False
                content_label_printed = False

                def _on_delta(chunk):
                    nonlocal shown, think_label_printed, content_label_printed
                    reasoning = chunk.get('reasoning_content') or ''
                    content = chunk.get('content') or ''
                    text = reasoning + content
                    if len(text) > shown:
                        if shown < len(reasoning) and not think_label_printed:
                            stream.write("[思考]\n")
                            think_label_printed = True
                        elif not content_label_printed:
                            stream.write_line("[正文]")
                            content_label_printed = True
                        stream.write(text[shown:])
                        shown = len(text)

                resp = self._llm_client.chat([{"role": "user", "content": prompt}], on_delta=_on_delta)
                stream.flush()
                print("\n====================================================\n")
                # 兜底：content 为空时回退到 reasoning，避免模型只思考不输出正文时骨架丢失
                response = resp.get('content', '') or resp.get('reasoning_content', '')
                if self._trim:
                    response = _extract_body(response, config)
                all_samples.append(response)
            except Exception:
                all_samples.append("")
        return all_samples

    def _build_request_content(self, content: str) -> str:
        """构造最终发送给 LLM 的内容：任务头 + 历史经验/残差注入。"""
        content = content.strip('\n').strip()

        # 尝试加载经验数据
        try:
            # 计算经验文件中所有类别经验的总数
            current_sample_order = 0

            experience_file = os.path.join(getattr(self, "_base_dir", "."), "experiences.json")

            if os.path.exists(experience_file):
                with open(experience_file, "r", encoding="utf-8") as f:
                    experiences = json.load(f)

                # 统计所有类别的经验总数
                for category in ["None", "Good", "Bad"]:
                    if category in experiences:
                        current_sample_order += len(experiences[category])

                # 准备存储筛选后的各类经验。
                # 规则：
                # - None：始终参与注入，最多 3 条；
                # - Good / Bad：各自独立以 0.5 概率参与注入，最多 2 条；
                # - 经验筛选范围与原逻辑保持一致。
                filtered_experiences = {"None": [], "Good": [], "Bad": []}
                optional_category_probability = 0.5
                category_max_samples = {"None": 3, "Good": 2, "Bad": 2}

                # 根据当前 sample_order 选择合适的经验
                for category in ["None", "Good", "Bad"]:
                    if category not in experiences or not experiences[category]:
                        continue

                    # None 类经验始终注入；Good / Bad 先按概率决定是否注入。
                    if category != "None" and random.random() >= optional_category_probability:
                        continue

                    # 筛选符合条件的经验
                    if current_sample_order <= 50:
                        # sample_order < 50 时，不限制经验的 sample_order
                        filtered_category = experiences[category]
                    else:
                        # sample_order > 50 时，只选择 sample_order 在当前值的 0.7~1 倍范围内的经验
                        min_order = current_sample_order * 0.7
                        max_order = current_sample_order
                        filtered_category = [
                            exp for exp in experiences[category]
                            if "sample_order" in exp and min_order <= exp["sample_order"] <= max_order
                        ]

                    filtered_experiences[category] = filtered_category

                # 合并所有类别的经验
                all_selected_experiences = []
                for category, exps in filtered_experiences.items():
                    for exp in exps:
                        experience_entry = {
                            "type": category,
                            "analysis": exp.get("analysis", ""),
                            "sample_order": exp.get("sample_order", "unknown"),
                        }

                        # 对于 None 类别，添加错误信息（如果有）
                        if category == "None" and "error" in exp:
                            error_msg = exp["error"]
                            # 移除特定错误信息（如果需要）
                            if error_msg == "Execution Error: too many values to unpack (expected 5)":
                                error_msg = ""

                            if error_msg:
                                experience_entry["error"] = error_msg

                        all_selected_experiences.append(experience_entry)

                # 如果有经验可用，构建经验提示
                if all_selected_experiences:
                    experience_prompt = pc.ideas_block_title

                    # 为每个经验分配编号，并标注类别（成功经验/待改进/失败教训），
                    # 帮助模型区分“要复制的成功因子”与“要避免的失败”。
                    label_map = {"Good": "successful experience", "Bad": "needs improvement", "None": "failure lesson"}
                    for i, exp in enumerate(all_selected_experiences, 1):
                        label = label_map.get(exp["type"], exp["type"])
                        experience_prompt += pc.idea_item_prefix.format(index=i, label=label)
                        print("=================================sample_order: ==================================\n", exp['sample_order'])

                        # 限制经验分析文本最多500个字符
                        analysis_text = exp["analysis"] if exp.get("analysis") else ""
                        if len(analysis_text) > 500:
                            analysis_text = analysis_text[:500] + "..."
                        experience_prompt += analysis_text

                        experience_prompt += "\n---\n\n"

                    # 若包含失败经验，追加参数预算提示，避免模型为修复越界而要求更多参数
                    if any(exp.get("type") == "None" for exp in all_selected_experiences):
                        max_params = (
                            self._prompt_ctx.max_param_count
                            if hasattr(self, "_prompt_ctx") and self._prompt_ctx is not None else None
                        )
                        if max_params is not None:
                            experience_prompt += (
                                f"Note: the evaluator passes exactly {max_params} trainable parameters "
                                f"(params[0]..params[{max_params - 1}]). "
                                "Keep every equation within this budget; do not request more parameters.\n"
                            )

                    # 将经验添加到原始内容中
                    content = experience_prompt + "\n\n" + content

            # 有p的几率进入以下代码：
            p = 0.5  # 设置执行概率为50%，你可以根据需要调整这个值

            if random.random() < p and os.path.exists(experience_file):
                print("use residual_analyze: True")

                residual_file = os.path.join(getattr(self, "_base_dir", "."), "residual_analyze.json")
                if os.path.exists(residual_file):
                    with open(residual_file, "r", encoding="utf-8") as f:
                        experiences = json.load(f)

                    # 提取最后一条信息
                    if experiences:
                        last_experience = experiences[-1]
                        last_analysis = last_experience.get("analysis", "")
                        last_sample_order = last_experience.get("sample_order", "unknown")
                        last_equation = last_experience.get("equation", "")
                        if last_equation is not None:
                            # 构建提示
                            experience_prompt = (
                                self._prompt_ctx.render_residual_block_title()
                                if hasattr(self, "_prompt_ctx") and self._prompt_ctx is not None
                                else pc.residual_block_title.format(problem=pc.problem_name_in_prompt)
                            )
                            if len(last_analysis) > 2000:
                                last_analysis = last_analysis[:2000] + "..."
                            experience_prompt += last_analysis
                            print("=================================sample_order: ==================================\n", last_sample_order)
                            # 将经验添加到原始内容中
                            content = experience_prompt + "\n\n" + content
                        else:
                            # 构建提示
                            experience_prompt = (
                                self._prompt_ctx.render_residual_block_title()
                                if hasattr(self, "_prompt_ctx") and self._prompt_ctx is not None
                                else pc.residual_block_title.format(problem=pc.problem_name_in_prompt)
                            )
                            if isinstance(last_analysis, list):
                                last_analysis = last_analysis[0] if last_analysis else ""
                            if len(last_analysis) > 2000:
                                last_analysis = last_analysis[:2000] + "..."
                            experience_prompt += last_analysis

                            # 将经验添加到原始内容中
                            content = experience_prompt + "\n\n" + content

        except Exception as e:
            print(f"加载经验数据时出错: {str(e)}")
            print("Error details:")
            traceback.print_exc()  # 输出详细的错误堆栈信息

        # 添加任务头
        if hasattr(self, "_prompt_ctx") and self._prompt_ctx is not None:
            head = self._prompt_ctx.render_head()
        else:
            head = pc.head_template.format(
                dependent=pc.dependent_name_in_prompt,
                problem=pc.problem_name_in_prompt,
                independent=pc.independent_name_in_prompt,
            )
        content = head + '\n' + content
        print_block("========================最终输入给大模型的content========================\n")
        print_block(content)
        return content


def _extract_body(sample: str, config: config_lib.Config) -> str:
    """
    Extract the function body from a response sample, removing any preceding descriptions
    and the function signature. Preserves indentation.
    ------------------------------------------------------------------------------------------------------------------
    Input example:
    ```
    This is a description...
    def function_name(...):
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    Output example:
    ```
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    If no function definition is found, returns the original sample.
    """
    # 提取 python 代码
    match = re.search(r'```([\s\S]*?)```', sample)
    if match:
        sample = match.group(1).strip()
    else:
        print("No python code found, returning original sample.")

    # 去除LLM回复中的python
    sample = sample.replace('python', '')

    # 检测缺少缩进的return语句，并加上缩进'    '
    if (sample[:6] == 'return'):
        sample = '    ' + sample
        return sample

    # 检测多一个缩进的return语句，并改成一个缩进'    '
    if (sample[:14] == '        return'):
        sample = sample.replace('        ', '    ')
        return sample

    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False

    for lineno, line in enumerate(lines):
        # find the first 'def' program statement in the response
        if (line[:3] == 'def'):
            func_body_lineno = lineno
            find_def_declaration = True
            break

    if find_def_declaration:
        # 统一处理：直接保留函数定义后的原始缩进与内容
        code = ''
        for line in lines[func_body_lineno + 1:]:
            code += line + '\n'
        return code

    return sample


class SamplingOrchestrator:
    """采样编排器：连续采样方程、评估并写入经验缓冲，支持断点续跑与并行 sampler。"""

    _global_samples_nums: int = 1

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
            prompt_ctx: pc.PromptContext | None = None,
            llm_client: LLMClient | None = None,
            llm_api: dict | None = None,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        # 覆盖模块级 API 配置（最小改动注入）
        if llm_api:
            try:
                global API_HOST, API_KEY, API_MODEL, MAX_TOKENS
                API_HOST = llm_api.get('host', API_HOST)
                API_KEY = llm_api.get('api_key', API_KEY)
                API_MODEL = llm_api.get('model', API_MODEL)
                MAX_TOKENS = llm_api.get('max_tokens', MAX_TOKENS)
            except Exception:
                pass
        self._prompt_ctx = prompt_ctx
        # 每个 sampler 克隆一份基础客户端，多线程并行时统计计数互不干扰
        self._llm_client = _clone_llm_client(llm_client) if llm_client else None

        # 采样、经验分析、残差分析各自使用独立 temperature 的客户端副本，
        # 避免原地修改同一个 llm_client 的 kwargs 互相覆盖（并覆盖 llm_summary.config 用户设置）。
        self._llm_client_experience = _clone_llm_client(
            llm_client,
            temperature=float(0.0),
            top_p=float(1.0),
            frequency_penalty=float(0.0),
        )
        self._llm_client_residual = _clone_llm_client(
            llm_client,
            temperature=float(0.4),
            top_p=float(0.9),
            frequency_penalty=float(0.1),
        )

        # 经验总结与残差分析组件
        self._summarizer = ExperienceSummarizer(
            self._llm_client_experience, prompt_ctx=self._prompt_ctx)
        self._analyzer = ResidualAnalyzer(
            self._llm_client_residual, prompt_ctx=self._prompt_ctx,
            results_root=config.results_root)

        # 传递上下文给 LLM，用于渲染指令与头部（生成方程组件 = Sampler）
        try:
            self._llm = llm_class(samples_per_prompt, prompt_ctx=self._prompt_ctx, llm_client=self._llm_client)
        except TypeError:
            # 向后兼容：旧实现不接收 prompt_ctx
            try:
                self._llm = llm_class(samples_per_prompt, prompt_ctx=self._prompt_ctx)
            except TypeError:
                self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config

    def sample(self, **kwargs):
        """ Continuously gets prompts, samples programs, sends them for analysis. """
        start_time = time.time()
        wall_limit = getattr(self.config, 'wall_time_limit_seconds', None)
        while True:
            # 基于总时长的退出：达到上限后优雅停止
            if wall_limit is not None and (time.time() - start_time) >= wall_limit:
                print(f'到达实验时长上限：{wall_limit} 秒，停止采样。')
                break

            # stop the search process if hit global max sample nums
            if self._max_sample_nums and self._get_global_sample_nums() >= self._max_sample_nums:
                break

            prompt = self._database.get_prompt()    # 从岛上拿一个可参考的方程框架 - 故可以独立反思

            island_id = prompt.island_id

            best_score = self._database._best_score_per_island[island_id]
            print(f"从岛屿 {island_id} 获取prompt，最佳分数: {best_score}")

            reset_time = time.time()

            print("调用大模型处理")

            # 向大模型采样出一个方程框架 - 核心
            samples, thinking_contents = self._llm.draw_samples(prompt.code, self.config)

            sample_time = (time.time() - reset_time) / self._samples_per_prompt

            print("获得了samples，在95行")
            print_block(samples)
            # This loop can be executed in parallel on remote evaluator machines.
            score_for_sample = []
            error_for_samlple = []
            quality_for_sample = []
            residual_data = None  # 用于存储每个样本的残差数据
            best_sample = None
            if_best = False
            id = 0
            temp_best_score = []
            for sample in samples:
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                score, error_msg, residual = chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )
                score_for_sample.append(score)
                error_for_samlple.append(error_msg)
                id += 1
                print(best_score)
                print(score)
                print('===================从chosen_evaluator.analyse中获得残差=====================\n')
                print_block(residual)
                if score is not None and score > best_score:
                    temp_best_score.append(score)
                    # 如果score比temp_best_score中的最大值大，就更新best
                    if score >= max(temp_best_score):
                        best_id = id
                        if_best = True
                        print("我在这里变成true了")
                        residual_data = residual
                        best_sample = sample
                        best_score_for_sample = score

            print("score_for_sample: ")
            print_block(score_for_sample)
            print("===========error_for_samlple:============================\n ")
            print_block(error_for_samlple)
            print("=========================residual_data: ================\n")
            print_block(residual_data)
            for each_score in score_for_sample:
                if each_score == None:
                    quality_for_sample.append('None')
                elif each_score > best_score:
                    quality_for_sample.append('Good')
                else:
                    quality_for_sample.append('Bad')

            print("quality_for_sample:")
            print('================================检查一下if_best的值====================\n')
            print(if_best)
            # 调用分析函数进行分析
            try:
                # 先直接进入第三次
                print_block("\n===== 方程和分数分析开始 =====")
                analysis_result = self._summarizer.analyze(
                    samples, quality_for_sample, error_for_samlple, prompt)
                print_block("总的分析结果：---------")
                print_block(analysis_result)
                print_block("===== 方程和分数分析结束 =====\n")

                # 添加第三次对话：残差分析
                print_block("\n===== 残差分析开始 =====")
                print_block(residual_data)
                print(if_best)
                if residual_data is not None and if_best:
                    # 只对有效样本进行残差分析
                    if_best = False
                    residual_result = self._analyzer.analyze(best_sample, residual_data)
                    print_block(f"样本残差分析结果: {residual_result}")
                    # 多线程下 residual_analyze.json 为读-改-写，需加锁防止丢失更新
                    with _SAMPLER_LOCK:
                        # 创建目录存放残差分析结果
                        json_residual_file = os.path.join(self.config.results_root or ".", "residual_analyze.json")

                        # 加载现有的残差分析数据（如果文件存在）
                        residual_data_list = []
                        if os.path.exists(json_residual_file):
                            try:
                                with open(json_residual_file, "r", encoding="utf-8") as f:
                                    existing_data = json.load(f)
                                    if isinstance(existing_data, list):
                                        residual_data_list = existing_data
                            except json.JSONDecodeError:
                                print(f"现有的残差分析JSON文件格式有误，将创建新文件")
                            except Exception as e:
                                print(f"读取现有残差分析文件时出错: {e}")

                        # 创建新的残差分析记录
                        current_sample_order = self._get_global_sample_nums() - len(samples) + best_id  # 获取当前样本的顺序号

                        # 创建残差分析数据结构
                        residual_record = {
                            "sample_order": current_sample_order,
                            "island_id": prompt.island_id,
                            "equation": best_sample,
                            "analysis": residual_result,
                            "best_score": best_score_for_sample,
                        }

                        # 添加到残差分析数据列表
                        residual_data_list.append(residual_record)

                        # 保存更新后的残差分析数据
                        try:
                            _atomic_write_json(json_residual_file, residual_data_list)
                            print(f"成功更新残差分析JSON文件: {json_residual_file}")
                        except Exception as e:
                            print(f"保存残差分析JSON文件时出错: {e}")

                # 创建目录存放分析结果
                # 多线程下 experiences.json 为读-改-写，需加锁防止丢失更新
                with _SAMPLER_LOCK:
                    json_experience_file = os.path.join(self.config.results_root or ".", "experiences.json")

                    # 加载现有的经验（如果文件存在）
                    experiences_data = {"None": [], "Good": [], "Bad": []}
                    if os.path.exists(json_experience_file):
                        try:
                            with open(json_experience_file, "r", encoding="utf-8") as f:
                                existing_data = json.load(f)
                                # 确保键存在
                                for key in ["None", "Good", "Bad"]:
                                    if key in existing_data:
                                        experiences_data[key] = existing_data[key]
                        except json.JSONDecodeError:
                            print(f"现有的 JSON 文件格式有误，将创建新文件")
                        except Exception as e:
                            print(f"读取现有经验文件时出错: {e}")

                    # 添加新经验
                    for i, (sample_text, quality, analysis, error_msg, thinking_content) in enumerate(zip(samples, quality_for_sample, analysis_result, error_for_samlple, thinking_contents)):
                        # 获取当前样本的信息
                        current_sample_order = self._get_global_sample_nums() - len(samples) + i + 1  # 计算当前样本的顺序号

                        # 确定分类
                        if quality == 'Good':
                            category = "Good"
                        elif quality == 'Bad':
                            category = "Bad"
                        else:  # 'None'
                            category = "None"

                        # 创建经验数据结构
                        experience = {
                            "island_id": prompt.island_id,
                            "analysis": analysis,
                            "sample_order": current_sample_order,  # 添加样本顺序号
                            "sample_time": sample_time,
                            "equation": sample_text,
                            "score": score_for_sample[i],

                            "thinking_content": thinking_content
                        }

                        # 对于 None 类型，添加错误信息
                        if category == "None" and error_msg:
                            experience["error"] = error_msg

                        # 添加到相应类别
                        experiences_data[category].append(experience)

                    # 保存更新后的经验数据
                    try:
                        _atomic_write_json(json_experience_file, experiences_data)
                        print(f"成功更新经验 JSON 文件: {json_experience_file}")
                    except Exception as e:
                        print(f"保存 JSON 经验文件时出错: {e}")
            except Exception as e:
                print(f"执行分析时出错: {e}")
                traceback.print_exc()

            # 每轮采样结束后保存 checkpoint（断点续跑）
            self._save_checkpoint()
            # 每轮采样结束后追加进度（可观测性）
            self._append_progress(island_id, start_time)

    def _get_global_sample_nums(self) -> int:
        with _SAMPLER_LOCK:
            return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        with _SAMPLER_LOCK:
            self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        with _SAMPLER_LOCK:
            self.__class__._global_samples_nums += 1

    def _checkpoint_path(self) -> str:
        return os.path.join(self.config.results_root or ".", "checkpoint.json")

    def _save_checkpoint(self):
        """保存当前经验缓冲 + 全局采样数到 checkpoint（断点续跑用）。"""
        try:
            with _SAMPLER_LOCK:
                self._database.save_checkpoint(self._checkpoint_path(), extra={
                    "global_sample_nums": self._get_global_sample_nums(),
                    "saved_at": time.time(),
                })
        except Exception as e:
            print(f"[WARN] 保存 checkpoint 失败: {e}")

    def _append_progress(self, island_id: int, start_time: float):
        """每轮采样结束后追加一行进度到 round_progress.csv（可观测性）。"""
        try:
            with _SAMPLER_LOCK:
                path = os.path.join(self.config.results_root or ".", "round_progress.csv")
                best = self._database._best_score_per_island[island_id]
                write_header = not os.path.exists(path)
                with open(path, "a", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(["timestamp", "wall_elapsed_s", "island_id",
                                         "best_score", "global_sample_nums"])
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        f"{(time.time() - start_time):.1f}",
                        island_id,
                        f"{best:.6f}" if best is not None and best != float('-inf') else "",
                        self._get_global_sample_nums(),
                    ])
        except Exception as e:
            print(f"[WARN] 写入 progress.csv 失败: {e}")
