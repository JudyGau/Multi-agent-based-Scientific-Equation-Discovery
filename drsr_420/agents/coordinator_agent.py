"""协调 Agent：驱动"采样→评估→反思→持久化"主循环，是唯一持有全链路编排权的角色。

角色：CoordinatorAgent 是"协调者"，每轮从共享记忆（ExperienceBuffer）取 prompt，
委托 SamplerAgent 采样骨架，再委托 EvaluatorAgent / ExperienceSummarizerAgent /
ResidualAnalyzerAgent 完成评估与反思，并把经验/残差/checkpoint 写回磁盘。

协作：
- 上游：pipeline.main 以线程方式启动本类（Sampler-i 线程），每个实例独享
  EvaluatorAgent 列表，共享 ExperienceBuffer 与全局采样计数（_SAMPLER_LOCK 保护）；
- 下游：SamplerAgent / EvaluatorAgent / ExperienceSummarizerAgent / ResidualAnalyzerAgent。

并行安全：
- 多 Sampler 线程共享的全局采样计数 _global_samples_nums（类属性）与
  experiences.json / residual_analyze.json / checkpoint.json 的读写均受
  模块级 _SAMPLER_LOCK（可重入锁）保护。
"""
from __future__ import annotations

import copy
import csv
import dataclasses
import json
import os
import threading
import time
import traceback
from typing import Any, Sequence, Type

import numpy as np

from drsr_420 import buffer
from drsr_420 import config as config_lib
from drsr_420 import prompt_config as pc
from drsr_420.console import print_block
from llm import LLMClient

from drsr_420.agents.sampler_agent import LLM, SamplerAgent
from drsr_420.agents.evaluator_agent import EvaluatorAgent
from drsr_420.agents.experience_summarizer_agent import ExperienceSummarizerAgent
from drsr_420.agents.residual_analyzer_agent import ResidualAnalyzerAgent

# 多 sampler 并行时保护共享文件读写与全局采样计数。
# 使用 RLock：内部方法 _get_global_sample_nums 会在 with _SAMPLER_LOCK 块内被再次调用，
# 普通 Lock 不可重入会导致同线程自锁死锁（Sampler 全部挂起）。
_SAMPLER_LOCK = threading.RLock()


def atomic_write_json(path: str, data) -> None:
    """原子写 JSON：先写临时文件再替换，避免并发读方读到半成品。"""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def clone_llm_client(client, task=None, **kwargs_overrides):
    """基于现有 LLMClient 复制一份独立实例，使各实例 kwargs 互不影响。

    原先对同一个 llm_client 连续执行 kwargs.update，导致采样/经验/残差三个用途
    的生成参数互相覆盖（最终全部变成 temperature=0.4），并覆盖了配置文件（如
    glm_glm-5.3-flash.config）中用户配置的温度。这里通过浅拷贝 + 独立 kwargs 字典修复。
    ``task`` 不为 None 时，先按任务注入配置声明的私有参数（如思考强度），
    再叠加 ``kwargs_overrides`` 覆盖（temperature 等）。
    """
    if client is None:
        return None
    if task is not None:
        new_client = client.clone_for_task(task)
    else:
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


@dataclasses.dataclass
class SampleBatch:
    """一轮采样产出的全部中间数据，在各子步骤（评估/分类/反思/持久化）间传递。"""
    prompt: buffer.Prompt
    samples: list[str]
    thinking_contents: list[str]
    sample_time: float
    scores: list = dataclasses.field(default_factory=list)
    errors: list = dataclasses.field(default_factory=list)
    qualities: list = dataclasses.field(default_factory=list)
    analyses: list = dataclasses.field(default_factory=list)
    best_sample: str | None = None
    best_residual: Any | None = None
    best_id: int | None = None        # 1-based，用于 sample_order 计算
    best_score: float | None = None


class CoordinatorAgent:
    """协调 Agent：连续采样方程、评估并写入经验缓冲，支持断点续跑与并行 sampler。"""

    _global_samples_nums: int = 1

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[EvaluatorAgent],
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
        self._prompt_ctx = prompt_ctx
        # 每个 sampler 克隆一份基础客户端，多线程并行时统计计数互不干扰。
        # 采样（骨架生成）+ 工具调用仅需轻量思考，思考强度在配置文件（如
        # glm_glm-5.3-flash.config）的 tasks.sampling 中声明（默认为 low），
        # 避免 max 强度下长时间推理阻塞并行采样。
        self._llm_client = clone_llm_client(llm_client, task='sampling') if llm_client else None

        # 采样、经验分析、残差分析各自使用独立 temperature 的客户端副本，
        # 避免原地修改同一个 llm_client 的 kwargs 互相覆盖（并覆盖配置文件用户设置）。
        # 思考强度分别从 tasks.experience / tasks.residual 声明（默认为 high）
        self._llm_client_experience = clone_llm_client(
            llm_client,
            task='experience',
            temperature=float(0.0),
            top_p=float(1.0),
            frequency_penalty=float(0.0),
        )
        # 残差分析需识别结构指导修正，使用 high 思考强度
        self._llm_client_residual = clone_llm_client(
            llm_client,
            task='residual',
            temperature=float(0.4),
            top_p=float(0.9),
            frequency_penalty=float(0.1),
        )

        # 经验总结与残差分析组件
        self._summarizer = ExperienceSummarizerAgent(
            self._llm_client_experience, prompt_ctx=self._prompt_ctx)
        self._analyzer = ResidualAnalyzerAgent(
            self._llm_client_residual, prompt_ctx=self._prompt_ctx,
            results_root=config.results_root)

        # 传递上下文给 LLM，用于渲染指令与头部（生成方程组件 = SamplerAgent）
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

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def sample(self, **kwargs) -> None:
        """运行"采样→评估→反思→持久化"主循环，直至达到全局采样数上限或实验时长上限。"""
        start_time = time.time()
        while not self._should_stop(start_time):
            prompt = self._database.get_prompt()    # 从岛上拿一个可参考的方程框架 - 故可以独立反思
            island_id = prompt.island_id
            best_score = self._database._best_score_per_island[island_id]  # 评估前捕获，不随评估更新
            print(f"从岛屿 {island_id} 获取prompt，最佳分数: {best_score}")

            batch = self._sample_batch(prompt)
            self._evaluate_batch(batch, best_score, **kwargs)
            self._classify_quality(batch, best_score)
            try:  # 分析失败不中断主循环（经验/残差分析均为增强项）
                self._summarize_experience(batch)
                self._analyze_residual(batch)
                self._persist_experiences(batch)
            except Exception as e:
                print(f"执行分析时出错: {e}")
                traceback.print_exc()

            # 每轮采样结束后保存 checkpoint（断点续跑）与进度（可观测性）
            self._save_checkpoint()
            self._append_progress(island_id, start_time)

    def _should_stop(self, start_time: float) -> bool:
        """达到实验时长上限或全局采样数上限时停止。"""
        wall_limit = getattr(self.config, 'wall_time_limit_seconds', None)
        if wall_limit is not None and (time.time() - start_time) >= wall_limit:
            print(f'到达实验时长上限：{wall_limit} 秒，停止采样。')
            return True
        if self._max_sample_nums and self._get_global_sample_nums() >= self._max_sample_nums:
            return True
        return False

    def _sample_batch(self, prompt: buffer.Prompt) -> SampleBatch:
        """从 LLM 采样一批方程骨架并封装为 SampleBatch（含平均采样耗时）。"""
        reset_time = time.time()
        print("调用大模型处理")
        # 向大模型采样出一个方程框架 - 核心
        samples, thinking_contents = self._llm.draw_samples(prompt.code, self.config)
        sample_time = (time.time() - reset_time) / self._samples_per_prompt
        print("获得了samples，在95行")
        print_block(samples)
        return SampleBatch(
            prompt=prompt,
            samples=samples,
            thinking_contents=thinking_contents,
            sample_time=sample_time,
        )

    def _evaluate_batch(self, batch: SampleBatch, best_score: float, **kwargs) -> None:
        """逐样本评估：全局计数 +1、随机选 evaluator 执行 analyse，追踪本轮最优样本。"""
        residual_data = None
        best_sample = None
        if_best = False
        id = 0
        temp_best_score = []
        for sample in batch.samples:
            self._global_sample_nums_plus_one()
            cur_global_sample_nums = self._get_global_sample_nums()
            chosen_evaluator: EvaluatorAgent = np.random.choice(self._evaluators)
            score, error_msg, residual = chosen_evaluator.analyse(
                sample,
                batch.prompt.island_id,
                batch.prompt.version_generated,
                **kwargs,
                global_sample_nums=cur_global_sample_nums,
                sample_time=batch.sample_time,
            )
            batch.scores.append(score)
            batch.errors.append(error_msg)
            id += 1
            print(best_score)
            print(score)
            print('===================从chosen_evaluator.analyse中获得残差=====================\n')
            print_block(residual)
            if score is not None and score > best_score:
                temp_best_score.append(score)
                # 如果score比temp_best_score中的最大值大，就更新best
                if score >= max(temp_best_score):
                    batch.best_id = id
                    if_best = True
                    print("我在这里变成true了")
                    residual_data = residual
                    best_sample = sample
                    batch.best_score = score
        batch.best_residual = residual_data
        batch.best_sample = best_sample

        print("score_for_sample: ")
        print_block(batch.scores)
        print("===========error_for_samlple:============================\n ")
        print_block(batch.errors)
        print("=========================residual_data: ================\n")
        print_block(residual_data)

    def _classify_quality(self, batch: SampleBatch, best_score: float) -> None:
        """按评估前 best_score 将每样本分为 Good/Bad/None。"""
        for each_score in batch.scores:
            if each_score == None:
                batch.qualities.append('None')
            elif each_score > best_score:
                batch.qualities.append('Good')
            else:
                batch.qualities.append('Bad')

        print("quality_for_sample:")
        print('================================检查一下if_best的值====================\n')
        print(batch.best_id is not None)

    def _summarize_experience(self, batch: SampleBatch) -> None:
        """委托 ExperienceSummarizerAgent 对整批样本做经验总结。"""
        print_block("\n===== 方程和分数分析开始 =====")
        batch.analyses = self._summarizer.analyze(
            batch.samples, batch.qualities, batch.errors, batch.prompt)
        print_block("总的分析结果：---------")
        print_block(batch.analyses)
        print_block("===== 方程和分数分析结束 =====\n")

    def _analyze_residual(self, batch: SampleBatch) -> None:
        """若本轮存在有效最优样本，委托 ResidualAnalyzerAgent 分析残差并持久化。"""
        print_block("\n===== 残差分析开始 =====")
        print_block(batch.best_residual)
        print(batch.best_id is not None)
        if batch.best_residual is not None and batch.best_id is not None:
            # 只对有效样本进行残差分析
            residual_result = self._analyzer.analyze(batch.best_sample, batch.best_residual)
            print_block(f"样本残差分析结果: {residual_result}")
            self._persist_residual(batch, residual_result)

    def _persist_residual(self, batch: SampleBatch, residual_result: str) -> None:
        """锁内读-改-写 residual_analyze.json，追加本轮最优样本的残差分析记录。"""
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

            # 当前样本顺序号：全局计数 - 本轮样本数 + 最优样本下标（1-based）
            current_sample_order = self._get_global_sample_nums() - len(batch.samples) + batch.best_id

            residual_record = {
                "sample_order": current_sample_order,
                "island_id": batch.prompt.island_id,
                "equation": batch.best_sample,
                "analysis": residual_result,
                "best_score": batch.best_score,
            }
            residual_data_list.append(residual_record)

            try:
                atomic_write_json(json_residual_file, residual_data_list)
                print(f"成功更新残差分析JSON文件: {json_residual_file}")
            except Exception as e:
                print(f"保存残差分析JSON文件时出错: {e}")

    def _persist_experiences(self, batch: SampleBatch) -> None:
        """锁内读-改-写 experiences.json，按 Good/Bad/None 分类追加本轮经验。"""
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
            for i, (sample_text, quality, analysis, error_msg, thinking_content) in enumerate(zip(
                    batch.samples, batch.qualities, batch.analyses, batch.errors, batch.thinking_contents)):
                current_sample_order = self._get_global_sample_nums() - len(batch.samples) + i + 1

                # 确定分类
                if quality == 'Good':
                    category = "Good"
                elif quality == 'Bad':
                    category = "Bad"
                else:  # 'None'
                    category = "None"

                experience = {
                    "island_id": batch.prompt.island_id,
                    "analysis": analysis,
                    "sample_order": current_sample_order,  # 添加样本顺序号
                    "sample_time": batch.sample_time,
                    "equation": sample_text,
                    "score": batch.scores[i],
                    "thinking_content": thinking_content,
                }

                # 对于 None 类型，添加错误信息
                if category == "None" and error_msg:
                    experience["error"] = error_msg

                experiences_data[category].append(experience)

            try:
                atomic_write_json(json_experience_file, experiences_data)
                print(f"成功更新经验 JSON 文件: {json_experience_file}")
            except Exception as e:
                print(f"保存 JSON 经验文件时出错: {e}")

    # ------------------------------------------------------------------
    # 全局采样计数（类属性，支持 pipeline 以类方式调用 set_global_sample_nums）
    # ------------------------------------------------------------------
    def _get_global_sample_nums(self) -> int:
        with _SAMPLER_LOCK:
            return self.__class__._global_samples_nums

    @classmethod
    def set_global_sample_nums(cls, num):
        with _SAMPLER_LOCK:
            cls._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        with _SAMPLER_LOCK:
            self.__class__._global_samples_nums += 1

    # ------------------------------------------------------------------
    # 断点续跑与可观测性
    # ------------------------------------------------------------------
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


# 兼容别名：旧模块名 drsr_420.sampler.SamplingOrchestrator 指向本类
SamplingOrchestrator = CoordinatorAgent
