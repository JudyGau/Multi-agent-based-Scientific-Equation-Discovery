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
from drsr_420.agents.base import (
    PIPELINE,
    THREAD_PER_SAMPLER,
    AgentSpec,
    BaseAgent,
)
from drsr_420.agents.messages import (
    QUALITY_BAD,
    QUALITY_GOOD,
    QUALITY_NONE,
    EvaluationOutcome,
    EvaluationRequest,
    ExperienceEntry,
    ResidualInsight,
    SampleBatch,
)

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


# SampleBatch 已迁至 drsr_420.agents.messages，并在本模块 re-export
# （保持 `from drsr_420.agents.coordinator_agent import SampleBatch` 可用）。


class CoordinatorAgent(BaseAgent):
    """协调 Agent：连续采样方程、评估并写入经验缓冲，支持断点续跑与并行 sampler。"""

    SPEC = AgentSpec(
        key="coordinator",
        role="协调者",
        mission="每轮从共享记忆取 prompt，驱动「采样→评估→反思→持久化」主循环",
        entrypoints=("sample",),
        upstream=(PIPELINE,),
        downstream=("sampler", "evaluator", "experience_summarizer", "residual_analyzer"),
        consumes=("buffer.Prompt",),
        produces=("SampleBatch",),
        artifacts=("checkpoint.json", "round_progress.csv",
                   "experiences.json", "residual_analyze.json"),
        thread_model=THREAD_PER_SAMPLER,
        llm_task="sampling",
        notes="唯一持有全链路编排权的角色；多实例共享 ExperienceBuffer 与全局采样计数，"
              "共享文件读写由可重入锁 _SAMPLER_LOCK 保护。",
    )

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
    def sample(self, profiler: Any = None) -> None:
        """运行"采样→评估→反思→持久化"主循环，直至达到全局采样数上限或实验时长上限。

        Args:
            profiler: ``profile.Profiler`` 实例（样本/进度记录）。参数原先是一个
                ``**kwargs`` 袋，而袋里唯一的内容就是它——现在显式声明。
        """
        start_time = time.time()
        while not self._should_stop(start_time):
            prompt = self._database.get_prompt()    # 从岛上拿一个可参考的方程框架 - 故可以独立反思
            island_id = prompt.island_id
            best_score = self._database._best_score_per_island[island_id]  # 评估前捕获，不随评估更新
            print(f"从岛屿 {island_id} 获取prompt，最佳分数: {best_score}")

            batch = self._sample_batch(prompt)
            self._evaluate_batch(batch, best_score, profiler)
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

    def _evaluate_batch(self, batch: SampleBatch, best_score: float,
                        profiler: Any = None) -> None:
        """逐样本评估：全局计数 +1、随机选 evaluator 执行 analyze，追踪本轮最优样本。

        best 追踪用单标量 round_best（O(n)）替代原先的 temp_best_score 列表 + max()
        （O(n^2)）；语义保持一致：仅严格超过评估前 best_score 的样本参与本轮最优
        评比，平局取后出现者（用 >= 比较）。
        """
        best_sample = None
        residual_data = None
        id = 0
        round_best = best_score  # 本轮已见最高分，初值=评估前阈值基准
        for sample in batch.samples:
            self._global_sample_nums_plus_one()
            cur_global_sample_nums = self._get_global_sample_nums()
            chosen_evaluator: EvaluatorAgent = np.random.choice(self._evaluators)
            outcome: EvaluationOutcome = chosen_evaluator.analyze(EvaluationRequest(
                sample=sample,
                island_id=batch.prompt.island_id,
                version_generated=batch.prompt.version_generated,
                global_sample_nums=cur_global_sample_nums,
                sample_time=batch.sample_time,
                profiler=profiler,
            ))
            score, error_msg, residual = outcome.score, outcome.error, outcome.residual
            batch.scores.append(score)
            batch.errors.append(error_msg)
            id += 1
            # 严格超过评估前最佳，且不劣于本轮已见最高分（平局取后出现者）
            if score is not None and score > best_score and score >= round_best:
                round_best = score
                batch.best_id = id
                residual_data = residual
                best_sample = sample
                batch.best_score = score
        batch.best_residual = residual_data
        batch.best_sample = best_sample

        # 批次摘要（替代原先散落的裸 print 调试噪声：best_score/score 裸打、长分隔线等）
        scores_preview = [round(s, 6) if isinstance(s, (int, float)) else s for s in batch.scores]
        print_block(
            f"[评估] 岛屿 {batch.prompt.island_id} 本轮 {len(batch.scores)} 个样本，"
            f"分数={scores_preview}，本轮最优 id={batch.best_id} score={batch.best_score}")

    def _classify_quality(self, batch: SampleBatch, best_score: float) -> None:
        """按评估前 best_score 将每样本分为 Good/Bad/None。"""
        for each_score in batch.scores:
            if each_score is None:
                batch.qualities.append(QUALITY_NONE)
            elif each_score > best_score:
                batch.qualities.append(QUALITY_GOOD)
            else:
                batch.qualities.append(QUALITY_BAD)

        # 质量分布摘要（替代原先的裸 print 噪声：quality_for_sample/if_best 等）
        print_block(
            f"[质量] Good={batch.qualities.count(QUALITY_GOOD)} "
            f"Bad={batch.qualities.count(QUALITY_BAD)} "
            f"None={batch.qualities.count(QUALITY_NONE)}，"
            f"本轮最优命中={batch.best_id is not None}")

    def _summarize_experience(self, batch: SampleBatch) -> None:
        """委托 ExperienceSummarizerAgent 对整批样本做经验总结。

        返回的是 :class:`ExperienceEntry` 列表（每条自带样本/质量/错误/分析文本），
        取代原先"平行 list[str] + 靠 zip 对齐"的隐式契约。
        """
        print_block("\n===== 方程和分数分析开始 =====")
        batch.experience_entries = self._summarizer.analyze(
            batch.samples, batch.qualities, batch.errors, batch.prompt)
        print_block("总的分析结果：---------")
        print_block([entry.analysis for entry in batch.experience_entries])
        print_block("===== 方程和分数分析结束 =====\n")

    def _analyze_residual(self, batch: SampleBatch) -> None:
        """若本轮存在有效最优样本，委托 ResidualAnalyzerAgent 分析残差并持久化。"""
        print_block("\n===== 残差分析开始 =====")
        print_block(batch.best_residual)
        if batch.best_residual is not None and batch.best_id is not None:
            # 只对有效样本进行残差分析
            insight: ResidualInsight = self._analyzer.analyze(
                batch.best_sample, batch.best_residual)
            print_block(f"样本残差分析结果: {insight.analysis}")
            self._persist_residual(batch, insight)

    def _persist_residual(self, batch: SampleBatch, insight: ResidualInsight) -> None:
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

            # 归属字段在此补齐：样本顺序号 = 全局计数 - 本轮样本数 + 最优样本下标（1-based）
            record = dataclasses.replace(
                insight,
                island_id=batch.prompt.island_id,
                sample_order=(self._get_global_sample_nums()
                              - len(batch.samples) + batch.best_id),
                best_score=batch.best_score,
            )
            residual_data_list.append(record.to_json())

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
            experiences_data = {QUALITY_NONE: [], QUALITY_GOOD: [], QUALITY_BAD: []}
            if os.path.exists(json_experience_file):
                try:
                    with open(json_experience_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                        # 确保键存在
                        for key in (QUALITY_NONE, QUALITY_GOOD, QUALITY_BAD):
                            if key in existing_data:
                                experiences_data[key] = existing_data[key]
                except json.JSONDecodeError:
                    print(f"现有的 JSON 文件格式有误，将创建新文件")
                except Exception as e:
                    print(f"读取现有经验文件时出错: {e}")

            # 每条经验自带样本/质量/分析文本（ExperienceEntry），不再 zip 5 个平行列表：
            # 历史实现里任一处顺序错位，都会把经验静默配到别的样本上。
            for i, entry in enumerate(batch.experience_entries):
                record = dataclasses.replace(
                    entry,
                    island_id=batch.prompt.island_id,
                    sample_order=self._get_global_sample_nums() - len(batch.samples) + i + 1,
                    sample_time=batch.sample_time,
                    score=batch.scores[i] if i < len(batch.scores) else None,
                    thinking_content=(batch.thinking_contents[i]
                                      if i < len(batch.thinking_contents) else ""),
                )
                experiences_data[record.quality].append(record.to_json())

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
