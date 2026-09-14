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

from drsr_420.core import buffer
from drsr_420.core import config as config_lib
from drsr_420.core import prompt_config as pc
from drsr_420.core.console import print_block
from drsr_420.llm import LLMClient
from drsr_420.llm.role_clients import RoleClients

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


#: round_progress.csv 的列定义（正常行 stop_reason 留空；早停终止行填原因）
_PROGRESS_COLUMNS = ["timestamp", "wall_elapsed_s", "island_id",
                     "best_score", "global_sample_nums", "stop_reason"]


def _captured_order(batch: SampleBatch, index: int) -> int:
    """取第 ``index`` 个样本（0-based）在评估阶段领到的全局采样序号。

    归属序号必须用**评估时捕获**的值，不能用"当前全局计数 - 本轮样本数 + index"
    反推：多岛并发下，全局计数在本轮评估与落盘之间还会被别的岛推进，反推出来的序号
    既会撞车也会漏号（实测一次运行：57 个序号承载了 132 条经验，其中 37 个序号重复、
    63 个序号一条经验都没有）。下游按 ``sample_order`` 匹配经验——提示词注入的新鲜度
    窗口、收尾的物理解释——序号错位就会取错条目，或像 ``explain_best_sample`` 那样
    一条都取不到。

    缺少捕获值时**显式报错**而不是退回反推：静默错配正是要修掉的那个 bug。
    """
    if index < len(batch.sample_orders):
        return batch.sample_orders[index]
    raise ValueError(
        f"样本缺少评估阶段捕获的 sample_order（第 {index + 1} 个；已捕获 "
        f"{len(batch.sample_orders)} 个）：该批次未经 _evaluate_batch 处理，"
        f"无法确定经验归属，拒绝静默错配。"
    )


def clone_llm_client(client, task=None, **kwargs_overrides):
    """基于现有 LLMClient 复制一份独立实例，使各实例 kwargs 互不影响。

    .. deprecated::
        已由 :class:`drsr_420.llm.roles.RoleClients` 取代——"哪个角色用哪套配置"
        现在由 ``config/agents.config.json`` 声明，克隆与参数注入统一走
        ``RoleClients.get(role)``（角色参数也不再写死在本模块的实参里）。
        保留本函数仅为兼容外部调用；库内代码请勿再使用。
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

    # ---- 收敛型早停的共享状态（类属性 + _SAMPLER_LOCK，多 sampler 线程任一判定
    # 停止、全体一起停；计数均为"全局批次"口径，与 _global_samples_nums 同理跨线程累加）----
    _early_stop_batches_done: int = 0
    _early_stale_batches: int = 0        # 距上次（容差内的）全局最优改进的全局批次数
    _early_failed_batches: int = 0       # 连续"所有样本评估失败"的全局批次数
    _early_last_global_best: float | None = None
    _early_stop_reason: str | None = None
    _early_stop_row_written: bool = False

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
            role_clients: Any | None = None,
            target_score: float | None = None,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._prompt_ctx = prompt_ctx

        # 三种用途（采样+工具调用 / 经验总结 / 残差分析）各用**独立**的客户端实例：
        # LLMClient.kwargs 是实例可变状态，共用一个实例会让生成参数互相覆盖
        # （历史上真的发生过——三者最后全变成 temperature=0.4）。
        #
        # 「哪个角色用哪套配置」由 config/agents.config.json 声明，本模块不再写死
        # 任何参数（原先 experience/residual 的 temperature 等直接写在这里的实参中）。
        # 未注入 role_clients 的调用方（测试、旧代码）由 RoleClients.single 兜底：
        # 只给一个客户端时所有角色共用它，参数仍按注册表注入。
        self._role_clients = (
            role_clients if role_clients is not None else RoleClients.single(llm_client)
        )
        self._llm_client = self._role_clients.get('sampling')
        self._llm_client_experience = self._role_clients.get('experience')
        self._llm_client_residual = self._role_clients.get('residual')

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

        # ---- 收敛型早停配置（全部默认关闭，预算型条件照常兜底）----
        # target_score 由 runtime 层按 var(outputs) 从 config.target_nmse 换算：
        # score = -MSE，NMSE = MSE/var(outputs) ⇒ target_score = -target_nmse·var。
        self._target_score = target_score
        self._early_stop_patience = config.early_stop_patience
        # warmup 缺省取岛屿数：保证每座岛至少轮到一次采样机会之前不判平台期
        # （实测一次运行曾连续 18 批无全局改进、之后才出现全场最优，patience
        # 与 warmup 都必须给足，否则会在探索低谷把好 run 掐死）。
        self._early_stop_min_batches = (
            config.min_batches_before_early_stop
            if config.min_batches_before_early_stop is not None
            else config.experience_buffer.num_islands)
        self._early_stop_max_failed = config.max_failed_batches
        # 类级计数器随新实验归零：同一进程里先后跑多个实验（或测试）时不串台
        self._reset_early_stop_state()

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def sample(self, profiler: Any = None) -> None:
        """运行"采样→评估→反思→持久化"主循环，直至预算上限（时长/样本数）
        或收敛型早停（目标 NMSE / 平台期 / 失败熔断）触发。

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
            # 收敛型早停计数：批次收尾时更新共享状态（目标/平台期/熔断）
            self._record_batch_for_early_stop(batch)

        # 循环结束后：若因早停退出，在进度 CSV 里留一行终止记录（原因可追溯）
        self._write_early_stop_row(start_time)

    def _should_stop(self, start_time: float) -> bool:
        """预算型条件（墙钟 / 样本数）与收敛型条件（早停）任一满足即停。"""
        wall_limit = getattr(self.config, 'wall_time_limit_seconds', None)
        if wall_limit is not None and (time.time() - start_time) >= wall_limit:
            print(f'到达实验时长上限：{wall_limit} 秒，停止采样。')
            return True
        if self._max_sample_nums and self._get_global_sample_nums() >= self._max_sample_nums:
            return True
        return self._early_stop_triggered()

    # ------------------------------------------------------------------
    # 收敛型早停：绝对目标（target_nmse）/ 平台期（patience）/ 失败熔断
    # ------------------------------------------------------------------
    @classmethod
    def _reset_early_stop_state(cls) -> None:
        with _SAMPLER_LOCK:
            cls._early_stop_batches_done = 0
            cls._early_stale_batches = 0
            cls._early_failed_batches = 0
            cls._early_last_global_best = None
            cls._early_stop_reason = None
            cls._early_stop_row_written = False

    def _global_best_score(self) -> float | None:
        """跨岛全局最优分（各岛 best 的最大值）；尚无有效分时返回 None。

        ``_best_score_per_island`` 在真实 ExperienceBuffer 里是 **list**（按下标
        取岛），测试替身里则常见 dict——两种容器都按"取全部值"兼容。
        """
        with _SAMPLER_LOCK:
            per_island = self._database._best_score_per_island
            vals = per_island.values() if isinstance(per_island, dict) else per_island
            bests = [s for s in vals
                     if isinstance(s, (int, float)) and s != float('-inf')]
            return max(bests) if bests else None

    def _early_stop_triggered(self) -> bool:
        """判定收敛型早停；命中时把原因写入共享状态并打印一次。"""
        with _SAMPLER_LOCK:
            cls = self.__class__
            if cls._early_stop_reason:
                return True    # 别的 sampler 线程已判停：跟随停止，不重复打印

            def _trigger(reason: str) -> bool:
                cls._early_stop_reason = reason
                print(f'[早停] {reason}')
                return True

            # ① 绝对目标：全局最优分已达到 target_nmse 换算出的目标分
            if self._target_score is not None:
                best = self._global_best_score()   # RLock 可重入，锁内调用安全
                if best is not None and best >= self._target_score:
                    return _trigger(f'全局最优 {best:.6g} 已达目标分 '
                                    f'{self._target_score:.6g}（target_nmse），停止采样。')
            # ② 失败熔断：连续 N 批全部样本评估失败（API 故障/解析崩坏不再烧预算）
            if (self._early_stop_max_failed is not None
                    and cls._early_failed_batches >= self._early_stop_max_failed):
                return _trigger(f'连续 {cls._early_failed_batches} 个批次所有样本评估失败'
                                '（熔断），停止采样。')
            # ③ 平台期：warmup 之后，连续 N 批无（容差内的）全局最优改进
            if (self._early_stop_patience is not None
                    and cls._early_stop_batches_done >= self._early_stop_min_batches
                    and cls._early_stale_batches >= self._early_stop_patience):
                return _trigger(f'连续 {cls._early_stale_batches} 个批次无全局最优改进'
                                '（平台期），停止采样。')
            return False

    def _record_batch_for_early_stop(self, batch: SampleBatch) -> None:
        """批次收尾时更新早停计数器（全局口径，跨 sampler 线程累加）。

        改进判定带容差：``best > prev + max(1e-12, 1e-9·|prev|)`` 才算真改进。
        没有容差的话，参数优化在 1e-13 量级的抖动（实测 run.err 里同一分数
        连续三次"increased"）会把平台期计数器永远清零，机制形同虚设。
        """
        with _SAMPLER_LOCK:
            cls = self.__class__
            cls._early_stop_batches_done += 1
            # 失败熔断计数：本批有任一有效分即视为系统健康、清零
            if any(s is not None for s in batch.scores):
                cls._early_failed_batches = 0
            else:
                cls._early_failed_batches += 1
            best = self._global_best_score()
            if best is None:
                return
            prev = cls._early_last_global_best
            tol = 0.0 if prev is None else max(1e-12, 1e-9 * abs(prev))
            if prev is None or best > prev + tol:
                cls._early_last_global_best = best
                cls._early_stale_batches = 0
            else:
                cls._early_stale_batches += 1

    def _write_early_stop_row(self, start_time: float) -> None:
        """因早停退出时在 round_progress.csv 追加一行终止记录（只写一次）。"""
        with _SAMPLER_LOCK:
            cls = self.__class__
            if not cls._early_stop_reason or cls._early_stop_row_written:
                return
            cls._early_stop_row_written = True
            reason = cls._early_stop_reason
        try:
            with _SAMPLER_LOCK:
                path = os.path.join(self.config.results_root or ".", "round_progress.csv")
                write_header = not os.path.exists(path)
                best = self._global_best_score()
                with open(path, "a", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(_PROGRESS_COLUMNS)
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        f"{(time.time() - start_time):.1f}",
                        "",    # island_id：终止行不属于任何岛
                        f"{best:.6g}" if best is not None else "",
                        self._get_global_sample_nums(),
                        reason,
                    ])
                print(f"[INFO] 早停原因已写入 {path}")
        except Exception as e:
            print(f"[WARN] 写入早停终止记录失败: {e}")

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
            # 序号在这里一次领取并随样本带走：多岛并发时全局计数随后还会被别的岛推进，
            # 落盘阶段再反推必然算错（见 _captured_order）。
            cur_global_sample_nums = self._next_global_sample_num()
            batch.sample_orders.append(cur_global_sample_nums)
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
                sample_order=_captured_order(batch, batch.best_id - 1),
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
                    sample_order=_captured_order(batch, i),
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

    @classmethod
    def _next_global_sample_num(cls) -> int:
        """自增并返回本次采样应使用的全局序号（**同一次加锁内完成**）。

        历史实现是 ``_global_sample_nums_plus_one()`` 之后再
        ``_get_global_sample_nums()``：两步各拿一次锁，两个岛线程可以在中间交错，
        于是**两个样本领到同一个序号**（实测一次运行里 37 个序号重复）。序号必须
        唯一，否则按 ``sample_order`` 取经验的下游会取错条目。
        """
        with _SAMPLER_LOCK:
            cls._global_samples_nums += 1
            return cls._global_samples_nums

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
                        writer.writerow(_PROGRESS_COLUMNS)
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        f"{(time.time() - start_time):.1f}",
                        island_id,
                        f"{best:.6f}" if best is not None and best != float('-inf') else "",
                        self._get_global_sample_nums(),
                        "",    # stop_reason：正常批次行留空，终止行由 _write_early_stop_row 写
                    ])
        except Exception as e:
            print(f"[WARN] 写入 progress.csv 失败: {e}")
