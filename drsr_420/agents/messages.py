"""Agent 之间的消息与产物契约。

本模块把"Agent 之间传什么"从隐式约定（裸元组 + ``**kwargs`` 袋 + 平行列表对齐）
变成显式类型：读一个 dataclass 就知道跨 Agent 边界的字段、含义与落盘字段。

契约一览
========

::

    coordinator ──EvaluationRequest──▶ evaluator
                ◀──EvaluationOutcome──

    coordinator ──samples / qualities / errors──▶ experience_summarizer
                ◀──list[ExperienceEntry]────────  coordinator 补齐归属字段后落盘

    coordinator ──sample + residual────────────▶ residual_analyzer
                ◀──ResidualInsight─────────────  coordinator 补齐归属字段后落盘

    sampler ──────content──────────────────────▶ tool_caller
                ◀──responses / thinking─────────  内部逐条记录 ToolCall

    轮内批次容器：SampleBatch（CoordinatorAgent 一轮内的中间数据）

落盘字段由 ``to_json()`` 唯一决定：``experiences.json`` / ``residual_analyze.json``
的历史字段名在这里固化，避免"字段名靠两侧约定对齐"（历史实验数据与既有分析脚本
依赖这些字段名）。
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:      # 仅类型检查：避免 messages → core 的运行时依赖
    from drsr_420.core import buffer
    from drsr_420.core.profile import Profiler

#: 样本质量标签取值。
QUALITY_GOOD = "Good"
QUALITY_BAD = "Bad"
QUALITY_NONE = "None"
QUALITIES = (QUALITY_GOOD, QUALITY_BAD, QUALITY_NONE)


# ----------------------------------------------------------------------
# 采样 → 评估
# ----------------------------------------------------------------------
@dataclasses.dataclass
class EvaluationRequest:
    """CoordinatorAgent → EvaluatorAgent：请求评估一个骨架样本。

    取代原先的 ``analyze(sample, island_id, version_generated, **kwargs)``：
    ``profiler`` / ``global_sample_nums`` / ``sample_time`` 原本混在 kwargs 袋里
    一路透传到沙箱任务，字段含义与是否存在只能靠读实现反推。
    """

    sample: str
    island_id: int | None
    version_generated: int | None
    global_sample_nums: int | None = None
    sample_time: float | None = None
    profiler: "Profiler | None" = None


@dataclasses.dataclass
class EvaluationOutcome:
    """EvaluatorAgent → CoordinatorAgent：一次评估的结果。

    取代裸元组 ``(score, error, residual)``。
    """

    score: float | None
    error: str | None
    residual: Any | None = None

    def to_legacy(self) -> tuple[float | None, str | None, Any | None]:
        """转为旧的裸元组形式（过渡期兼容第三方调用方）。"""
        return self.score, self.error, self.residual

    def __iter__(self):
        """支持 ``score, error, residual = outcome`` 的旧式解包。"""
        return iter(self.to_legacy())


# ----------------------------------------------------------------------
# 反思 → 落盘（experiences.json / residual_analyze.json）
# ----------------------------------------------------------------------
@dataclasses.dataclass
class ExperienceEntry:
    """一条经验条目，对应 ``experiences.json`` 中的一个元素。

    职责划分：ExperienceSummarizerAgent 负责填 ``sample`` / ``quality`` / ``error`` /
    ``analysis``；CoordinatorAgent 负责补归属字段（``island_id`` / ``sample_order`` /
    ``sample_time`` / ``score`` / ``thinking_content``）后落盘。
    """

    sample: str
    quality: str
    analysis: str
    error: str | None = None
    score: float | None = None
    thinking_content: str = ""
    island_id: int | None = None
    sample_order: int | None = None
    sample_time: float | None = None

    def __post_init__(self) -> None:
        if self.quality not in QUALITIES:
            raise ValueError(f"quality 必须是 {QUALITIES} 之一，实际为 {self.quality!r}")

    def to_json(self) -> dict:
        """转为写入 ``experiences.json`` 的字典（字段名与历史产物保持一致）。"""
        record = {
            "island_id": self.island_id,
            "analysis": self.analysis,
            "sample_order": self.sample_order,
            "sample_time": self.sample_time,
            "equation": self.sample,
            "score": self.score,
            "thinking_content": self.thinking_content,
        }
        # 仅 None 类别携带错误信息（与历史产物一致）
        if self.quality == QUALITY_NONE and self.error:
            record["error"] = self.error
        return record


@dataclasses.dataclass
class ResidualInsight:
    """一条残差分析记录，对应 ``residual_analyze.json`` 中的一个元素。

    职责划分：ResidualAnalyzerAgent 负责填 ``sample`` / ``analysis``；
    CoordinatorAgent 负责补 ``island_id`` / ``sample_order`` / ``best_score`` 后落盘。
    """

    sample: str
    analysis: str
    island_id: int | None = None
    sample_order: int | None = None
    best_score: float | None = None

    def to_json(self) -> dict:
        """转为写入 ``residual_analyze.json`` 的字典（字段名与历史产物保持一致）。"""
        return {
            "sample_order": self.sample_order,
            "island_id": self.island_id,
            "equation": self.sample,
            "analysis": self.analysis,
            "best_score": self.best_score,
        }


# ----------------------------------------------------------------------
# 工具调用
# ----------------------------------------------------------------------
@dataclasses.dataclass
class ToolCall:
    """ToolCallerAgent 内的一次工具调用及其结果。

    Note:
        工具**执行器**的可注入契约仍是 ``executor(name, arguments) -> str``
        （测试用 ``lambda *a: "{}"`` 之类替身），本类型只负责把一次调用
        结构化地记录下来，便于日志与失败定位。
    """

    name: str
    arguments: dict
    result: str | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


# ----------------------------------------------------------------------
# 轮内批次容器
# ----------------------------------------------------------------------
@dataclasses.dataclass
class SampleBatch:
    """一轮采样产出的全部中间数据，在各子步骤（评估/分类/反思/持久化）间传递。

    字段顺序即数据流顺序：先有 prompt/samples，评估后填 scores/errors/qualities，
    反思后填 analyses，最后确定 best_*。
    """

    prompt: "buffer.Prompt"
    samples: list[str]
    thinking_contents: list[str]
    sample_time: float
    scores: list = dataclasses.field(default_factory=list)
    errors: list = dataclasses.field(default_factory=list)
    qualities: list = dataclasses.field(default_factory=list)
    #: 评估阶段逐样本领取的全局采样序号（与 samples/scores 平行、同序）。经验记录与
    #: 残差记录的归属序号一律取自这里，**不要**用"当前全局计数 - 本轮样本数"反推：
    #: 多岛并发时全局计数会被别的岛推进，反推会撞号、漏号（见 coordinator 的
    #: ``_captured_order``）。
    sample_orders: list = dataclasses.field(default_factory=list)
    #: 反思阶段产物：ExperienceSummarizerAgent 返回的 list[ExperienceEntry]。
    experience_entries: list = dataclasses.field(default_factory=list)
    best_sample: str | None = None
    best_residual: Any | None = None
    best_id: int | None = None        # 1-based，用于 sample_order 计算
    best_score: float | None = None


def check_alignment(samples: Sequence[Any], *parallel: Sequence[Any]) -> None:
    """校验平行列表长度一致，不一致时立即报错。

    历史问题：多份平行列表（samples / qualities / errors / analyses /
    thinking_contents）靠 ``zip`` 对齐，长度不一致会被静默截断成"少写几条经验"
    或"经验错配到别的样本上"。这里把静默错配变成显式异常。
    """
    expected = len(samples)
    for i, seq in enumerate(parallel):
        if len(seq) != expected:
            raise ValueError(
                f"平行列表长度不一致：samples={expected}，第 {i + 2} 个列表={len(seq)}"
                f"（会静默错配样本与经验，已拒绝）"
            )
