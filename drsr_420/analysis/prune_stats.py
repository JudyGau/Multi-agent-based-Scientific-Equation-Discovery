"""剪枝统计：记录每一次剪枝操作并汇总剪枝率。

角色归属
--------
敏感度剪枝的**可观测性产物**：``SensitivityPruner.stats`` 与顶层函数
``sensitivity_prune()`` 的返回值都是这里的类型，``summary()`` 直接面向日志。

单独成模块的理由：数据结构（记录 + 汇总 + 文本渲染）与剪枝算法互不依赖，
放在一起会让算法文件被"打印格式"淹没。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import sympy as sp


@dataclass
class PruneRecord:
    """单次剪枝操作的详细记录。"""
    node_type: str      # 'term_of_Add' 或 'factor_of_Mul'
    removed: sp.Expr    # 被移除的子表达式
    sensitivity: float  # 该次剪枝时的敏感度值
    depth: int          # 节点在树中的深度


@dataclass
class PruneStats:
    """剪枝过程的统计汇总。

    ``ops_before`` / ``ops_after`` / ``simplify_applied`` / ``simplified_expr`` 回答
    "这次剪枝到底做了什么"——只报 ``nodes_pruned`` 会让 0 项剪枝但被 ``simplify``
    通分的情形看起来像"剪枝改变了公式"（见 ``sensitivity_prune.prune`` 的返回值契约）。
    """
    nodes_visited: int = 0
    nodes_pruned: int = 0
    records: List[PruneRecord] = field(default_factory=list)
    ops_before: int = 0                  # 原表达式的运算节点数
    ops_after: int = 0                   # 最终返回表达式的运算节点数
    simplify_applied: bool = False       # 是否真的调用了 sp.simplify
    #: 诊断：0 项剪枝时 simplify 会给出的**不同**形式（与原文同形时留 None）
    simplified_expr: Optional[sp.Expr] = None

    @property
    def prune_rate(self) -> float:
        return self.nodes_pruned / self.nodes_visited if self.nodes_visited else 0.0

    @property
    def actually_pruned(self) -> bool:
        """是否**真正**移除了子表达式（0 项时为 False，此时公式沿用原式）。"""
        return self.nodes_pruned > 0

    def summary(self) -> str:
        bar = "─" * 58
        lines = [
            bar,
            f"  节点访问数 : {self.nodes_visited}",
            f"  节点剪枝数 : {self.nodes_pruned}",
            f"  剪枝率     : {self.prune_rate:.1%}",
            self._actual_pruning_line(),
        ]
        if self.records:
            lines += ["", "  已剪枝节点："]
            for r in self.records:
                lines.append(
                    f"    [depth={r.depth:2d}] {r.node_type:20s} "
                    f"sens={r.sensitivity:.2e}  removed: {r.removed}"
                )
        lines.append(bar)
        return "\n".join(lines)

    def _actual_pruning_line(self) -> str:
        """渲染"是否真的剪掉了项"一行：把 0 项剪枝与 simplify 形式重排区分开。"""
        if self.actually_pruned:
            return (f"  实际剪枝   : 是（移除 {self.nodes_pruned} 项，"
                    f"节点数 {self.ops_before} → {self.ops_after}）")
        note = ""
        if self.simplified_expr is not None:
            note = (f"；simplify 只会改写为等价形式（节点数 {self.ops_before} → "
                    f"{sp.count_ops(self.simplified_expr)}），故公式沿用原式")
        return f"  实际剪枝   : 否（未移除任何项{note}）"
