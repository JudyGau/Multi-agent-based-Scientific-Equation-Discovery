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
from typing import List

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
    """剪枝过程的统计汇总。"""
    nodes_visited: int = 0
    nodes_pruned: int = 0
    records: List[PruneRecord] = field(default_factory=list)

    @property
    def prune_rate(self) -> float:
        return self.nodes_pruned / self.nodes_visited if self.nodes_visited else 0.0

    def summary(self) -> str:
        bar = "─" * 58
        lines = [
            bar,
            f"  节点访问数 : {self.nodes_visited}",
            f"  节点剪枝数 : {self.nodes_pruned}",
            f"  剪枝率     : {self.prune_rate:.1%}",
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
