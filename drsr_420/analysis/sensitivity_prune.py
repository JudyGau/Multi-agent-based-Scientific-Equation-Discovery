"""
sensitivity_prune.py
=====================
对 SymPy 表达式树进行敏感度剪枝 (Sensitivity-based Pruning)。

算法原理
--------
遍历表达式树中每个可剪枝节点（Add 的加法项、Mul 的乘法因子），
对每个候选子表达式 t：

  1. 将 t 替换为"中性元"（Add → 0，Mul → 1），
     构造"移除后"的局部表达式。
  2. 在 num_samples 个随机采样点分别求值"移除前"与"移除后"。
  3. 计算最大相对变化（或绝对变化）作为敏感度 s。
  4. 若 s ≤ threshold → 执行剪枝（移除该子表达式）；
     否则保留，继续向下递归其子树。

贪心策略：在每个 Add/Mul 层，按平均绝对贡献从小到大依次判断，
优先尝试移除最不重要的项，避免因高敏感项的存在掩盖低敏感项。

模块分工（本文件只负责"遍历与决策"）
------------------------------------
* 求值与敏感度度量 → :mod:`drsr_420.analysis.expr_evaluation`（``ExpressionEvaluator``）；
* 剪枝记录与统计   → :mod:`drsr_420.analysis.prune_stats`（``PruneStats``）；
* 可视化演示       → ``python -m drsr_420.analysis.prune_demo``。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import sympy as sp

from drsr_420.analysis.expr_evaluation import ExpressionEvaluator
from drsr_420.analysis.prune_stats import PruneRecord, PruneStats

__all__ = ["SensitivityPruner", "sensitivity_prune", "PruneRecord", "PruneStats"]


class SensitivityPruner:
    """
    对 SymPy 表达式树进行基于敏感度的剪枝。

    Parameters
    ----------
    symbols      : 表达式中的自由变量列表，决定采样维度。
    threshold    : 敏感度阈值，≤ 该值时执行剪枝（默认 0.05）。
    num_samples  : 随机采样点数（默认 100）。
    sample_range : 各变量的均匀采样区间（默认 [-3, 3]）。
    metric       : 敏感度指标，'relative'（相对）或 'absolute'（绝对）。
    reduction    : 采样点上的聚合方式，'max'/'mean'/'median'/'p95'。
    seed         : 随机种子，保证可复现性。

    典型用法
    --------
    >>> x, y = sp.symbols('x y')
    >>> expr = x**2 + y**2 + sp.Rational(1,1000)*x*y
    >>> pruner = SensitivityPruner([x, y], threshold=0.01)
    >>> pruned = pruner.prune(expr, verbose=True)
    """

    def __init__(
        self,
        symbols: List[sp.Symbol],
        threshold: float = 0.05,
        num_samples: int = 100,
        sample_range: Tuple[float, float] = (-3.0, 3.0),
        metric: str = "relative",
        reduction: str = "max",
        seed: Optional[int] = 42,
    ) -> None:
        # 参数校验与采样网格都在求值器里（metric/reduction 只在这里被使用）
        self.evaluator = ExpressionEvaluator(
            symbols,
            num_samples=num_samples,
            sample_range=sample_range,
            metric=metric,
            reduction=reduction,
            seed=seed,
        )
        self.symbols = list(symbols)
        self.threshold = threshold
        self.stats: PruneStats = PruneStats()
        self._verbose: bool = False

    # ── 公共接口 ─────────────────────────────────────────────

    def prune(self, expr: sp.Expr, verbose: bool = False) -> sp.Expr:
        """
        对表达式 expr 执行敏感度剪枝，返回剪枝并化简后的结果。

        Parameters
        ----------
        expr    : 待剪枝的 SymPy 表达式。
        verbose : 是否输出逐步剪枝日志。

        Returns
        -------
        sp.Expr : 剪枝后的表达式（已 simplify）。
        """
        self.stats = PruneStats()
        self._verbose = verbose
        self.evaluator.clear_cache()

        result = self._prune_node(expr, depth=0)
        result = sp.simplify(result)

        if verbose:
            print("\n" + self.stats.summary())
        return result

    # ── 递归主逻辑 ───────────────────────────────────────────

    def _prune_node(self, node: sp.Expr, depth: int) -> sp.Expr:
        """递归地对当前节点执行剪枝，返回新（可能被剪枝的）节点。"""
        if node.is_Atom:                   # 叶节点：符号/数字
            return node
        if isinstance(node, sp.Add):
            return self._prune_add(node, depth)
        if isinstance(node, sp.Mul):
            return self._prune_mul(node, depth)
        # Pow、三角函数、exp 等：仅递归子节点
        return self._recurse_children(node, depth)

    def _prune_add(self, node: sp.Add, depth: int) -> sp.Expr:
        """
        处理 Add 节点：
        1. 先递归剪枝每个加法项内部的子树。
        2. 再在当前层尝试整体移除贡献最小的项（替换为 0）。
        """
        # Step 1: 递归剪枝各子项内部
        pruned_terms = [self._prune_node(t, depth + 1) for t in node.args]

        # Step 2: 贪心移除 —— 按贡献从小到大排序
        kept: List[sp.Expr] = list(pruned_terms)
        self._greedy_remove(kept, neutral=sp.Integer(0), reducer=sp.Add,
                            kind="term_of_Add", depth=depth)

        if not kept:
            return sp.Integer(0)
        return kept[0] if len(kept) == 1 else sp.Add(*kept)

    def _prune_mul(self, node: sp.Mul, depth: int) -> sp.Expr:
        """
        处理 Mul 节点：
        1. 先递归剪枝每个因子内部的子树。
        2. 再在当前层尝试移除贡献最小的非数字因子（替换为 1）。
        纯数字因子（系数）不参与移除（保留量纲/尺度）。
        """
        # Step 1: 递归剪枝
        pruned_factors = [self._prune_node(f, depth + 1) for f in node.args]

        # Step 2: 只对符号类因子尝试移除
        kept: List[sp.Expr] = list(pruned_factors)
        sym_factors = [f for f in kept if not f.is_number]

        if len(sym_factors) > 1:    # 至少保留一个符号因子
            self._greedy_remove(kept, neutral=sp.Integer(1), reducer=sp.Mul,
                                kind="factor_of_Mul", depth=depth,
                                skip_numbers=True)

        if not kept:
            return sp.Integer(1)
        return kept[0] if len(kept) == 1 else sp.Mul(*kept)

    def _recurse_children(self, node: sp.Expr, depth: int) -> sp.Expr:
        """对 Pow/函数等节点，仅递归处理子节点，不在此层尝试移除。"""
        new_args = [self._prune_node(arg, depth + 1) for arg in node.args]
        if tuple(new_args) == node.args:
            return node
        try:
            return node.func(*new_args)
        except Exception:
            return node     # 重建失败：安全回退

    # ── 贪心移除 ─────────────────────────────────────────────

    def _greedy_remove(
        self,
        kept: List[sp.Expr],
        neutral: sp.Expr,
        reducer,
        kind: str,
        depth: int,
        skip_numbers: bool = False,
    ) -> None:
        """
        就地修改 kept 列表，贪心地移除敏感度低的子表达式。

        Parameters
        ----------
        kept         : 当前保留的子表达式列表（原位修改）。
        neutral      : 中性元（Add → 0，Mul → 1）。
        reducer      : 父节点构造器（sp.Add 或 sp.Mul），与 neutral 配对。
        kind         : 节点类型标签，用于日志/统计。
        depth        : 当前深度。
        skip_numbers : 若 True，跳过纯数字项（用于 Mul）。
        """
        # 构建父节点的工厂函数：显式传入 reducer，避免靠 neutral == sp.Integer(0)
        # 这种隐式相等比较来推断构造器（脆弱且每次重建 Integer 对象）。
        def build(items):
            if not items:
                return neutral
            if len(items) == 1:
                return items[0]
            return reducer(*items)

        # 按各项在采样点上的贡献排序（贡献小的优先尝试）。
        # 用 nanmedian 抗离群点；若全部无效则视为 0 贡献（最后再尝试移除）。
        def contrib(e):
            v = self.evaluator.evaluate(e)
            valid = np.abs(v)[np.isfinite(v)]
            if len(valid) == 0:
                return 0.0
            return float(np.nanmedian(valid))

        order = sorted(range(len(kept)), key=lambda i: contrib(kept[i]))

        i = 0
        while i < len(order):
            idx = order[i]
            # 索引可能因前面的移除而偏移，需用实际元素查找
            if idx >= len(kept):
                i += 1
                continue
            candidate = kept[idx]

            # 跳过数字项
            if skip_numbers and candidate.is_number:
                i += 1
                continue

            # 至少保留一个非数字项（Mul 情况）或任意一项（Add 情况）
            non_num = [f for f in kept if not f.is_number]
            if skip_numbers and len(non_num) <= 1:
                break
            if not skip_numbers and len(kept) <= 1:
                break

            # 构造移除后的父节点
            others = [kept[j] for j in range(len(kept)) if kept[j] is not candidate]
            parent_orig = build(kept)
            parent_pruned = build(others)

            orig_vals = self.evaluator.evaluate(parent_orig)
            pruned_vals = self.evaluator.evaluate(parent_pruned)
            s = self.evaluator.sensitivity(orig_vals, pruned_vals)
            self.stats.nodes_visited += 1

            if s <= self.threshold:
                self._log(depth, kind, candidate, s)
                self.stats.nodes_pruned += 1
                self.stats.records.append(PruneRecord(kind, candidate, s, depth))
                kept.remove(candidate)
                # 重建 order（kept 长度变化，重新排序剩余项）
                order = sorted(range(len(kept)), key=lambda j: contrib(kept[j]))
                i = 0   # 重新从最小贡献项开始
            else:
                i += 1

    # ── 日志 ─────────────────────────────────────────────────

    def _log(self, depth: int, kind: str, expr: sp.Expr, s: float) -> None:
        if self._verbose:
            indent = "  " * depth
            # 不用剪刀等符号：GBK 控制台（Windows 默认代码页）编码不了，会直接抛
            # UnicodeEncodeError 打断剪枝流程。
            print(f"{indent}剪枝 [{kind}] {expr}  (sensitivity={s:.2e})")


def sensitivity_prune(
    expr: sp.Expr,
    symbols: List[sp.Symbol],
    threshold: float = 0.05,
    num_samples: int = 500,
    sample_range: Tuple[float, float] = (-3.0, 3.0),
    metric: str = "relative",
    reduction: str = "max",
    seed: Optional[int] = 42,
    verbose: bool = False,
) -> Tuple[sp.Expr, PruneStats]:
    """
    对 SymPy 表达式执行敏感度剪枝（顶层便捷函数）。

    Parameters
    ----------
    expr         : 待剪枝的 SymPy 表达式。
    symbols      : 表达式中的自由变量列表。
    threshold    : 相对/绝对敏感度阈值（默认 0.05，即 5%）。
    num_samples  : 随机采样点数（默认 500）。
    sample_range : 各变量的采样区间（默认 [-3, 3]）。
    metric       : 'relative'（相对误差）或 'absolute'（绝对误差）。
    reduction    : 采样点聚合方式 'max'/'mean'/'median'/'p95'（默认 'max'）。
    seed         : 随机种子（默认 42）。
    verbose      : 是否打印详细过程（默认 False）。

    Returns
    -------
    Tuple[sp.Expr, PruneStats]
        pruned_expr : 剪枝后的表达式。
        stats       : 剪枝统计信息对象（含 .summary() 方法）。

    Examples
    --------
    >>> x, y = sp.symbols('x y')
    >>> expr = x**2 + y**2 + sp.Rational(1, 1000)*x*y
    >>> pruned, stats = sensitivity_prune(expr, [x, y], threshold=0.01)
    >>> print(pruned)          # x**2 + y**2
    >>> print(stats.summary())
    """
    pruner = SensitivityPruner(
        symbols=symbols,
        threshold=threshold,
        num_samples=num_samples,
        sample_range=sample_range,
        metric=metric,
        reduction=reduction,
        seed=seed,
    )
    pruned = pruner.prune(expr, verbose=verbose)
    return pruned, pruner.stats


if __name__ == "__main__":
    from drsr_420.analysis.prune_demo import main

    main()
