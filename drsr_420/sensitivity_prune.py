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
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import sympy as sp


# ─────────────────────────────────────────────────────────────
# 数据类：记录与统计
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# 核心类
# ─────────────────────────────────────────────────────────────

class SensitivityPruner:
    """
    对 SymPy 表达式树进行基于敏感度的剪枝。

    Parameters
    ----------
    symbols      : 表达式中的自由变量列表，决定采样维度。
    threshold    : 敏感度阈值，≤ 该值时执行剪枝（默认 0.05）。
    num_samples  : 随机采样点数（默认 500）。
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
        if not symbols:
            raise ValueError("symbols 不能为空")
        if metric not in ("relative", "absolute"):
            raise ValueError("metric 须为 'relative' 或 'absolute'")
        if reduction not in ("max", "mean", "median", "p95"):
            raise ValueError("reduction 须为 'max'/'mean'/'median'/'p95'")

        self.symbols = list(symbols)
        self.threshold = threshold
        self.num_samples = num_samples
        self.sample_range = sample_range
        self.metric = metric
        self.reduction = reduction
        self.seed = seed

        rng = np.random.default_rng(seed)
        # shape: (num_samples, n_symbols)
        self._samples: np.ndarray = rng.uniform(
            sample_range[0], sample_range[1],
            size=(num_samples, len(symbols)),
        )
        # 各列单独切片，供 lambdify 调用
        self._pts: List[np.ndarray] = [
            self._samples[:, i] for i in range(len(symbols))
        ]
        self._cache: dict = {}       # lambdify 缓存
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
        self._cache.clear()

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
        self._greedy_remove(kept, neutral=sp.Integer(0),
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
            self._greedy_remove(kept, neutral=sp.Integer(1),
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
        kind         : 节点类型标签，用于日志/统计。
        depth        : 当前深度。
        skip_numbers : 若 True，跳过纯数字项（用于 Mul）。
        """
        # 构建父节点的工厂函数
        def build(items):
            if not items:
                return neutral
            if len(items) == 1:
                return items[0]
            return (sp.Add if neutral == sp.Integer(0) else sp.Mul)(*items)

        # 按各项在采样点上的贡献排序（贡献小的优先尝试）。
        # 用 nanmedian 抗离群点；若全部无效则视为 0 贡献（最后再尝试移除）。
        def contrib(e):
            v = self._evaluate(e)
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

            orig_vals = self._evaluate(parent_orig)
            pruned_vals = self._evaluate(parent_pruned)
            s = self._sensitivity(orig_vals, pruned_vals)
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

    # ── 求值 ─────────────────────────────────────────────────

    def _evaluate(self, expr: sp.Expr) -> np.ndarray:
        """
        在所有采样点批量求值表达式。
        优先 lambdify（NumPy 向量化），失败时逐点 subs 备用。
        结果缓存以避免重复编译同一表达式。
        """
        # 以规范化字符串为缓存键：结构等价（repr 相同）的表达式可复用求值结果，
        # 比 id(expr) 命中率更高（贪心循环中反复构造同类父节点）。
        key = repr(expr)
        if key in self._cache:
            return self._cache[key]

        result = self._eval_lambdify(expr)
        self._cache[key] = result
        return result

    def _eval_lambdify(self, expr: sp.Expr) -> np.ndarray:
        try:
            f = sp.lambdify(self.symbols, expr, modules="numpy")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = f(*self._pts)
            if np.ndim(out) == 0:
                return np.full(self.num_samples, float(out))
            return np.asarray(out, dtype=float)
        except Exception:
            return self._eval_slow(expr)

    def _eval_slow(self, expr: sp.Expr) -> np.ndarray:
        """逐点 subs 求值（备用，适用于 lambdify 无法处理的情形）。"""
        vals = []
        for pt in self._samples:
            sub = {sym: float(v) for sym, v in zip(self.symbols, pt)}
            try:
                v = complex(expr.subs(sub).evalf())
                vals.append(v.real if abs(v.imag) < 1e-9 else np.nan)
            except Exception:
                vals.append(np.nan)
        return np.array(vals, dtype=float)

    # ── 敏感度计算 ───────────────────────────────────────────

    def _sensitivity(self, orig: np.ndarray, pruned: np.ndarray) -> float:
        """
        计算 orig 与 pruned 之间的变化量作为敏感度。

        metric     : 'relative' → 逐点相对变化 |Δ|/(|orig|+1e-12)；
                     'absolute' → 逐点绝对变化 |Δ|。
        reduction  : 采样点上的聚合方式。
            max    : max_k |Δ_k|        （默认，最保守）
            mean   : mean_k |Δ_k|
            median : median_k |Δ_k|     （抗离群点）
            p95    : 95% 分位 |Δ_k|     （忽略罕见尖峰）
        """
        diff = np.abs(orig - pruned)
        if self.metric == "relative":
            denom = np.abs(orig) + 1e-12
            diff = diff / denom
        valid = diff[np.isfinite(diff)]
        if len(valid) == 0:
            return 0.0
        if self.reduction == "mean":
            return float(np.mean(valid))
        if self.reduction == "median":
            return float(np.median(valid))
        if self.reduction == "p95":
            return float(np.percentile(valid, 95))
        return float(np.max(valid))

    # ── 日志 ─────────────────────────────────────────────────

    def _log(self, depth: int, kind: str, expr: sp.Expr, s: float) -> None:
        if self._verbose:
            indent = "  " * depth
            print(f"{indent}✂ [{kind}] {expr}  (sensitivity={s:.2e})")


# ─────────────────────────────────────────────────────────────
# 顶层便捷函数
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# 演示
# ─────────────────────────────────────────────────────────────

def _run_demo() -> None:
    x, y, z = sp.symbols("x y z", real=True)
    eps = sp.Rational(1, 1000)
    SEP = "═" * 65

    cases = [
        (
            "案例 1 · 多项式：含极小系数项",
            x**3 + 2*x**2 + eps * x + eps**2,
            [x], 0.01,
            "预期：eps·x 和 eps²  被剪枝，保留 x³ + 2x²",
        ),
        (
            "案例 2 · 三角函数：含微小高频项",
            sp.sin(x) + sp.cos(x) + eps * sp.sin(50*x),
            [x], 0.05,
            "预期：eps·sin(50x) 被剪枝，保留 sin(x)+cos(x)",
        ),
        (
            "案例 3 · 多变量：含可忽略交叉项",
            x**2 + y**2 + z**2 + eps * x*y + eps**2 * x*y*z,
            [x, y, z], 0.02,
            "预期：两个交叉项被剪枝，保留 x²+y²+z²",
        ),
        (
            "案例 4 · 乘积：含接近 1 的小扰动因子",
            (1 + eps * x) * (x**2 + y**2) * sp.exp(-eps * y),
            [x, y], 0.05,
            "预期：(1+eps·x) 和 exp(-eps·y) 被剪枝，保留 x²+y²",
        ),
        (
            "案例 5 · 嵌套：sin 内部含小扰动",
            sp.sin(x + eps * y) + x**2 + eps**2 * z,
            [x, y, z], 0.01,
            "预期：eps²·z 被剪枝；sin 内部的 eps·y 视阈值可能被剪",
        ),
        (
            "案例 6 · 负面：所有项均重要，不应剪枝",
            x**2 + 2*x + 1,
            [x], 0.01,
            "预期：无项被剪枝",
        ),
    ]

    for title, expr, syms, thr, hint in cases:
        print(f"\n{SEP}\n  {title}\n{SEP}")
        print(f"  表达式 : {expr}")
        print(f"  阈值   : {thr}   ← {hint}\n")

        pruned, stats = sensitivity_prune(
            expr, syms, threshold=thr, verbose=True,
        )

        print(f"\n  原始 : {expr}")
        print(f"  剪枝 : {pruned}")
        print(f"  节点 : {stats.nodes_pruned} 已剪 / {stats.nodes_visited} 已访问  "
              f"({stats.prune_rate:.0%})\n")


if __name__ == "__main__":
    _run_demo()
