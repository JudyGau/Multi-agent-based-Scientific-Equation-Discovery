"""表达式求值与敏感度度量：在随机采样点上比较"改动前 / 改动后"的表达式。

角色归属
--------
敏感度剪枝（:mod:`drsr_420.analysis.sensitivity_prune`）的**数值内核**：
与"怎么遍历表达式树、剪哪一项"的决策逻辑正交，因此单独成模块。

职责
----
1. 按 ``seed`` 生成可复现的采样网格（``num_samples × len(symbols)``）；
2. 求值：优先 ``lambdify``（NumPy 向量化），失败时退回逐点 ``subs``；
   以 ``repr(expr)`` 为键缓存结果（贪心剪枝会反复构造结构相同的父节点）；
3. 敏感度：``metric`` 决定逐点变化量（相对/绝对），``reduction`` 决定跨采样点聚合
   （max/mean/median/p95）。

聚合方式的选择有实际含义：``max`` 最保守（任一采样点出现大偏差就保留该项），
``median`` 抗离群点，``p95`` 忽略罕见尖峰。
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
import sympy as sp


class ExpressionEvaluator:
    """在固定采样网格上求值 SymPy 表达式，并给出变更前后的敏感度。

    Parameters
    ----------
    symbols      : 表达式中的自由变量列表，决定采样维度。
    num_samples  : 随机采样点数（默认 100）。
    sample_range : 各变量的均匀采样区间（默认 [-3, 3]）。
    metric       : 敏感度指标，'relative'（相对）或 'absolute'（绝对）。
    reduction    : 采样点上的聚合方式，'max'/'mean'/'median'/'p95'。
    seed         : 随机种子，保证可复现性。

    采样点在建实例时一次性生成（``samples`` / ``points``），因此同一实例上
    所有求值都在同一组点上进行——这是"改动前后可比"的前提。
    """

    #: 支持的敏感度指标。
    METRICS = ("relative", "absolute")
    #: 支持的跨采样点聚合方式。
    REDUCTIONS = ("max", "mean", "median", "p95")

    def __init__(
        self,
        symbols: List[sp.Symbol],
        num_samples: int = 100,
        sample_range: Tuple[float, float] = (-3.0, 3.0),
        metric: str = "relative",
        reduction: str = "max",
        seed: Optional[int] = 42,
    ) -> None:
        if not symbols:
            raise ValueError("symbols 不能为空")
        if metric not in self.METRICS:
            raise ValueError("metric 须为 'relative' 或 'absolute'")
        if reduction not in self.REDUCTIONS:
            raise ValueError("reduction 须为 'max'/'mean'/'median'/'p95'")

        self.symbols = list(symbols)
        self.num_samples = num_samples
        self.sample_range = sample_range
        self.metric = metric
        self.reduction = reduction
        self.seed = seed

        rng = np.random.default_rng(seed)
        # shape: (num_samples, n_symbols)
        self.samples: np.ndarray = rng.uniform(
            sample_range[0], sample_range[1],
            size=(num_samples, len(self.symbols)),
        )
        # 各列单独切片，供 lambdify 调用
        self.points: List[np.ndarray] = [
            self.samples[:, i] for i in range(len(self.symbols))
        ]
        self._cache: dict = {}       # repr(expr) -> 求值结果

    def clear_cache(self) -> None:
        """清空求值缓存（新一轮剪枝开始时调用，避免跨轮持有大量数组）。"""
        self._cache.clear()

    # ── 求值 ─────────────────────────────────────────────────
    def evaluate(self, expr: sp.Expr) -> np.ndarray:
        """在所有采样点批量求值表达式。

        以规范化字符串为缓存键：结构等价（repr 相同）的表达式可复用求值结果，
        比 id(expr) 命中率更高（贪心循环中反复构造同类父节点）。
        """
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
                out = f(*self.points)
            if np.ndim(out) == 0:
                return np.full(self.num_samples, float(out))
            return np.asarray(out, dtype=float)
        except Exception:
            return self._eval_slow(expr)

    def _eval_slow(self, expr: sp.Expr) -> np.ndarray:
        """逐点 subs 求值（备用，适用于 lambdify 无法处理的情形，如 sp.zeta）。"""
        vals = []
        for pt in self.samples:
            sub = {sym: float(v) for sym, v in zip(self.symbols, pt)}
            try:
                v = complex(expr.subs(sub).evalf())
                vals.append(v.real if abs(v.imag) < 1e-9 else np.nan)
            except Exception:
                vals.append(np.nan)
        return np.array(vals, dtype=float)

    # ── 敏感度 ───────────────────────────────────────────────
    def sensitivity(self, orig: np.ndarray, pruned: np.ndarray) -> float:
        """计算 orig 与 pruned 之间的变化量作为敏感度。

        metric     : 'relative' → 逐点相对变化 |Δ|/(|orig|+1e-12)；
                     'absolute' → 逐点绝对变化 |Δ|。
        reduction  : 采样点上的聚合方式。
            max    : max_k |Δ_k|        （默认，最保守）
            mean   : mean_k |Δ_k|
            median : median_k |Δ_k|     （抗离群点）
            p95    : 95% 分位 |Δ_k|     （忽略罕见尖峰）

        **无有效采样点**（表达式在该区间求值不出任何有限值）时返回 ``inf``：这是
        "测不出来" 而不是 "敏感度为零"。判据是 ``s <= threshold``，把它当成 0 会让
        每一项都被判为可删——实测把最优样本整个剪成一个常数（NMSE 8.7e-07 → 6.81）。
        """
        diff = np.abs(orig - pruned)
        if self.metric == "relative":
            denom = np.abs(orig) + 1e-12
            diff = diff / denom
        valid = diff[np.isfinite(diff)]
        if len(valid) == 0:
            return float("inf")
        if self.reduction == "mean":
            return float(np.mean(valid))
        if self.reduction == "median":
            return float(np.median(valid))
        if self.reduction == "p95":
            return float(np.percentile(valid, 95))
        return float(np.max(valid))
