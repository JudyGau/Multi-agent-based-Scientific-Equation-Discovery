"""拟合后动态范围体检：角点钉扎/下溢尖峰类病理解的通用探测器。

背景（实测 MRFCompress-Cuboid_20260921-161549 及 45 次历史实验普查）：LLM 骨架
爱用"只在个别数据点非零"的局部化器件——大负指数幂（``2989.9*(λ12λ23)**-126``）、
窄高斯、下溢尖峰——去消化个别难拟合的数据点。训练点 MSE 完全看不见点与点之间的
行为，这类"训练分好看、域内行为是数值病理"的解因此一路通关（最极端实测：
grid_min=-1.5e6、grid_max=3.9e28）。

体检在**训练数据包围盒**的网格（均匀网格 + 各角点向域内的 1+δ 对数壳层）上评估
拟合后的方程，动态范围按 ``span_ratio = (grid_max - grid_min) / 输出跨度`` 归一。
同一份判据被两处消费（本模块只提供数值内核，避免多处各判一次）：

* ``evaluation/problems.evaluate``：拟合后对每个候选样本评分罚分（罚分 = 超出
  阈值的幅度），给采样阶段"别再造局部化器件"的正确信号；
* ``analysis/find_best_eq.prune_and_visualize``：收尾时对**最终发布**的表达式
  体检并写进剪枝摘要 → explain.md 权威「动态范围体检」小节。

为什么用体检罚分而不是"指数类参数设界"：参数位置无关，评估器无法泛化地知道
哪个参数是指数；且实测本数据上 |指数|≤50 仍可在山脊（λ12λ23≥3.92）之外局部化，
要 |指数|≤3 才能阻止——那会废掉合法幂律（λ^0.5、λ^2）。体检在评分层治理一切
形态的病理输出（尖峰、深谷、exp 爆炸、除零奇点），不侵入拟合本身。

本模块在 core 层：纯 numpy、无任何上层依赖——evaluation 与 analysis 都要用它，
而分层规则（tests/test_architecture.py）不允许 analysis 依赖 evaluation。
"""
from __future__ import annotations

import numpy as np

# ── 体检参数 ─────────────────────────────────────────────────────────
#: 体检网格单维点数上限。均匀网格按总点数预算在各维间分摊（总点数封顶见下），
#: 粗到便宜（每次评估多算几百个点，相对 least_squares 的开销可忽略），
#: 细到能撞进角点邻域的深谷。
RANGE_GRID_PER_AXIS = 24
#: 均匀网格总点数预算：每维点数 = min(RANGE_GRID_PER_AXIS, int(round(TOTAL**(1/d))))，
#: 仍超（高维）时退化为域内均匀随机 1000 点——网格尺寸不能随维度指数爆炸。
RANGE_GRID_TOTAL = 576
#: 每个角点的对数壳层数：在 2^d 个角点沿对角向域内偏移 δ·range（δ 对数取点）。
#: 深谷/尖峰器件在角点处的衰减可快于均匀网格步长，壳层保证撞见它们。
RANGE_SHELL_STEPS = 16
RANGE_SHELL_EPS = (1e-6, 1e-1)      # 角点壳层的相对偏移范围
#: 病理判定阈值：网格动态范围 / 数据输出跨度 超过该值即判病理并按超出量罚分。
#: 实测普查（45 次历史实验）：良性最优解 span_ratio ≤ 8，病理 ≥ 16（下溢尖峰
#: 钉扎 ≈38，精密插值 ≈1e5，溢出尖峰 ≥1e26），15 居间且两侧都有量级余量。
RANGE_SPAN_RATIO_LIMIT = 15.0
#: 罚分上限（JSON/下游比较安全，等效于"基本否决"该样本的评分）。
RANGE_PENALTY_CAP = 1e9

__all__ = [
    "RANGE_GRID_PER_AXIS", "RANGE_GRID_TOTAL", "RANGE_SHELL_STEPS",
    "RANGE_SHELL_EPS", "RANGE_SPAN_RATIO_LIMIT", "RANGE_PENALTY_CAP",
    "range_check_points", "dynamic_range_check",
]


def range_check_points(inputs: np.ndarray) -> np.ndarray:
    """动态范围体检的评估点：训练数据包围盒均匀网格 + 各角点向域内的对数壳层。"""
    lo = inputs.min(axis=0)
    hi = inputs.max(axis=0)
    rng = hi - lo
    d = inputs.shape[1]
    n_axis = min(RANGE_GRID_PER_AXIS, max(3, int(round(RANGE_GRID_TOTAL ** (1.0 / d)))))
    if n_axis ** d <= 2000:
        axes = [np.linspace(lo[j], hi[j], n_axis) for j in range(d)]
        pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, d)
    else:  # 高维兜底：总点数预算装不下的维度退化为域内均匀随机采样
        pts = np.random.default_rng(0).uniform(lo, hi, size=(1000, d))
    if d <= 6:  # 角点数 2^d 指数增长，很高维时壳层也放弃
        deltas = np.logspace(np.log10(RANGE_SHELL_EPS[0]),
                             np.log10(RANGE_SHELL_EPS[1]), RANGE_SHELL_STEPS)
        shells = []
        for bits in range(2 ** d):
            corner = np.array([hi[j] if (bits >> j) & 1 else lo[j] for j in range(d)])
            inward = np.array([-1.0 if (bits >> j) & 1 else 1.0 for j in range(d)])
            shells.append(corner + inward * deltas[:, None] * rng)
        pts = np.vstack([pts] + shells)
    return pts


def dynamic_range_check(inputs: np.ndarray, outputs: np.ndarray,
                        evaluate_fn) -> dict:
    """在训练数据包围盒网格上评估方程，返回动态范围诊断与病理罚分。

    Args:
        inputs: (n, d) 自变量训练数据。
        outputs: (n,) 因变量训练数据（提供数据尺度；常数输出时退化为
            ``max|outputs|``，再退化为 1e-12）。
        evaluate_fn: ``evaluate_fn(*columns) -> array``，输入按列拆开（与
            ``equation(*inputs.T, params)`` 的调用习惯一致；SymPy 表达式可用
            ``lambdify`` 结果直接传入，参数已代入则无需 params）。

    Returns:
        dict：``span_ratio``（网格动态范围/输出跨度，无有效网格点时为 inf）、
        ``grid_min`` / ``grid_max``（全非有限时为 None）/ ``penalty``（0 表示
        体检通过；罚分 = max(0, span_ratio - limit)，上限 RANGE_PENALTY_CAP）/
        ``limit`` / ``n_points``。
    """
    inputs = np.asarray(inputs, dtype=float)
    outputs = np.asarray(outputs, dtype=float)
    pts = range_check_points(inputs)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        try:
            vals = np.asarray(evaluate_fn(*pts.T), dtype=float)
        except Exception:
            vals = np.array([], dtype=float)
    vals = vals[np.isfinite(vals)]
    scale = float(outputs.max() - outputs.min())
    if scale <= 0:
        scale = max(float(np.max(np.abs(outputs))), 1e-12)
    if not len(vals):
        # 包围盒上整体不可用（处处溢出/NaN）：最重病理，按上限罚分
        return {"span_ratio": float("inf"), "grid_min": None, "grid_max": None,
                "penalty": float(RANGE_PENALTY_CAP),
                "limit": RANGE_SPAN_RATIO_LIMIT, "n_points": int(len(pts))}
    gmin, gmax = float(vals.min()), float(vals.max())
    span_ratio = (gmax - gmin) / scale
    excess = span_ratio - RANGE_SPAN_RATIO_LIMIT
    penalty = 0.0 if excess <= 0 else min(excess, RANGE_PENALTY_CAP)
    return {"span_ratio": float(span_ratio), "grid_min": gmin, "grid_max": gmax,
            "penalty": float(penalty), "limit": RANGE_SPAN_RATIO_LIMIT,
            "n_points": int(len(pts))}
