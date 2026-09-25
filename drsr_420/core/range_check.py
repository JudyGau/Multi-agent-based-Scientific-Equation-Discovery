"""拟合后动态范围体检：角点钉扎/下溢尖峰类病理解的通用探测器。

背景（实测 MRFCompress-Cuboid_20260921-161549 及 45 次历史实验普查）：LLM 骨架
爱用"只在个别数据点非零"的局部化器件——大负指数幂（``2989.9*(λ12λ23)**-126``）、
窄高斯、下溢尖峰——去消化个别难拟合的数据点。训练点 MSE 完全看不见点与点之间的
行为，这类"训练分好看、域内行为是数值病理"的解因此一路通关（最极端实测：
grid_min=-1.5e6、grid_max=3.9e28）。

三条判据（同一内核，罚分取其中最重的一条）
============================================
在**训练数据包围盒**的网格（均匀网格 + 各角点向域内的 1+δ 对数壳层）上评估
拟合后的方程：

1. **输出跨度** ``span_ratio = (grid_max - grid_min) / 输出跨度``。抓"把输出放大"
   的器件：溢出尖峰、深谷、除零奇点。
2. **局部斜率** ``slope_max = max |f(x+h) - f(x)| / (输出跨度 · h)``（逐维、朝包围盒
   中心偏移 ``h = RANGE_PROBE_REL × 该维 range``）。抓"在角点外**下溢消失**"的
   门控器件——它们不放大输出跨度，因此判据 1 结构上看不见。
3. **大系数抵消** ``coef_ratio = max|生效参数| / 输出跨度``。抓"输出是几个远大于
   输出的系数相减而来"的参数化——它既不放大跨度、也不产生陡斜率，前两条判据都判
   正常，但系数没有物理读数、且解会在近乎平坦的方向上漂到参数边界（详见
   :data:`RANGE_COEF_RATIO_LIMIT`）。

判据 2 的必要性（实测 MRFCompress-Cuboid_20260925-112514）：该实验的**发布最优解**
σ = 119.054 + 3.24323·λ12² + (35.1515 − 14.0837·λ12^−40.153)·λ23^0.8106
+ 49.6889·λ12^−40.153 的核心器件是 ``λ12^(−40.153)``：λ12=1 时为 1、1.017 时
0.5、2 时 8.2e-13、5 时 8.6e-29——**器件自身动态范围 1.16e28**，但它只在角点非零、
在域内其余位置下溢到 0，于是网格输出跨度只有数据跨度的 2.15 倍（判据 1 判"正常"，
罚分 0），而它确实是把 λ12=1 的两个数据点单独"分支"出来的角点钉扎器件。同一次
实验的 top-10 里 8 个带这种门控（判据 1 的 span_ratio 仅 1.17~8.25 全部通过，
判据 2 给出 4.0~34.4，发布解 16.5），而判据 1 抓到的 12 起全是放大型
（span_ratio 18.7~7.4e5）。两条判据合起来才覆盖"角点钉扎"这一族的两种数值形态。

判据 3 的必要性（实测 MRFCompress-Cuboid_20260925-134149）：该实验中段有 12 轮
无改进，top-10 里 9 个是同一族 ``P0·λ23^P1 + P2·λ12^P3 + P4``，其拟合参数
P0 ≈ 7.3e3~8.2e3 而数据输出跨度只有 159——即输出由两个 ~7e3 的项相消而来
（λ23 的指数同时退化为 0.0075，实际只是"近线性项"的另一种写法）。这些解
span_ratio 1.2、slope_max 0.3（判据 1/2 全过），却在同一次实验里留下 3 次
[BOUND] 贴边——参数被 ±1e4 边界截断。判据 3 把它们与健康解分开（健康 0.66~5.87、
该族 12.06~60.85，阈值取 8.0）。

同一份判据被两处消费（本模块只提供数值内核，避免多处各判一次）：

* ``evaluation/problems.evaluate``：拟合后对每个候选样本评分罚分（罚分 = 超出
  阈值的幅度），给采样阶段"别再造局部化器件"的正确信号；
* ``analysis/find_best_eq.prune_and_visualize``：收尾时对**最终发布**的表达式
  体检并写进剪枝摘要 → report.md 权威「动态范围体检」小节。

为什么用体检罚分而不是"指数类参数设界"：参数位置无关，评估器无法泛化地知道
哪个参数是指数；且实测本数据上 |指数|≤50 仍可在山脊（λ12λ23≥3.92）之外局部化，
要 |指数|≤3 才能阻止——那会废掉合法幂律（λ^0.5、λ^2）。体检在评分层治理一切
形态的病理输出（尖峰、深谷、exp 爆炸、除零奇点、门控），不侵入拟合本身。

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
#: 局部斜率探针的偏移量：``h = RANGE_PROBE_REL × 该维 range``（朝包围盒中心偏移，
#: 保证探针点仍在包围盒内）。1e-3 足以在门控（半衰期约为该维 range 的 4e-3）上
#: 取到 Δf ≈ 5% 输出跨度；再小会撞上浮点噪声，再大会跨过门控过渡区。
RANGE_PROBE_REL = 1e-3
#: 病理判定阈值：归一化局部斜率（"每单位自变量变化造成的输出变化 / 数据跨度"）
#: 超过该值即判病理并按超出量罚分。标定（MRFCompress-Cuboid 训练集，15 个形态：
#: 幂律/饱和/二次/陡幂 + 本次 2 个无门控解）：
#:   * 健康参数化 0.035 ~ 0.68（最陡的是 exp 饱和型与 l12² 型）；
#:   * 门控/局部化参数化 4.0 ~ 268。
#: 取 2.0 居间：距健康带上沿 2.9 倍、距门控带下沿 2 倍。
#: 注意"是否病理"取决于**拟合出来的参数化**而非骨架名义形式：实测同一数据集上
#: ``a*l12^b+c`` 的最小二乘最优是 b=-2739（把输出钉在 λ12=1 的尖峰），
#: ``a*(l12*l23)^b+c`` 的最优是 b=-877，斜率分别 59.8 / 188——被本判据拦下是
#: 期望行为（那些基线 NMSE 正是靠尖峰取得的），不是误报。
#: 另外罚分 = 超阈值的幅度（不是一票否决）：斜率略超阈值只带来 O(1) 罚分，
#: 与候选之间的 MSE 差同量级，起的是排序偏好作用。
RANGE_SLOPE_LIMIT = 2.0
#: 罚分上限（JSON/下游比较安全，等效于"基本否决"该样本的评分）。
RANGE_PENALTY_CAP = 1e9
#: 病理判定阈值：**真正影响输出的参数**里最大量级 / 数据输出跨度 超过该值即判病理。
#:
#: 抓的是"大系数抵消"型参数化——输出量级由两个远大于输出的系数相减得到。实测
#: 20260925-134149 的最优族 σ = −7208.13 + 7401.91·λ23^0.008287 + 3.2e-5·λ12^8.518
#: （两个 ~7400 的项相消出 ~300），次优族 σ = 1918.89 − 1725.93·λ12^−0.0376 + …。
#: 这类解在训练点上 MSE 可以很好，包围盒上的输出跨度与局部斜率也都正常（判据 1/2
#: 判"未检出"），但：各系数的物理读数毫无意义（"基线应力 −7208"、"λ23 指数 0.0083"），
#: 而且该族的 (P0, P1, 常数) 方向近乎平坦——实测把 P0 放大 7.3 倍、P1 反比缩小，
#: 包围盒内预测只变 0.4%——于是优化器沿平方向漂到 ±1e4 边界（该族样本 P0=8182，
#: 已达边界 82%，同一次实验另有 3 次 [BOUND] 贴边），参数由边界而非数据决定。
#:
#: 标定（MRFCompress-Cuboid 训练集，数据输出跨度 159.14；样本 = 两次实验的 top-10 +
#: 7 条候选骨架基线；只统计"扰动后预测会变的"参数，见 _active_params）：
#:   * 健康参数化 0.66 ~ 5.87（7 条基线 0.70 ~ 3.06，含幂律/二次/饱和/对数型）；
#:   * 大系数抵消族 12.06 ~ 60.85（本次 order 28/48/49/52/87/91、上一实验 order 15）。
#: 取 8.0 居间：距健康带上沿 1.36 倍、距病理带下沿 1.5 倍。
#: 罚分同样是超阈值的幅度（排序偏好，不是一票否决）。
RANGE_COEF_RATIO_LIMIT = 8.0

__all__ = [
    "RANGE_GRID_PER_AXIS", "RANGE_GRID_TOTAL", "RANGE_SHELL_STEPS",
    "RANGE_SHELL_EPS", "RANGE_SPAN_RATIO_LIMIT", "RANGE_SLOPE_LIMIT",
    "RANGE_COEF_RATIO_LIMIT", "RANGE_PROBE_REL", "RANGE_PENALTY_CAP",
    "range_check_points", "local_slope_check", "coefficient_cancellation_check",
    "dynamic_range_check",
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


def _evaluate_on(pts: np.ndarray, evaluate_fn) -> np.ndarray:
    """在给定网格点上向量化求值，返回与 ``pts`` 行列对齐的数组。

    约定与 :func:`dynamic_range_check` 的 ``evaluate_fn(*columns)`` 一致。以下情况
    返回空数组（调用方按"网格上整体不可用"处理或跳过该判据）：

    * 求值抛异常（方程在该区域越界/除零）；
    * 返回长度与点数不符（含标量返回被广播成常数——那不是缺失，见下）。

    标量/单元素返回按"处处同一常数"广播到位：常数模型是合法解（输出跨度 0），
    不能与"求值失败"混为一谈。
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        try:
            vals = np.asarray(evaluate_fn(*pts.T), dtype=float)
        except Exception:
            return np.array([], dtype=float)
    if vals.size == pts.shape[0]:
        return vals.reshape(-1)
    if vals.size == 1:      # 常数方程：广播成常数数组，跨度判据应判 0 而非失败
        return np.full(pts.shape[0], float(vals.reshape(-1)[0]))
    return np.array([], dtype=float)


def _output_scale(outputs: np.ndarray) -> float:
    """数据输出跨度（常数输出时退化为 max|outputs|，再退化为 1e-12）。"""
    scale = float(np.max(outputs) - np.min(outputs)) if outputs.size else 0.0
    if scale <= 0:
        scale = max(float(np.max(np.abs(outputs))) if outputs.size else 0.0, 1e-12)
    return scale


def local_slope_check(inputs: np.ndarray, outputs: np.ndarray,
                      evaluate_fn, *,
                      pts: np.ndarray | None = None,
                      base_values: np.ndarray | None = None) -> dict:
    """局部斜率判据：逐维朝包围盒中心偏移 ``h``，取归一化差商的最大值。

    器件形态决定它抓什么：门控型器件（如 ``λ12**(-40)``）只在角点非零、在域内其余
    位置**下溢到 0**，因此既不放大输出跨度、也不在均匀网格上留下痕迹；但它每跨过
    自己的过渡区就会让输出移动可观的一截——**差商**正是这个信号的量。归一化后
    "每单位自变量变化造成的输出变化 = 多少倍数据跨度"，与数据尺度无关。

    Args:
        inputs: (n, d) 自变量训练数据（提供包围盒与数据尺度）。
        outputs: (n,) 因变量训练数据（提供归一化跨度）。
        evaluate_fn: 同 :func:`dynamic_range_check` 的 ``evaluate_fn(*columns)``。
        pts: 复用的网格点（缺省按 :func:`range_check_points` 生成）。
        base_values: ``pts`` 上已算好的值（供 :func:`dynamic_range_check` 复用，
            避免重复求值）；形状须与 ``pts`` 对齐，未过滤非有限值。

    Returns:
        ``slope_max``（未测得时为 0.0）/ ``slope_limit`` / ``slope_penalty``。
        整片网格上求值失败时 ``slope_max`` 为 ``inf``、罚分取上限。
    """
    inputs = np.asarray(inputs, dtype=float)
    outputs = np.asarray(outputs, dtype=float)
    pts = range_check_points(inputs) if pts is None else np.asarray(pts, dtype=float)
    scale = _output_scale(outputs)

    base = (_evaluate_on(pts, evaluate_fn) if base_values is None
            else np.asarray(base_values, dtype=float))
    if base.size != pts.shape[0]:
        base = _evaluate_on(pts, evaluate_fn)
    if base.size != pts.shape[0]:
        return {"slope_max": float("inf"), "slope_limit": RANGE_SLOPE_LIMIT,
                "slope_penalty": float(RANGE_PENALTY_CAP)}

    lo, hi = inputs.min(axis=0), inputs.max(axis=0)
    center = 0.5 * (lo + hi)
    worst = 0.0
    for j in range(pts.shape[1]):
        span_j = float(hi[j] - lo[j])
        if span_j <= 0:      # 退化维（该列无变化）：没有可用于差商的方向
            continue
        h = RANGE_PROBE_REL * span_j
        moved = pts.copy()
        # 朝包围盒中心偏移 → 探针点必在包围盒内（贴面点也只会朝内走）
        moved[:, j] = pts[:, j] + np.where(pts[:, j] < center[j], h, -h)
        probe = _evaluate_on(moved, evaluate_fn)
        if probe.size != moved.shape[0]:
            continue
        both = np.isfinite(base) & np.isfinite(probe)
        if not both.any():   # 该方向两端都非有限：交给输出跨度判据（它按 inf 重罚）
            continue
        slope = np.abs(probe[both] - base[both]) / (scale * h)
        worst = max(worst, float(np.max(slope)))

    excess = worst - RANGE_SLOPE_LIMIT
    penalty = 0.0 if excess <= 0 else min(excess, RANGE_PENALTY_CAP)
    return {"slope_max": float(worst), "slope_limit": RANGE_SLOPE_LIMIT,
            "slope_penalty": float(penalty)}


def coefficient_cancellation_check(inputs: np.ndarray, outputs: np.ndarray, params,
                                   probe_fn=None) -> dict:
    """判据三：大系数抵消——真正影响输出的参数里最大量级 / 数据输出跨度。

    ``params`` / ``probe_fn`` 任一缺失时弃权（``coef_ratio=None``、罚分 0），
    不影响另外两条判据。

    Args:
        inputs: (n, d) 训练自变量（探测"哪些参数真的在起作用"用）。
        outputs: (n,) 训练因变量（提供归一化尺度）。
        params: 拟合出的参数向量（可能含未被方程使用的分量）。
        probe_fn: ``probe_fn(*columns, params) -> array``；判据三需要"换一组参数再算
            一次"的能力，已绑定参数的 ``evaluate_fn`` 做不到，故单独要求。

    Returns:
        ``coef_ratio`` / ``coef_limit`` / ``coef_penalty`` / ``coef_max`` /
        ``n_active_params``。
    """
    limit = RANGE_COEF_RATIO_LIMIT
    blank = {"coef_ratio": None, "coef_limit": limit, "coef_penalty": 0.0,
             "coef_max": None, "n_active_params": 0}
    if params is None or probe_fn is None:
        return blank
    x = np.asarray(params, dtype=float).reshape(-1)
    if not x.size:
        return blank
    cols = np.asarray(inputs, dtype=float).T
    with np.errstate(all="ignore"):
        try:
            base = np.asarray(probe_fn(*cols, x), dtype=float).reshape(-1)
        except Exception:
            return blank
    if base.size != len(outputs):
        return blank
    finite = base[np.isfinite(base)]
    if not finite.size:
        return blank
    tol = 1e-12 * max(1.0, float(np.max(np.abs(finite))))
    active = []
    for j in range(x.size):
        probe = x.copy()
        probe[j] += 1e-3 * max(1.0, abs(x[j]))
        with np.errstate(all="ignore"):
            try:
                alt = np.asarray(probe_fn(*cols, probe), dtype=float).reshape(-1)
            except Exception:
                active.append(j)     # 换这个参数直接算不出来：它显然在起作用
                continue
        if alt.size != base.size:
            active.append(j)
            continue
        both = np.isfinite(alt) & np.isfinite(base)
        if not both.any():
            active.append(j)     # 参数一动就整片非有限：同样在起作用
            continue
        if np.max(np.abs(alt[both] - base[both])) > tol:
            active.append(j)
    if not active:
        return blank
    coef_max = float(np.max(np.abs(x[active])))
    ratio = coef_max / _output_scale(outputs)
    excess = ratio - limit
    penalty = 0.0 if excess <= 0 else min(excess, RANGE_PENALTY_CAP)
    return {"coef_ratio": float(ratio), "coef_limit": limit,
            "coef_penalty": float(penalty), "coef_max": coef_max,
            "n_active_params": len(active)}


def dynamic_range_check(inputs: np.ndarray, outputs: np.ndarray,
                        evaluate_fn, *, params=None, probe_fn=None) -> dict:
    """在训练数据包围盒网格上评估方程，返回动态范围/局部斜率诊断与病理罚分。

    三条判据共用同一份网格与同一份求值结果（见模块 docstring），罚分取三者中
    更重的一条（不是相加：同一个器件常同时触发多条判据，相加会重复计罚）。

    Args:
        inputs: (n, d) 自变量训练数据。
        outputs: (n,) 因变量训练数据（提供数据尺度；常数输出时退化为
            ``max|outputs|``，再退化为 1e-12）。
        evaluate_fn: ``evaluate_fn(*columns) -> array``，输入按列拆开（与
            ``equation(*inputs.T, params)`` 的调用习惯一致；SymPy 表达式可用
            ``lambdify`` 结果直接传入，参数已代入则无需 params）。
        params: 拟合出的参数向量；与 ``probe_fn`` 一起给出时启用判据三（大系数
            抵消）。缺任一则判据三弃权。**不要**传"包含未使用分量的大向量"作为
            理由去放弃这条判据——判据内部会自行剔除不影响输出的参数。
        probe_fn: ``probe_fn(*columns, params) -> array``（未绑定参数的求值函数），
            判据三做扰动探测用。

    Returns:
        dict：``span_ratio``（网格动态范围/输出跨度，无有效网格点时为 inf）、
        ``grid_min`` / ``grid_max``（全非有限时为 None）、``slope_max``（归一化局部
        斜率）、``coef_ratio``（判据三，弃权时为 None）、``span_penalty`` /
        ``slope_penalty`` / ``coef_penalty``（各判据的罚分）、``penalty``
        （取最大，0 表示全部判据通过，上限 RANGE_PENALTY_CAP）、
        ``limit`` / ``slope_limit`` / ``coef_limit`` / ``n_points``。
    """
    inputs = np.asarray(inputs, dtype=float)
    outputs = np.asarray(outputs, dtype=float)
    coef = coefficient_cancellation_check(inputs, outputs, params, probe_fn)
    cap = float(RANGE_PENALTY_CAP)
    hard = {"coef_ratio": coef["coef_ratio"], "coef_limit": coef["coef_limit"],
            "coef_penalty": coef["coef_penalty"], "coef_max": coef["coef_max"],
            "n_active_params": coef["n_active_params"]}
    pts = range_check_points(inputs)
    raw = _evaluate_on(pts, evaluate_fn)
    scale = _output_scale(outputs)
    if not raw.size:
        # 包围盒上整体不可用（处处溢出/NaN）：最重病理，按上限罚分
        return {"span_ratio": float("inf"), "grid_min": None, "grid_max": None,
                "span_penalty": cap,
                "slope_max": float("inf"),
                "slope_limit": RANGE_SLOPE_LIMIT,
                "slope_penalty": cap,
                "penalty": cap,
                "limit": RANGE_SPAN_RATIO_LIMIT, "n_points": int(len(pts)), **hard}
    vals = raw[np.isfinite(raw)]
    if not vals.size:
        # 网格点全部非有限（与"求值异常"同性质）：最重病理
        return {"span_ratio": float("inf"), "grid_min": None, "grid_max": None,
                "span_penalty": cap,
                "slope_max": float("inf"),
                "slope_limit": RANGE_SLOPE_LIMIT,
                "slope_penalty": cap,
                "penalty": cap,
                "limit": RANGE_SPAN_RATIO_LIMIT, "n_points": int(len(pts)), **hard}
    gmin, gmax = float(vals.min()), float(vals.max())
    span_ratio = (gmax - gmin) / scale
    span_excess = span_ratio - RANGE_SPAN_RATIO_LIMIT
    span_penalty = 0.0 if span_excess <= 0 else min(span_excess, RANGE_PENALTY_CAP)

    slope = local_slope_check(inputs, outputs, evaluate_fn, pts=pts, base_values=raw)
    return {"span_ratio": float(span_ratio), "grid_min": gmin, "grid_max": gmax,
            "span_penalty": float(span_penalty),
            "slope_max": slope["slope_max"],
            "slope_limit": slope["slope_limit"],
            "slope_penalty": slope["slope_penalty"],
            "penalty": float(max(span_penalty, slope["slope_penalty"],
                                 coef["coef_penalty"])),
            "limit": RANGE_SPAN_RATIO_LIMIT, "n_points": int(len(pts)), **hard}
