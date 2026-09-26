"""评估模块：对 LLM 生成的方程做参数优化并打分。

统一返回契约：
    evaluate() 成功返回 (score, result_matrix, optimized_params) 三元组；
    优化无法给出有限解（所有起点损失非有限但无异常）时返回 (None, None, None)。
    以下情况显式抛异常（供上层 remark 携带真实原因，喂给经验回路）：
      - 数据集本身含 NaN/inf 或维度非法（ValueError，配置错误应响亮的失败）；
      - 所有优化起点均以异常告终（透传首个真实异常，如方程越界 IndexError）。
    方程在某个参数点溢出/除零（LLM 常把参数当指数用）不算失败：残差会被清洗到
    ±RESIDUAL_CAP 之内后照常交给优化器，既不刷 RuntimeWarning 也不丢样本，详见
    RESIDUAL_CAP 与 _sanitize_residual。
    score 取负均方误差（越大越好）；result_matrix 为 (输入, 输出, 残差) 拼接矩阵，
    供残差分析回路消费（残差列保持全精度）；optimized_params 可直接作为下一轮
    优化的热启动起点。开体检时 **score = −(拟合 MSE + 罚分)**——罚分只加在评分上、
    不改残差，故要拿到"拟合 MSE"必须从残差列算（sandbox 就是这么做的，见
    _run_evaluation_task）：直接取 −score 会把罚分当成 MSE。
"""
from __future__ import annotations

import sys

import numpy as np

# 模块级默认配置，可通过 evaluate() 关键字参数覆盖
MAX_NPARAMS = 10                    # 方程参数个数
DECIMAL_PLACES = 3                  # 结果矩阵保留的小数位数（仅展示，不影响评分）
N_STARTS = 5                        # 多起点优化的起点数
MAX_ITER = 300                      # 每个起点的最大函数评估次数
# 参数边界，防止无界优化导致溢出/NaN。注释里的"历史 |p|≈738"已过时：实测
# MRFCompress-Cuboid_20260921-161549 的最优参数 |p| 达 2990（±10000 界内的角点
# 钉扎器件），故边界不能据历史分布收窄——病理输出由动态范围体检（下见
# dynamic_range_check）在评分层治理，这里保持宽界不干扰正常拟合。边界是否
# 真的在起作用由贴边监控（[BOUND] 日志，见 _report_bound_contact）持续积累
# 实证，为将来调整提供依据；起点量级覆盖见 _log_uniform_start。
PARAMS_BOUNDS = (-10000.0, 10000.0)
SAMPLE_SIZE = 100                   # 残差采样点数上限

# ── 拟合后动态范围体检（角点钉扎/下溢尖峰类病理解治理，20260921-161549）──
# 数值内核在 core 层（drsr_420.core.range_check）：analysis 收尾也要用同一判据，
# 而分层规则不允许 analysis 依赖 evaluation，故下沉到两层都合法的 core。
# 这里再导出保持"评分器即体检宿主"的可读性。
from drsr_420.core.range_check import (  # noqa: E402
    RANGE_COEF_RATIO_LIMIT,
    RANGE_GRID_PER_AXIS,
    RANGE_GRID_TOTAL,
    RANGE_PENALTY_CAP,
    RANGE_PROBE_REL,
    RANGE_SHELL_EPS,
    RANGE_SHELL_STEPS,
    RANGE_SLOPE_LIMIT,
    RANGE_SPAN_RATIO_LIMIT,
    coefficient_cancellation_check,
    dynamic_range_check,
    local_slope_check,
    range_check_points,
)

#: 残差幅值上限（清洗阈值），见 _sanitize_residual。
#:
#: LLM 写出的骨架常把参数当指数用（如 ``lambda12 ** params[2]``、
#: ``params[7] * np.exp(params[8] * x)``）。参数边界（``PARAMS_BOUNDS``，现值 ±10000）
#: 放宽后，优化器只要往边界方向探几步，指数就落到 1e4 量级，方程输出一路涨到
#: 1e50…1e300（还没到 inf 的区间）或直接溢出成 inf（0/0 则是 NaN）。旧实现把这些值
#: 原样递给 scipy，于是 trf 的信任域运算成片溢出——实测一个 scale=1e50 的残差就能刷出
#: 876 条 "overflow encountered in power/square"、"invalid value encountered in cast"
#: （trf.py:183/195/238/263、common.py:112/115/141/154/161/285/316/320/398），
#: 而这些行没有任何地方消费：
#:   * 评估跑在常驻 worker 子进程里（evaluation/sandbox.py），子进程的 stderr
#:     是继承来的控制台 fd，不经过主进程的 run.err tee —— 告警只在 IDE 控制台
#:     一闪而过，实验产物里查不到；
#:   * 真正该被记录的信息（哪个样本、哪组参数溢出）随之丢失，噪声却雪崩；
#:   * scale=1e200 一档干脆让 scipy 抛 ValueError，样本白白丢掉。
#: 因此这里把残差**截断**到 ±cap（非有限值按 ±cap 填充），优化器只看到
#: "这片区域极差"，信任域会自己缩步长离开。
#:
#: 取值 1e12 的理由：cap² = 1e24、scipy 内部 (cap²)³ = 1e72，离 float64 上限
#: （≈1.8e308）还有十几个数量级——不会把溢出从残差搬到 scipy 内部（这正是旧
#: 行为刷警告的机制）；同时比 MRF 类数据（输出 ~1e2）大 10 个数量级，任何被
#: 截断的残差都已经是"极差拟合"，区分度本来就无意义。
RESIDUAL_CAP = 1e12

#: 上限相对数据尺度的倍数：``cap = max(RESIDUAL_CAP, RESIDUAL_CAP_RATIO * max|outputs|)``。
#: 绝对下限 1e12 对本站所有问题都够用；倍数项只为"输出本身极大"的数据集兜底
#: （否则截断会把正常拟合也压平）。1e6 倍数据尺度后 (cap²)³ 仍在 1e100 量级内。
RESIDUAL_CAP_RATIO = 1e6


def _clamp_params(x0: np.ndarray, bounds) -> np.ndarray:
    """把初始参数裁剪进边界（least_squares 要求初始点在边界内）。"""
    lower, upper = bounds
    return np.clip(x0, lower, upper)


def residual_cap(outputs: np.ndarray) -> float:
    """按数据尺度算出残差幅值上限（见 RESIDUAL_CAP / RESIDUAL_CAP_RATIO）。"""
    scale = float(np.max(np.abs(outputs))) if np.size(outputs) else 0.0
    if not np.isfinite(scale):
        return RESIDUAL_CAP
    return max(RESIDUAL_CAP, RESIDUAL_CAP_RATIO * scale)


def _sanitize_residual(res: np.ndarray, cap: float = RESIDUAL_CAP) -> np.ndarray:
    """把残差截断到 ±cap，非有限值（NaN/±inf）按同号 cap 填充。

    保留 ±inf 的符号：正/负方向的溢出对优化器应是相反方向的推力；NaN 没有方向
    可言，一律取正号（幅值足够大，只起"离开该区域"的作用）。截断而非只填 inf，
    是因为**有限但巨大**的残差（exp 型方程在半溢出区间的输出）同样会击穿
    scipy 的信任域运算——这才是实测刷屏最多的那一类告警。
    """
    if res.size == 0 or (np.isfinite(res).all() and np.abs(res).max() <= cap):
        return res  # 常规路径：原样返回，不复制数组
    return np.clip(np.nan_to_num(res, nan=cap, posinf=cap, neginf=-cap), -cap, cap)


def _log_uniform_start(rng: np.random.Generator, n_params: int, bounds) -> np.ndarray:
    """对数均匀随机起点：每个分量的量级在 [1e-3, max|bound|] 十进位指数上均匀、符号随机。

    为什么不用 U(-1,1)：随机起点全挤在 O(1)，而合法系数的量级由骨架形式决定
    （如 ``y = p0*exp(p1*x)`` 里的 p0），可以是 1e2~1e4——旧口径下这些系数要靠
    信任域在 max_nfev 步内逐步"走"上去，拟合经常半途而废。对数均匀让起点直接
    覆盖各量级。代价：部分起点的大参数会把方程推入溢出区（指数用法 p>709/|x|
    时 exp 溢出），这些起点会被"起点预检"整体跳过——用少量起点换量级覆盖；
    大参数撞出的病理尖峰本就由动态范围体检罚分在评分层治理，不在起点层设防。
    """
    lower, upper = bounds
    mag_max = max(abs(float(lower)), abs(float(upper)))
    hi = float(np.log10(mag_max)) if mag_max > 0 else -3.0
    lo = min(-3.0, hi)  # 极小边界（max|bound| ≤ 1e-3）时退化为在边界幅值处取值
    mags = 10.0 ** rng.uniform(lo, hi, size=n_params)
    signs = rng.integers(0, 2, size=n_params) * 2.0 - 1.0
    return mags * signs


def _multi_start_least_squares(
        residual_func,
        n_params: int,
        *,
        n_starts: int = N_STARTS,
        max_iter: int = MAX_ITER,
        bounds=PARAMS_BOUNDS,
        x0: np.ndarray | None = None,
        seed: int | None = None,
        initial_residual_func=None,
) -> tuple[np.ndarray | None, float]:
    """多起点最小二乘求解，返回 (最优参数, 最小均方误差)；全部失败时返回 (None, inf)。

    相比 BFGS：least_squares 利用残差结构求 Jacobian，收敛更快更稳；
    带参数边界可避免无界优化使方程参数发散（产生 NaN/inf）。

    Args:
        residual_func: 交给 scipy 的残差函数，**契约是返回值有界**（超出
            ±RESIDUAL_CAP 的部分已在上层截断，见 _sanitize_residual）。
        initial_residual_func: 未清洗的残差函数，仅用于"起点是否可用"预检；缺省时
            直接复用 residual_func（此时预检退化为按清洗后的值判断）。
    """
    from scipy.optimize import least_squares

    rng = np.random.default_rng(seed)
    starts: list[np.ndarray] = []
    if x0 is not None:  # 热启动：把上一轮最优参数作为额外首起点（在 n_starts 随机起点之外）
        starts.append(np.asarray(x0, dtype=float))
    # 随机起点混合口径：首个保持 U(-1,1)——对系数 O(1) 的多数骨架仍是最稳的起点，
    # 也保证至少一个起点不因大参数溢出被起点预检整体浪费；其余对数均匀铺量级
    # （见 _log_uniform_start），让合法的大尺度系数不用靠信任域"走"过去。
    for i in range(n_starts):
        if i == 0:
            starts.append(rng.uniform(-1.0, 1.0, size=n_params))
        else:
            starts.append(_log_uniform_start(rng, n_params, bounds))

    probe_func = initial_residual_func or residual_func
    best_x, best_loss = None, np.inf
    first_exc: Exception | None = None
    for start in starts:
        start = _clamp_params(start, bounds)
        try:
            # 起点预检：残差全为非有限说明该参数点下方程整体不可用，此时既没有
            # 优化方向也谈不上拟合，必须把真实原因报给经验回路。scipy 自己只在
            # 初始点做这一步校验，而我们已经把非有限值清洗成有限大数（故意的），
            # 它再也不会发现——所以用未清洗的残差在这里显式复刻该契约
            # （错误文案与 scipy 一致，便于历史日志比对）。
            initial = np.asarray(probe_func(start), dtype=float)
            if not np.isfinite(initial).any():
                raise ValueError('Residuals are not finite in the initial point.')
            result = least_squares(
                residual_func,
                start,
                bounds=bounds,
                max_nfev=max_iter,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
            )
        except Exception as e:
            # 该起点失败（如方程在该参数域不可用/越界索引），尝试下一个起点，
            # 但保留首个真实异常：若所有起点都以异常告终，向上抛出。旧实现把
            # 异常彻底吞掉，LLM 生成的方程里 `params[10]` 越界之类的错误全部
            # 伪装成无信息量的 'no output'，经验学习回路（error 注入提示词）
            # 因此永远学不到任何东西。
            if first_exc is None:
                first_exc = e
            continue
        loss = float(np.mean(np.square(result.fun)))
        if not np.isfinite(loss):
            continue
        if loss < best_loss:
            best_loss, best_x = loss, result.x
    if best_x is None and first_exc is not None:
        raise first_exc
    return best_x, best_loss


def _report_bound_contact(best_x: np.ndarray, bounds) -> None:
    """贴边监控（只记日志，不影响评分与优化）：最优参数分量触及边界 ±0.1% 时打标。

    长期积累"边界是否真的在起作用"的实证，为将来调整 PARAMS_BOUNDS 提供依据：
    合法拟合频繁贴边 = 边界过窄的信号；只有病理解贴边 = 宽界在放大病理（治理
    靠动态范围体检罚分，不靠收窄边界）。

    输出走 stderr 而非 stdout：采样评估跑在常驻 worker 子进程里，只有 stderr
    会被旁路进实验的 run.err（sandbox.attach_worker_stderr），stdout 的 print
    只上控制台、落不进实验产物。下面的 [RANGE] 输出同理。
    """
    x = np.asarray(best_x, dtype=float)
    lower, upper = bounds
    tol = 1e-3  # 距边界 0.1%（相对边界幅值）以内视为贴边
    at_lower = x <= lower + tol * abs(lower)
    at_upper = x >= upper - tol * abs(upper)
    hit = np.flatnonzero(at_lower | at_upper)
    if not hit.size:
        return
    detail = ", ".join(
        f"p[{i}]={x[i]:+.4g}({'L' if at_lower[i] else 'U'})" for i in hit)
    print(f"[BOUND] 最优参数贴边 {hit.size}/{x.size}：{detail}", file=sys.stderr)


def evaluate(
        data: dict,
        equation,
        *,
        n_params: int = MAX_NPARAMS,
        decimal_places: int = DECIMAL_PLACES,
        n_starts: int = N_STARTS,
        max_iter: int = MAX_ITER,
        bounds=PARAMS_BOUNDS,
        x0: np.ndarray | None = None,
        seed: int | None = None,
        verbose: bool = False,
        range_check: bool = True,
) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
    """对 `equation(*X.T, params)` 做参数优化并评分。

    Args:
        data: 含 'inputs' 与 'outputs' 的数据字典。
        equation: 可调用对象，签名 equation(*feature_arrays, params)。
        x0: 热启动参数（例如上一轮评估得到的最优参数），作为首个优化起点。
        range_check: 拟合后是否做动态范围体检并对病理解罚分（见
            :func:`dynamic_range_check`；默认开。关掉它就回到"只看训练点 MSE"
            的旧口径——角点钉扎/下溢尖峰类病理解将不受惩罚）。

    Returns:
        (score, result_matrix, optimized_params)；优化失败时为 (None, None, None)。
    """
    inputs = np.asarray(data['inputs'], dtype=float)
    outputs = np.asarray(data['outputs'], dtype=float)

    # 数据前置校验：一个 NaN 目标值会让每个起点产生 NaN 损失，所有样本
    # 统一失败成无信息量的 'no output'，整场实验静默产出零程序。
    # （np.var(outputs) 为 nan 也会击穿下面的 var_outputs != 0 保护。）
    if inputs.ndim == 1:
        # 1 维按"n 个样本 × 1 个特征"解释，避免 `*inputs.T` 把标量当特征列表展开
        inputs = inputs.reshape(-1, 1)
    elif inputs.ndim != 2:
        raise ValueError(f'inputs 需为 1/2 维数组，实际 ndim={inputs.ndim}')
    if not np.isfinite(inputs).all():
        raise ValueError('inputs 包含 NaN/inf，请先清洗数据（如 read_csv 后 dropna）')
    if not np.isfinite(outputs).all():
        raise ValueError('outputs 包含 NaN/inf，请先清洗数据（如 read_csv 后 dropna）')

    cap = residual_cap(outputs)

    def raw_residual(params) -> np.ndarray:
        """未清洗的残差：方程自身可以在边界附近溢出/除零/对负数开方。

        ``np.errstate`` 只把 numpy 逐次浮点告警就地静音——这些数值事件由下面的
        ``residual()`` 统一清洗，而每个样本要试 5+ 个起点、每个起点上百次迭代，
        逐个刷 RuntimeWarning 的代价是整场实验的日志被噪声淹没（且因为评估跑在
        worker 子进程里，这些行连 run.err 都进不去，见 RESIDUAL_CAP 注释）。
        """
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            predictions = equation(*inputs.T, params)
        return np.asarray(predictions, dtype=float) - outputs

    def residual(params) -> np.ndarray:
        # 交给 scipy 的残差必须有界：inf/nan 以及 1e50+ 量级的有限值都会让 trf
        # 的信任域运算溢出成片 RuntimeWarning（旧行为），而优化器从中也读不到
        # 任何有用信息。
        return _sanitize_residual(raw_residual(params), cap)

    best_x, best_loss = _multi_start_least_squares(
        residual,
        n_params,
        n_starts=n_starts,
        max_iter=max_iter,
        bounds=bounds,
        x0=x0,
        seed=seed,
        initial_residual_func=raw_residual,
    )
    if best_x is None:
        return None, None, None

    # 贴边监控：只记日志（stderr → worker 旁路进 run.err），不改变评分/优化行为
    _report_bound_contact(best_x, bounds)

    # 残差列与 score 用同一口径（同一个清洗后残差函数取反）：一方面避免把 inf/NaN
    # 写进 ResidualAnalyzerAgent 的唯一输入（residual[:, -1] —— "nan" 喂进提示词
    # 只会让分析回路说废话），另一方面保证矩阵里的残差与 best_loss 严格一致。
    res = -residual(np.asarray(best_x, dtype=float))
    var_outputs = float(np.var(outputs))
    if var_outputs > 0:
        nmse = best_loss / var_outputs
    else:
        # 常数输出数据集：完美拟合记 nmse=0（R²=1），否则 R² 无定义记为 inf
        nmse = 0.0 if best_loss <= 0 else np.inf
    if verbose:
        # 不用 'R²' 上标：GBK 控制台（Windows 默认代码页）无法编码 \xb2，
        # verbose 路径会直接 UnicodeEncodeError 拖垮一次评估。
        print(f'R2 指标: {1.0 - nmse:.6f}  NMSE 指标: {nmse:.6f}')

    # 拟合后动态范围体检：训练点 MSE 完全看不见"点与点之间"的行为，而 LLM 骨架
    # 恰恰爱用角点钉扎/下溢尖峰这类局部化器件去消化个别难拟合的数据点（实测 45 次
    # 历史实验约 40 次的最优解携带，最极端 grid_min=-1.5e6、grid_max=3.9e28）。
    # 罚分只加在**评分**上（加性、无量纲、随严重度增长），不改残差——least_squares
    # 仍忠实于训练数据，残差分析回路看到的也是真实残差。
    penalty = 0.0
    if range_check:
        # params/probe_fn 一起传：判据三（大系数抵消）需要"换一组参数再算一次"，
        # 已绑定参数的 evaluate_fn 做不到（见 core.range_check 的文档）。
        info = dynamic_range_check(
            inputs, outputs, lambda *cols: equation(*cols, np.asarray(best_x)),
            params=np.asarray(best_x),
            probe_fn=lambda *args: equation(*args[:-1], np.asarray(args[-1])))
        penalty = float(info.get("penalty") or 0.0)
        if penalty > 0:
            # 三条判据分别打标：放大型（输出跨度）与门控型（局部斜率）、大系数抵消型
            # 的处置方式不同——跨度罚分随幅度指数级增长（几乎等于否决），斜率与系数
            # 是排序偏好。
            hits = []
            if info.get("span_penalty"):
                hits.append(f"输出跨度={info['span_ratio']:.3g}(上限{info['limit']})")
            if info.get("slope_penalty"):
                hits.append(f"局部斜率={info['slope_max']:.3g}"
                            f"(上限{info['slope_limit']})")
            if info.get("coef_penalty"):
                hits.append(f"系数抵消={info['coef_ratio']:.3g}"
                            f"(上限{info['coef_limit']})")
            print(f"[RANGE] 动态范围体检：命中 {'；'.join(hits)}，"
                  f"评分罚分 {penalty:.4g}", file=sys.stderr)

    # 输入/输出列按 decimal_places 取整仅供展示；残差列必须保持完整精度：
    # 它就是 ResidualAnalyzerAgent 的唯一输入（residual[:, -1]），按绝对
    # 3 位小数取整会把拟合越好的样本变成全 0 残差——恰好毁掉残差分析
    # 回路最该批评的那批样本。
    result_data = np.column_stack((
        np.round(inputs, decimal_places),
        np.round(outputs, decimal_places),
        res,
    ))
    return -(best_loss + penalty), result_data, np.asarray(best_x)
