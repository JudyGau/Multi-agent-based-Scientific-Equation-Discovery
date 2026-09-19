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
    优化的热启动起点。
"""
from __future__ import annotations

import numpy as np

# 模块级默认配置，可通过 evaluate() 关键字参数覆盖
MAX_NPARAMS = 10                    # 方程参数个数
DECIMAL_PLACES = 3                  # 结果矩阵保留的小数位数（仅展示，不影响评分）
N_STARTS = 5                        # 多起点优化的起点数
MAX_ITER = 300                      # 每个起点的最大函数评估次数
PARAMS_BOUNDS = (-10000.0, 10000.0) # 参数边界，防止无界优化导致溢出/NaN；据历史最优参数分布(最大|p|≈738)定
SAMPLE_SIZE = 100                   # 残差采样点数上限

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
    starts.extend(rng.uniform(-1.0, 1.0, size=n_params) for _ in range(n_starts))

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
) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
    """对 `equation(*X.T, params)` 做参数优化并评分。

    Args:
        data: 含 'inputs' 与 'outputs' 的数据字典。
        equation: 可调用对象，签名 equation(*feature_arrays, params)。
        x0: 热启动参数（例如上一轮评估得到的最优参数），作为首个优化起点。

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

    # 输入/输出列按 decimal_places 取整仅供展示；残差列必须保持完整精度：
    # 它就是 ResidualAnalyzerAgent 的唯一输入（residual[:, -1]），按绝对
    # 3 位小数取整会把拟合越好的样本变成全 0 残差——恰好毁掉残差分析
    # 回路最该批评的那批样本。
    result_data = np.column_stack((
        np.round(inputs, decimal_places),
        np.round(outputs, decimal_places),
        res,
    ))
    return -best_loss, result_data, np.asarray(best_x)
