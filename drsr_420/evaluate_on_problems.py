"""评估模块：对 LLM 生成的方程做参数优化并打分。

统一返回契约：
    evaluate() 恒返回 (score, result_matrix, optimized_params) 三元组；
    优化失败（NaN/inf 损失、方程异常、所有起点失败）时返回 (None, None, None)。
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
PARAMS_BOUNDS = (-1000.0, 1000.0)   # 参数边界，防止无界优化导致溢出/NaN；据历史最优参数分布(最大|p|≈738)定
SAMPLE_SIZE = 100                   # 残差采样点数上限


def _clamp_params(x0: np.ndarray, bounds) -> np.ndarray:
    """把初始参数裁剪进边界（least_squares 要求初始点在边界内）。"""
    lower, upper = bounds
    return np.clip(x0, lower, upper)


def _multi_start_least_squares(
        residual_func,
        n_params: int,
        *,
        n_starts: int = N_STARTS,
        max_iter: int = MAX_ITER,
        bounds=PARAMS_BOUNDS,
        x0: np.ndarray | None = None,
        seed: int | None = None,
) -> tuple[np.ndarray | None, float]:
    """多起点最小二乘求解，返回 (最优参数, 最小均方误差)；全部失败时返回 (None, inf)。

    相比 BFGS：least_squares 利用残差结构求 Jacobian，收敛更快更稳；
    带参数边界可避免无界优化使方程参数发散（产生 NaN/inf）。
    """
    from scipy.optimize import least_squares

    rng = np.random.default_rng(seed)
    starts: list[np.ndarray] = []
    if x0 is not None:  # 热启动：把上一轮最优参数作为额外首起点（在 n_starts 随机起点之外）
        starts.append(np.asarray(x0, dtype=float))
    starts.extend(rng.uniform(-1.0, 1.0, size=n_params) for _ in range(n_starts))

    best_x, best_loss = None, np.inf
    first_exc: Exception | None = None
    for start in starts:
        start = _clamp_params(start, bounds)
        try:
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

    def residual(params):
        return equation(*inputs.T, params) - outputs

    best_x, best_loss = _multi_start_least_squares(
        residual,
        n_params,
        n_starts=n_starts,
        max_iter=max_iter,
        bounds=bounds,
        x0=x0,
        seed=seed,
    )
    if best_x is None:
        return None, None, None

    predictions = equation(*inputs.T, best_x)
    res = outputs - predictions
    var_outputs = float(np.var(outputs))
    if var_outputs > 0:
        nmse = best_loss / var_outputs
    else:
        # 常数输出数据集：完美拟合记 nmse=0（R²=1），否则 R² 无定义记为 inf
        nmse = 0.0 if best_loss <= 0 else np.inf
    if verbose:
        print(f'R² 指标: {1.0 - nmse:.6f}  NMSE 指标: {nmse:.6f}')

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
