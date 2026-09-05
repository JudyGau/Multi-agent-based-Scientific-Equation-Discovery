"""评估模块：对 LLM 生成的方程做参数优化并打分。

统一返回契约：
    evaluate() 恒返回 (score, result_matrix, optimized_params) 三元组；
    优化失败（NaN/inf 损失、方程异常、所有起点失败）时返回 (None, None, None)。
    score 取负均方误差（越大越好）；result_matrix 为 (输入, 输出, 残差) 拼接矩阵，
    仅用于残差分析展示；optimized_params 可直接作为下一轮优化的热启动起点。
"""
from __future__ import annotations

import numpy as np

# 模块级默认配置，可通过 evaluate() 关键字参数覆盖
MAX_NPARAMS = 10                    # 方程参数个数
DECIMAL_PLACES = 3                  # 结果矩阵保留的小数位数（仅展示，不影响评分）
N_STARTS = 5                        # 多起点优化的起点数
MAX_ITER = 300                      # 每个起点的最大函数评估次数
PARAMS_BOUNDS = (-10.0, 10.0)       # 参数边界，防止无界优化导致溢出/NaN
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
    if x0 is not None:  # 热启动：把上一轮最优参数作为首个起点
        starts.append(np.asarray(x0, dtype=float))
    starts.extend(rng.uniform(-1.0, 1.0, size=n_params) for _ in range(n_starts))

    best_x, best_loss = None, np.inf
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
        except Exception:
            continue  # 该起点失败（如方程在该参数域不可用），尝试下一个起点
        loss = float(np.mean(np.square(result.fun)))
        if not np.isfinite(loss):
            continue
        if loss < best_loss:
            best_loss, best_x = loss, result.x
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
    inputs, outputs = data['inputs'], data['outputs']

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
    nmse = best_loss / var_outputs if var_outputs != 0 else np.inf
    if verbose:
        print(f'R² 指标: {1.0 - nmse:.6f}  NMSE 指标: {nmse:.6f}')

    # 结果矩阵仅用于残差分析展示：输入/输出/残差统一按 decimal_places 取整
    result_data = np.column_stack((
        np.round(inputs, decimal_places),
        np.round(outputs, decimal_places),
        np.round(res, decimal_places),
    ))
    return -best_loss, result_data, np.asarray(best_x)
