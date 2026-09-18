"""剪枝的量化评估：剪枝前后表达式在**训练数据**上的拟合对比。

角色归属
--------
收尾分析（analysis）阶段的评估工具，服务于两处：

* ``find_best_eq.prune_and_visualize``：把对比结果放进剪枝摘要（控制台 + explain 提示词）；
* ``explain``：解释 LLM 要论证"剪枝合理"，必须有实测数值支撑（剪枝前后 MSE 变化、
  最大逐点偏差）。只凭公式长相论证"这些项可以去掉"是不可验证的空话——而用户明确
  要求 explain.md 解释剪枝过程与合理性。

为什么单独成模块
----------------
"定位并读取训练数据"（config_snapshot.json 的 data_csv → 绝对路径）与"两条表达式在
数据点上求值对比"这两件事，既不属于绘图（expr_curves，曲线是给眼睛看的投影），也不
属于剪枝算法（sensitivity_prune，只关心敏感度阈值）。

失败策略：一律只告警并返回 ``None``/空字典——剪枝评估不该拖垮收尾流程。
"""
from __future__ import annotations

import json
import os

import numpy as np
import sympy as sp

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_csv(data_csv: str, results_root: str = "") -> str | None:
    """data_csv 依次按 results_root、项目根、cwd 解析；兼容绝对路径。

    config_snapshot 里通常存项目根相对路径（./data/...），自包含实验目录
    （如测试夹具）则是 results_root 相对路径——两处都要试。
    """
    if os.path.isabs(data_csv):
        return data_csv if os.path.isfile(data_csv) else None
    for base in (results_root, _REPO_ROOT, os.getcwd()):
        p = os.path.join(base, data_csv)
        if os.path.isfile(p):
            return p
    return None


def load_training_data(results_root: str) -> np.ndarray | None:
    """按 config_snapshot.json 的 data_csv 读取训练数据（结构化数组）；失败返回 None。"""
    snap_path = os.path.join(results_root, "config_snapshot.json")
    try:
        with open(snap_path, "r", encoding="utf-8") as f:
            data_csv = json.load(f).get("data_csv")
    except Exception as e:
        print(f"[WARN] 读取 config_snapshot.json 失败: {e}")
        return None
    if not data_csv:
        print("[WARN] config_snapshot.json 里没有 data_csv，无法定位训练数据。")
        return None
    csv_path = resolve_csv(data_csv, results_root)
    if csv_path is None:
        print(f"[WARN] 数据文件不存在: {data_csv}")
        return None
    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
    except Exception as e:
        print(f"[WARN] 读取训练数据失败: {e}")
        return None
    if data.dtype.names is None or data.size == 0:
        print(f"[WARN] 训练数据为空或缺少表头: {csv_path}")
        return None
    return data


def _evaluate(expr, sym_names: list[str], args: list[np.ndarray]) -> np.ndarray | None:
    """把 SymPy 表达式 lambdify 后在数据点上求值；失败/形状非法返回 None。

    用 ``errstate`` 静音数值告警：剪枝前后的表达式都可能在被修剪的项上溢出，
    这里是"评估"而不是"优化"，没有清洗残差的必要，但也没必要刷警告。
    """
    try:
        func = sp.lambdify(list(sp.symbols(sym_names)), expr, modules="numpy")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            vals = np.asarray(func(*args), dtype=float)
    except Exception as e:
        print(f"[WARN] 表达式数值化失败: {e}")
        return None
    return vals


def compare_fits(dependent: str, sym_names: list[str], data: np.ndarray,
                 expr, pruned=None) -> dict:
    """在训练数据上对比剪枝前后的拟合，返回可直接写进提示词/日志的量化摘要。

    Returns:
        dict，可能包含 ``n_points`` / ``mse_before`` / ``nmse_before`` /
        ``mse_after`` / ``nmse_after`` / ``max_abs_diff`` / ``identical``，
        以及失败时的 ``error``（缺字段即该项无法计算）。
    """
    out: dict = {}
    if data is None or dependent not in (data.dtype.names or ()):
        return out
    missing = [v for v in sym_names if v not in data.dtype.names]
    if missing:
        return out

    y = np.asarray(data[dependent], dtype=float)
    args = [np.asarray(data[name], dtype=float) for name in sym_names]
    out["n_points"] = int(y.size)

    pred_before = _evaluate(expr, sym_names, args)
    if pred_before is None:
        out["error"] = "剪枝前表达式求值失败"
        return out
    mse_before = float(np.mean(np.square(pred_before - y)))
    out["mse_before"] = mse_before
    var_y = float(np.var(y))
    out["nmse_before"] = mse_before / var_y if var_y > 0 else None

    if pruned is None:
        return out
    pred_after = _evaluate(pruned, sym_names, args)
    if pred_after is None:
        out["error"] = "剪枝后表达式求值失败"
        return out
    mse_after = float(np.mean(np.square(pred_after - y)))
    out["mse_after"] = mse_after
    out["nmse_after"] = mse_after / var_y if var_y > 0 else None
    diff = np.abs(pred_after - pred_before)
    finite = np.isfinite(diff)
    out["max_abs_diff"] = float(diff[finite].max()) if finite.any() else float("inf")
    # 逐点完全相同（剪枝率 0% 的常见情形）：告诉解释 LLM"确实一个点都没变"
    out["identical"] = bool(np.array_equal(pred_after, pred_before))
    return out


def format_fit_summary(fit: dict | None) -> str:
    """把 :func:`compare_fits` 的结果渲染成一行中文摘要（无数据时给出原因）。"""
    if not fit:
        return "剪枝前后的拟合对比：本次无法计算（训练数据或表达式不可用）。"
    if fit.get("error"):
        return f"剪枝前后的拟合对比：{fit['error']}。"
    n = fit.get("n_points")
    before = fit.get("mse_before")
    after = fit.get("mse_after")
    if after is None or before is None:
        return f"剪枝前的拟合：{n} 个数据点上 MSE={before:.6g}。"
    if fit.get("identical"):
        return (f"剪枝前后的拟合对比：{n} 个数据点上逐点完全相同，"
                f"MSE 均为 {before:.6g}（剪枝未改变模型）。")
    change = (after - before) / before * 100.0 if before else float("inf")
    return (f"剪枝前后的拟合对比：{n} 个数据点上 MSE {before:.6g} → {after:.6g}"
            f"（相对变化 {change:+.2f}%），最大逐点偏差 {fit.get('max_abs_diff', float('nan')):.6g}。")
