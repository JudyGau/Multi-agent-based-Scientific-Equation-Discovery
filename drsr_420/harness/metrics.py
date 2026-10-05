"""论文指标定义：E1 主表那几列的纯函数。

每一列的口径都写死在这里，避免"论文里的 NMSE"与"报告里的 NMSE"各算一套：

* :func:`mse` / :func:`nmse` —— 预测精度。``nmse`` 的分母是目标的**总体方差**
  （``ddof=0``），与评估器 :func:`drsr_420.execution.problems.evaluate` 的
  ``NMSE = MSE / var(outputs)`` 同口径；这条一致性是刻意的，否则跨 run 汇总出的
  NMSE 与单 run 报告里的 NMSE 会差一个 ``n/(n-1)`` 因子而无人察觉。
* :func:`acc_at` —— ``Acc@tol``：NMSE 不超过阈值的**比例**（论文默认 ``tol=0.1``）。
* :func:`symbolic_accuracy` —— SA：发现式与真值式是否**符号等价**。真值表达式本地
  尚缺（见 ``docs/RESEARCH_PLAN.md`` §2.7），因此它现在是可单测的指标定义，
  尚未接进 :mod:`drsr_420.harness.aggregate` 的汇总链。

约定的"未知 ≠ 失败"：``nmse=None`` / SA 解析失败一律**排除**在分母之外，而不是
记成未命中——把"测不出来"说成"没测过"正是本仓库反复记录过的缺陷成因。
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

#: 论文 ``Acc@tol`` 的默认阈值（NMSE ≤ 0.1 视为命中）。
DEFAULT_ACC_TOL = 0.1


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """均方误差 ``mean((y_true - y_pred)^2)``。"""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"形状不一致: y_true{true.shape} vs y_pred{pred.shape}")
    if true.size == 0:
        raise ValueError("mse 需要至少一个数据点")
    return float(np.mean((true - pred) ** 2))


def nmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """归一化 MSE ``MSE / var(y_true)``（分母为总体方差，与评估器同口径）。

    目标方差为 0 时：预测完全一致记 ``0.0``，否则记 ``inf``——常数目标下"归一的
    MSE"无定义，返回 ``inf`` 会让它在任何"越小越好"的排序/阈值里自然落选，
    比返回 ``nan``（会在 median / 比较里静默传播）安全。
    """
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    variance = float(np.var(true))
    if variance == 0.0:
        return 0.0 if np.array_equal(true, pred) else float("inf")
    return mse(true, pred) / variance


def acc_at(values: Iterable[float | None], tol: float = DEFAULT_ACC_TOL) -> float:
    """``Acc@tol``：``values`` 中不超过 ``tol`` 的比例（``None`` 不计入分母）。

    空集合抛 ``ValueError`` 而不是返回 0 或 1：没有样本时"命中率"没有意义，
    静默给个数字会变成表格里一个看着正常的假值。
    """
    known = [float(value) for value in values if value is not None]
    if not known:
        raise ValueError("acc_at 需要至少一个有效 NMSE（None 不计入分母）")
    return sum(1.0 for value in known if value <= tol) / len(known)


def symbolic_accuracy(candidate: str, truth: str, symbols: Sequence[str]) -> bool | None:
    """SA：两条表达式是否**符号等价**（``simplify(candidate - truth) == 0``）。

    ``symbols`` 是自变量名（其余名字回落到 sympy 命名空间，见
    :func:`drsr_420.equations.parse` 里关于撞名的说明）。解析失败或化简异常返回
    ``None``——"测不出来"与"不等价"必须分开，否则 SA 会被一堆解析失败悄悄拉低。

    sympy 为惰性导入：本模块的其余指标（供 :mod:`drsr_420.harness.aggregate` 用）
    不必为了一个尚未接入的指标拖起 sympy。
    """
    import sympy as sp

    local = {name: sp.Symbol(name) for name in symbols}
    try:
        left = sp.parse_expr(str(candidate), local_dict=dict(local))
        right = sp.parse_expr(str(truth), local_dict=dict(local))
    except Exception:
        return None
    try:
        return bool(sp.simplify(left - right) == 0)
    except Exception:
        return None