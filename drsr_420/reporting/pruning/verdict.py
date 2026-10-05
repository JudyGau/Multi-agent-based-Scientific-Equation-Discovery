"""剪枝的量化评估：剪枝前后表达式在**训练数据**上的拟合对比 + 剪枝实质判定。

命名
----
本模块原名 ``prune_report``。那个名字在**结构上**是错的：它当时同时装着"读 CSV /
对齐列名 / 求值"等全层共享的数据工具，于是 ``holdout``、``expr_curves`` 只能 import
它的私有名，一个名字叫"剪枝报告"的模块成了事实上的公共工具库。数据工具已拆到
:mod:`drsr_420.reporting.data_io` 与 :mod:`drsr_420.equations.numeric`，本模块现在
确实只做"对一次剪枝做量化评估"，故改名为 ``prune_eval``——**名实一致**，而不是靠
docstring 解释一个不准确的名字（那正是上次改名留下的债）。

角色归属
--------
收尾分析（analysis）阶段的评估工具，服务于两处：

* ``find_best_eq.prune_and_visualize``：把对比结果放进剪枝摘要（控制台 + explain 提示词）；
* ``explain``：解释 LLM 要论证"剪枝合理"，必须有实测数值支撑（剪枝前后 MSE 变化、
  最大逐点偏差）。只凭公式长相论证"这些项可以去掉"是不可验证的空话——而用户明确
  要求 report.md 解释剪枝过程与合理性。

除了拟合对比，本模块还提供"剪枝到底做了什么"的判定（``classify_pruning``）：
``nodes_pruned`` 是"是否真剪枝"的唯一硬判据，而 ``simplify`` 只做通分/展开时公式
**数学上没变**——这种情况必须如实说明并沿用原式，否则 report.md 会被逼着解释一次
并不存在的剪枝（见 ``max_relative_difference`` 关于假阴性的说明）。

为什么单独成模块
----------------
"两条表达式在数据点上求值对比"与"判定剪枝的实质"这两件事，既不属于绘图
（expr_curves，曲线是给眼睛看的投影），也不属于剪枝算法（sensitivity_prune，
只关心敏感度阈值）。

失败策略：一律只告警并返回 ``None``/空字典——剪枝评估不该拖垮收尾流程。
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from drsr_420.reporting.data_io import (
    infer_data_csv,
    load_training_data,
    resolve_columns,
    resolve_csv,
    warn_once,
)
from drsr_420.equations.numeric import lambdify_eval

# ── 兼容 re-export（对象同一）────────────────────────────────────
# 这些名字原先定义在本模块，现位于 data_io / expr_numeric；保留别名供既有调用方
# 与外部脚本按旧路径取用，避免"搬家即失联"。
_warn_once = warn_once

#: 判定"剪枝是否只是换了写法"（而非真的删掉项）的相对误差阈值。
#: 实测 36 次真实运行：仅形式变化的样本最大相对差 ≤ 9.6e-13，真剪枝的样本 ≥ 7.6e-3
#: —— 中间有 10 个数量级的安全间隔，1e-6 两侧都留足余量。
FORM_ONLY_RTOL = 1e-6

#: 形式等价校验的采样点数与随机种子（种子固定，同一实验可复现）。
VERIFY_SAMPLES = 200
VERIFY_SEED = 20260919


def _evaluate(expr, sym_names: list[str], args: list[np.ndarray]) -> np.ndarray | None:
    """把 SymPy 表达式 lambdify 后在数据点上求值；失败返回 None（保留异常正文）。

    求值样板（errstate 静音 + 标量广播）在 :mod:`drsr_420.equations.numeric`，
    这里只负责把失败翻译成一句带原因的告警。
    """
    values, error = lambdify_eval(expr, sym_names, args)
    if error is not None:
        print(f"[WARN] 表达式数值化失败: {error}")
        return None
    return values


def compare_fits(dependent: str, sym_names: list[str], data: np.ndarray,
                 expr, pruned=None) -> dict:
    """在训练数据上对比剪枝前后的拟合，返回可直接写进提示词/日志的量化摘要。

    Returns:
        dict，可能包含 ``n_points`` / ``mse_before`` / ``nmse_before`` /
        ``max_abs_err_before`` / ``max_rel_err_before`` 与（给了 ``pruned`` 时）
        ``mse_after`` / ``nmse_after`` / ``max_abs_err_after`` / ``max_rel_err_after`` /
        ``max_abs_diff`` / ``identical``，以及失败时的 ``error``（缺字段即该项无法计算）。
    """
    out: dict = {}
    if data is None:
        return out
    try:
        dep_col, ind_cols, note = resolve_columns(data, dependent, sym_names)
    except KeyError as e:
        print(f"[WARN] 剪枝前后的拟合对比跳过：{e}")
        return out
    if note:
        _warn_once(f"变量名与数据列不完全一致：{note}")

    y = np.asarray(data[dep_col], dtype=float)
    args = [np.asarray(data[name], dtype=float) for name in ind_cols]
    out["n_points"] = int(y.size)

    pred_before = _evaluate(expr, sym_names, args)
    if pred_before is None:
        out["error"] = "剪枝前表达式求值失败"
        return out
    mse_before = float(np.mean(np.square(pred_before - y)))
    out["mse_before"] = mse_before
    var_y = float(np.var(y))
    out["nmse_before"] = mse_before / var_y if var_y > 0 else None
    # 样本内的最大误差：report.md 的「样本外验证」小节要用它与 held-out 同口径对比
    err_before = np.abs(pred_before - y)
    out["max_abs_err_before"] = float(np.max(err_before))
    out["max_rel_err_before"] = float(np.max(err_before / np.maximum(np.abs(y), 1e-12)))

    if pruned is None:
        return out
    pred_after = _evaluate(pruned, sym_names, args)
    if pred_after is None:
        out["error"] = "剪枝后表达式求值失败"
        return out
    mse_after = float(np.mean(np.square(pred_after - y)))
    out["mse_after"] = mse_after
    out["nmse_after"] = mse_after / var_y if var_y > 0 else None
    # 剪枝后的最大误差：对外发布的是剪枝后表达式，样本内对照必须与它同口径
    # （见 holdout.in_sample_metrics，缺这两项就只能回退剪枝前的值）
    err_after = np.abs(pred_after - y)
    out["max_abs_err_after"] = float(np.max(err_after))
    out["max_rel_err_after"] = float(np.max(err_after / np.maximum(np.abs(y), 1e-12)))
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


# ── 剪枝实质判定（真剪枝 / 只是形式变化）───────────────────────

def sample_points(sym_names: list[str], sample_range, num_samples: int = VERIFY_SAMPLES,
                  seed: int = VERIFY_SEED) -> list[np.ndarray]:
    """在 ``sample_range`` 内均匀随机采样，返回每个自变量的取值数组（固定种子）。"""
    rng = np.random.default_rng(seed)
    lo, hi = float(sample_range[0]), float(sample_range[1])
    return [rng.uniform(lo, hi, num_samples) for _ in sym_names]


def max_relative_difference(expr_a, expr_b, sym_names: list[str],
                            sample_range=(1.0, 14.0), num_samples: int = VERIFY_SAMPLES,
                            seed: int = VERIFY_SEED, extra_points=None) -> float | None:
    """两条表达式在采样网格上的最大相对差；无有效采样点（全非有限）时返回 None。

    **这是"公式到底变没变"的权威判据。** ``sp.simplify(a - b) == 0`` 与 ``a.equals(b)``
    在含浮点指数的表达式上都会给出**假阴性**：实测某次 0 项剪枝的样本
    ``simplify(差) == 0`` 与 ``equals()`` 都是 False，而它在 8 个训练数据点与 100 个
    采样点上的相对差恰为 0（``equals`` 会在负实数/复数域上取样，浮点指数的分支不同）。
    本函数只在与剪枝决策**同一个有效定义域**（``sample_range``）上比较。

    ``extra_points``（训练数据点）并入比较网格：均匀随机撒点对"只在个别数据点
    承重"的项是盲的——实测 (1,1) 角点锚项被剪后随机网格相对差仍报 0.0e+00，
    而训练点上差 2989.9；数据点必须参与，否则"剪枝没改变模型"的结论是假阴性。
    """
    if not sym_names:
        return None
    args = sample_points(list(sym_names), sample_range, num_samples, seed)
    if extra_points is not None:
        if isinstance(extra_points, np.ndarray):
            rows = np.asarray(extra_points, dtype=float)
            if rows.ndim == 1:
                rows = rows.reshape(1, -1)
        else:
            rows = np.column_stack(
                [np.asarray(c, dtype=float).ravel() for c in extra_points])
        if rows.ndim == 2 and rows.shape[1] == len(sym_names) and len(rows):
            args = [np.concatenate([a, rows[:, j]])
                    for j, a in enumerate(args)]
    va = _evaluate(expr_a, list(sym_names), args)
    vb = _evaluate(expr_b, list(sym_names), args)
    if va is None or vb is None:
        return None
    finite = np.isfinite(va) & np.isfinite(vb)
    if not finite.any():
        return None
    a, b = va[finite], vb[finite]
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)
    return float(np.max(np.abs(b - a) / scale))


def _rel_note(out: dict) -> str:
    rel = out.get("max_rel_diff")
    return "" if rel is None else f"，剪枝前后采样最大相对差 {rel:.1e}"


def classify_pruning(expr, published, stats, sym_names: list[str] | None = None,
                     sample_range=(1.0, 14.0), extra_points=None) -> dict:
    """判定本次"剪枝"的实质，返回可直接写进日志/解释提示词的证据字典。

    判定分层（``nodes_pruned`` 是唯一硬判据，数值比较只用来论证"形式变化"）::

        nodes_pruned > 0 但剪枝结果不含任何自变量 → 'degenerate'（退化剪枝，公式回退原式）
        nodes_pruned > 0                        → 'pruned'（真剪枝，公式用剪枝结果）
        nodes_pruned == 0 且 simplify 没改写形式 → 'none'（什么都没变）
        nodes_pruned == 0 且 simplify 改写了形式 → 'form_only'（通分/重排，公式已回退原式）
        同上但数值等价性无法确认/不成立           → 'form_only_unverified'（同样回退，需人工看一眼）

    为什么需要它：``SensitivityPruner.prune`` 在 0 项剪枝时会返回原表达式，但**日志与
    report.md 仍要如实说明**"simplify 本来会把它改写成什么形式、那只是通分"，
    否则读者会以为公式被剪枝改变了（也可能反过来怀疑剪枝没生效）。

    返回的 ``max_rel_diff`` 是**判定所依据的那一次比较**：真剪枝时是"剪枝结果 vs 原式"
    （剪枝改变了多少模型，应当明显非零）；0 项剪枝时是"simplify 会给出的形式 vs 原式"
    （只是换写法，应当≈0）。

    Args:
        expr: 剪枝前（参数已代入）的表达式。
        published: ``SensitivityPruner.prune`` 返回、即将对外发布的表达式。
        stats: 同一次剪枝的 ``PruneStats``。
        sym_names / sample_range: 形式等价校验的变量与采样区间，应与剪枝参数一致。
        extra_points: 训练数据点，并入比较网格防止"只在个别数据点承重"的项漏判
            （见 :func:`max_relative_difference` 与实测 20260921-161549）。
    """
    actually = bool(getattr(stats, "actually_pruned", False))
    simplified = getattr(stats, "simplified_expr", None)
    ops_before = getattr(stats, "ops_before", 0) or sp.count_ops(expr)

    out: dict = {
        "kind": "pruned" if actually else "none",
        "actually_pruned": actually,
        "used_original": not actually,
        "nodes_visited": int(getattr(stats, "nodes_visited", 0)),
        "nodes_pruned": int(getattr(stats, "nodes_pruned", 0)),
        "prune_rate": float(getattr(stats, "prune_rate", 0.0)),
        "ops_before": int(ops_before),
        "ops_published": int(sp.count_ops(published)),
        "simplify_ops": int(sp.count_ops(simplified)) if simplified is not None else None,
        # simplify 是否会改写形式（0 项剪枝时才有意义）：
        "form_rewritten": bool(simplified is not None and simplified != expr),
        # 对外发布的公式是否与剪枝前不同：
        "form_changed": bool(published != expr),
        "numerically_equivalent": None,
        "max_rel_diff": None,
        "summary": "",
    }

    if sym_names:
        # 比较对象随判定而变化：真剪枝时比"剪枝结果 vs 原式"（剪枝改变了多少模型）；
        # 0 项剪枝时比"simplify 会给出的形式 vs 原式"（那只是换写法，应当≈0）。
        # 不能拿发布的表达式去比——0 项剪枝时它就是原式，比出来恒为 0，什么也证明不了。
        compared = simplified if (not actually and out["form_rewritten"]) else published
        out["max_rel_diff"] = max_relative_difference(
            expr, compared, list(sym_names), sample_range, extra_points=extra_points)
    if out["max_rel_diff"] is not None:
        out["numerically_equivalent"] = bool(out["max_rel_diff"] <= FORM_ONLY_RTOL)

    # 退化剪枝：剪枝结果不再含任何自变量（因变量被写成一个常数）。这在语义上就是
    # 剪枝失败——发布它等于宣称"σ 与自变量无关"：实测最优样本因此从 NMSE 8.7e-07
    # 退到 6.81（比"预测样本均值"的 1.0 还差 6.8 倍）。判据是"公式里还有没有自变量"，
    # 属**结果层面**的兜底（与"节点数是否变少"无关），故在这里把发布形式一并回退原式。
    independents = {str(s) for s in (sym_names or [])}
    kept_symbols = {str(s) for s in getattr(published, "free_symbols", set())}
    if actually and independents and not (independents & kept_symbols):
        out.update({
            "kind": "degenerate",
            "actually_pruned": False,
            "used_original": True,
            "form_changed": False,
            "ops_published": int(ops_before),
        })
        out["summary"] = (
            f"本次剪枝被拒（退化剪枝）：剪枝结果不含任何自变量（公式退化为常数），"
            f"判为剪枝失败、公式沿用剪枝前的表达式"
            f"（被移除 {out['nodes_pruned']} 项，节点数 {out['ops_before']} → "
            f"{out['ops_published']}，被拒结果 {sp.count_ops(published)} 节点）"
            + _rel_note(out))
        return out

    if actually:
        out["summary"] = (
            f"本次实际剪枝：移除 {out['nodes_pruned']} 项，公式采用剪枝后的表达式"
            f"（节点数 {out['ops_before']} → {out['ops_published']}）" + _rel_note(out))
        return out

    if not out["form_rewritten"]:
        out["summary"] = (f"本次未实际剪枝（移除 0 项）：公式与剪枝前完全相同"
                          f"（节点数 {out['ops_before']}）")
        return out

    if out["numerically_equivalent"]:
        out["kind"] = "form_only"
        out["summary"] = (
            f"本次未实际剪枝（移除 0 项）：simplify 只会把公式改写成等价形式"
            f"（通分/展开/重排，节点数 {out['ops_before']} → {out['simplify_ops']}，"
            f"采样最大相对差 {out['max_rel_diff']:.1e}），公式沿用剪枝前的形式")
    else:
        out["kind"] = "form_only_unverified"
        rel_txt = ("无法判定（无有效采样点）" if out["max_rel_diff"] is None
                   else f"{out['max_rel_diff']:.1e}")
        out["summary"] = (
            f"本次未实际剪枝（移除 0 项）：simplify 改写了公式，但数值等价性无法确认"
            f"（最大相对差 {rel_txt}，判定阈值 {FORM_ONLY_RTOL:.0e}），"
            f"为稳妥起见仍沿用剪枝前的形式")
    return out


__all__ = [
    "FORM_ONLY_RTOL", "VERIFY_SAMPLES", "VERIFY_SEED",
    "compare_fits", "format_fit_summary", "sample_points",
    "max_relative_difference", "classify_pruning",
    # 兼容 re-export（定义已移至 data_io）
    "load_training_data", "resolve_columns", "resolve_csv", "infer_data_csv", "warn_once",
]