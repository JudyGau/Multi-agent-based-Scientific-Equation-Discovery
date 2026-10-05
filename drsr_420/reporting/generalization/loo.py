"""留一交叉验证（LOO）：训练点太少时的样本外口径。

角色归属
--------
与 :mod:`drsr_420.reporting.generalization.holdout` 并列的另一条**泛化性口径**，两者互斥地服务同一个
问题："公式在没见过的点上行不行"。

* ``holdout``：训练点够多，切得出独立 held-out（ID / OOD 两条通道）；
* ``loo``（本模块）：训练点太少（< :data:`LOO_MAX_TRAIN`）时切不出可信的独立 held-out，
  改为每次留出 1 点、用其余 n-1 点**重新拟合参数**再预测该点。

为什么必须与 held-out 分开成模块
--------------------------------
"重拟合"这件事带来了 held-out 没有的三样东西：一套自包含的多起点有界最小二乘
（:func:`_loo_fit`）、样本骨架的编译（:func:`skeleton_callable`）、以及一处必须**镜像**
评估器默认值的参数口径（:data:`LOO_FIT_BOUNDS` 等）。这些既不属于"读 held-out CSV"，
也不属于"渲染样本外小节"，混在 ``holdout`` 里会让那个模块名越来越名不副实——
``skeleton_callable`` 与"样本外验证"本就毫无关系。

契约与 ``holdout`` 一致：**只报告，不参与采样、打分、早停与样本选择**。一旦参与选择，
它就不再是样本外。

已知的重复（有意保留）
----------------------
``LOO_FIT_BOUNDS`` / ``LOO_FIT_N_STARTS`` / ``LOO_FIT_MAX_ITER`` 是
``evaluation.problems`` 评估器默认值的**镜像**：analysis 层不得依赖 evaluation 层
（分层规则见 ``tests/test_architecture.py``），故在此独立定义，并由
``tests/test_holdout.py`` 的漂移护栏断言两边一致。
"""
from __future__ import annotations

import re

import numpy as np

from drsr_420.reporting.data_io import resolve_columns, warn_once
from drsr_420.reporting.md_sections import strip_section

#: 训练点数 **低于** 该值时，自动改用留一法（LOO）替代 held-out。
#: 理由：MRF 六个体系只有 7–19 点，其 ``test.csv`` 仅 2 行且落在训练区间内（插值），
#: 切不出可信的独立 held-out；n 很小时 LOO 才是统计上诚实的做法。
LOO_MAX_TRAIN = 30

#: report.md 里留一交叉验证小节的标题（机器生成，正文同名小节会被替换）。
LOO_HEADING = "## 留一交叉验证（LOO）"

#: LOO 重拟合的参数口径——**镜像** ``evaluation.problems`` 的评估器默认值。
#: analysis 层不得依赖 evaluation 层（分层规则），故在此独立定义；
#: ``tests/test_holdout.py`` 有一条漂移护栏断言两边一致。
LOO_FIT_BOUNDS = (-10000.0, 10000.0)
LOO_FIT_N_STARTS = 5
LOO_FIT_MAX_ITER = 300
LOO_FIT_SEED = 0     # 固定种子：同一实验的 LOO 可复现


def skeleton_callable(func: str):
    """把样本函数骨架字符串编译成可调用对象 ``f(*features, params)``；失败返回 ``None``。

    与评估器调用样本的方式一致（``exec`` 样本自带的 ``def``，命名空间注入 ``np``）；
    这是 LOO 能在**每个折上重新拟合参数**的前提——收尾用的
    :func:`drsr_420.equations.parse.expr_substitution` 已把 ``params[k]`` 全部
    替换成数值，无法再拟合。
    """
    if not func:
        return None
    match = re.search(r"^def\s+\w+\s*\(", func, re.M)
    if not match:
        return None
    namespace: dict = {"np": np}
    try:
        exec(func[match.start():], namespace)   # noqa: S102 - 执行的是实验自己选出的样本
    except Exception as e:
        print(f"[WARN] LOO：样本骨架编译失败，跳过留一验证: {e}")
        return None
    for key, value in namespace.items():
        if callable(value) and not key.startswith("__"):
            return value
    return None


def _loo_fit(fn, X: np.ndarray, y: np.ndarray, n_params: int, seed: int = LOO_FIT_SEED):
    """在给定子集上做多起点有界最小二乘，返回拟合参数（全失败返回 ``None``）。

    自包含实现（不 import evaluation.problems）：残差按训练集尺度截断以保证给 scipy
    的输入有界，起点口径与评估器一致（首点 U(-1,1)、其余按量级对数均匀）。
    """
    from scipy.optimize import least_squares

    bounds = LOO_FIT_BOUNDS
    scale = float(np.var(y))
    cap = max(1.0, scale * 1e6) if np.isfinite(scale) and scale > 0 else 1.0

    def residual(params) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pred = np.asarray(fn(*X.T, np.asarray(params, dtype=float)), dtype=float)
        if pred.shape != y.shape:
            pred = np.broadcast_to(pred, y.shape)
        r = np.asarray(pred - y, dtype=float)
        r = np.where(np.isfinite(r), r, cap)
        return np.clip(r, -cap, cap)

    rng = np.random.default_rng(seed)
    starts = [rng.uniform(-1.0, 1.0, size=n_params)]
    for _ in range(max(0, LOO_FIT_N_STARTS - 1)):
        starts.append(np.exp(rng.uniform(np.log(1e-3), np.log(1e4), size=n_params)))

    best_x, best_loss = None, np.inf
    for start in starts:
        start = np.clip(start, bounds[0], bounds[1])
        try:
            result = least_squares(residual, start, bounds=bounds,
                                   max_nfev=LOO_FIT_MAX_ITER)
        except Exception:
            continue
        loss = float(np.mean(np.square(result.fun)))
        if np.isfinite(loss) and loss < best_loss:
            best_loss, best_x = loss, result.x
    return best_x


def evaluate_loo(dependent: str, sym_names: list[str], func: str,
                 data: np.ndarray, n_params: int, seed: int = LOO_FIT_SEED) -> dict | None:
    """留一交叉验证：每次留出 1 点、用其余 n-1 点**重新拟合**参数，再预测该点。

    只报告，不参与任何选择（与 ``holdout.evaluate_holdout`` 同一契约）。返回指标字典
    （列名对不上 / 骨架编译失败 / 点数不足返回 ``None``）：
    ``n_points`` / ``n_ok`` / ``mse`` / ``nmse``（分母=训练集方差）/ ``median_abs_err``
    / ``p95_abs_err`` / ``median_rel_err`` / ``p95_rel_err`` / ``train_var`` / ``rows``。

    **为什么必须重拟合**：收尾的 ``expr_substitution`` 已把参数替换成数值，直接用那组
    参数在每个留出点上求值得到的只是"样本内逐点误差"，参数见过该点，不是 LOO。
    """
    if data is None:
        return None
    try:
        dep_col, ind_cols, note = resolve_columns(data, dependent, sym_names)
    except KeyError as e:
        print(f"[WARN] LOO 跳过：{e}")
        return None
    if note:
        warn_once(f"变量名与数据列不完全一致：{note}")

    y = np.asarray(data[dep_col], dtype=float)
    X = np.column_stack([np.asarray(data[c], dtype=float) for c in ind_cols])
    n = int(y.size)
    if n < 3:
        print(f"[WARN] LOO 跳过：训练点仅 {n} 个，留一法至少需要 3 个。")
        return None

    fn = skeleton_callable(func)
    if fn is None:
        return None
    if n_params <= 0:
        n_params = 1

    rows, ok = [], 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        fitted = _loo_fit(fn, X[mask], y[mask], n_params, seed=seed + i)
        if fitted is None:
            pred = float("nan")
        else:
            try:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    out = np.asarray(fn(*X[i:i + 1].T, fitted), dtype=float).reshape(-1)
                pred = float(out[0]) if out.size else float("nan")
            except Exception:
                pred = float("nan")
        if not np.isfinite(pred):
            abs_err = rel_err = float("nan")
        else:
            abs_err = abs(pred - y[i])
            rel_err = abs_err / max(abs(y[i]), 1e-12)
            ok += 1
        rows.append({
            "variables": {name: float(X[i, j]) for j, name in enumerate(ind_cols)},
            "observed": float(y[i]),
            "predicted": pred,
            "abs_err": abs_err,
            "rel_err": rel_err,
        })

    abs_arr = np.array([r["abs_err"] for r in rows], dtype=float)
    rel_arr = np.array([r["rel_err"] for r in rows], dtype=float)
    finite = np.isfinite(abs_arr)
    train_var = float(np.var(y))
    mse = float(np.nanmean(np.square(abs_arr))) if finite.any() else float("nan")

    def _stat(arr, q):
        good = arr[np.isfinite(arr)]
        return float(np.percentile(good, q)) if good.size else None

    return {
        "n_points": n,
        "n_ok": ok,
        "mse": mse,
        "nmse": (mse / train_var) if train_var else None,
        "median_abs_err": _stat(abs_arr, 50),
        "p95_abs_err": _stat(abs_arr, 95),
        "median_rel_err": _stat(rel_arr, 50),
        "p95_rel_err": _stat(rel_arr, 95),
        "train_var": train_var,
        "rows": rows,
    }


def format_loo_summary(loo: dict | None) -> str:
    """把 LOO 指标渲染成一行控制台文本。"""
    if not loo:
        return "留一交叉验证：本次未执行（训练点不足 / 骨架不可编译 / 数据缺失）。"
    parts = [f"留一交叉验证：{loo['n_points']} 折（成功 {loo['n_ok']}），"
             f"LOO MSE={loo['mse']:.6g}"]
    if loo.get("nmse") is not None:
        parts.append(f"NMSE={loo['nmse']:.6g}")
    if loo.get("median_abs_err") is not None:
        parts.append(f"|误差|中位数={loo['median_abs_err']:.6g}")
        parts.append(f"95分位={loo['p95_abs_err']:.6g}")
    return "，".join(parts) + "。LOO 只报告，不参与选择。"


def render_loo_section(loo: dict | None) -> str:
    """渲染 report.md 的「留一交叉验证」小节（机器生成）。

    与「样本外验证」并列：训练点太少（< :data:`LOO_MAX_TRAIN`）时，held-out 切不出可信
    规模，本小节用留一法给出样本外口径，并**显式声明 n 小、不构成泛化能力声明**。
    """
    lines = [LOO_HEADING, ""]
    if not loo:
        lines.append(f"本次未执行留一交叉验证（不满足条件：训练点 ≥ {LOO_MAX_TRAIN}、"
                     f"或骨架不可编译、或数据缺失）。")
        return "\n".join(lines)

    n, ok = loo["n_points"], loo["n_ok"]
    lines.append(f"训练数据仅 **{n}** 个点（< {LOO_MAX_TRAIN}）：切不出可信的独立 held-out"
                 f"（本类数据集的 `test.csv` 只有 2 行且落在训练区间内，属插值），"
                 f"故改用**留一法**——每次留出 1 个点、用其余 {n - 1} 个点"
                 f"**重新拟合参数**，再预测被留出的点；重复 {n} 次。")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 折数 / 成功折数 | {n} / {ok} |")
    lines.append(f"| LOO MSE | {loo['mse']:.6g} |")
    nmse = loo.get("nmse")
    nmse_txt = "不可用" if nmse is None else format(nmse, ".6g")
    lines.append(f"| LOO NMSE（分母=训练集方差 {loo['train_var']:.6g}） | {nmse_txt} |")
    for label, key, spec in (("逐点绝对误差 中位数", "median_abs_err", "{:.6g}"),
                             ("逐点绝对误差 95 分位", "p95_abs_err", "{:.6g}"),
                             ("逐点相对误差 中位数", "median_rel_err", "{:.2%}"),
                             ("逐点相对误差 95 分位", "p95_rel_err", "{:.2%}")):
        value = loo.get(key)
        lines.append(f"| {label} | {'不可用' if value is None else spec.format(value)} |")
    if ok < n:
        lines.append(f"| **拟合失败折数** | {n - ok}（该折预测记 NaN，计入上方统计的分母以外） |")

    lines.append("")
    lines.append("逐点明细（每行即一折：用其余点拟合后对该点的预测）：")
    lines.append("")
    var_names = list(loo["rows"][0]["variables"]) if loo["rows"] else []
    lines.append("| # | " + " | ".join(var_names) + " | 观测 | 预测 | 绝对误差 | 相对误差 |")
    lines.append("|" + "---|" * (len(var_names) + 5))
    for i, row in enumerate(loo["rows"], 1):
        vals = " | ".join(f"{row['variables'][v]:.6g}" for v in var_names)
        pred = "NaN" if not np.isfinite(row["predicted"]) else f"{row['predicted']:.6g}"
        ae = "NaN" if not np.isfinite(row["abs_err"]) else f"{row['abs_err']:.6g}"
        re_ = "NaN" if not np.isfinite(row["rel_err"]) else f"{row['rel_err']:.2%}"
        lines.append(f"| {i} | {vals} | {row['observed']:.6g} | {pred} | {ae} | {re_} |")

    lines.append("")
    lines.append("> 口径：LOO 只衡量「去掉一个点后能否补上」，是**插值式**泛化"
                 "（被留出的点通常仍落在其余点包围范围内），**不是外推**。")
    lines.append(f"> **n 很小（{n} 点），本小节不构成泛化能力声明，也不做 OOD**；"
                 f"指标只用于报告，不参与采样、打分、早停与样本选择。")
    return "\n".join(lines)


def strip_loo_section(text: str) -> str:
    """去掉正文里自带的「留一交叉验证」小节（数字一律由系统生成）。"""
    return strip_section(text, LOO_HEADING)


__all__ = [
    "LOO_MAX_TRAIN", "LOO_HEADING",
    "LOO_FIT_BOUNDS", "LOO_FIT_N_STARTS", "LOO_FIT_MAX_ITER", "LOO_FIT_SEED",
    "skeleton_callable", "evaluate_loo", "format_loo_summary",
    "render_loo_section", "strip_loo_section",
]