"""剪枝的量化评估：剪枝前后表达式在**训练数据**上的拟合对比 + 剪枝实质判定。

角色归属
--------
收尾分析（analysis）阶段的评估工具，服务于两处：

* ``find_best_eq.prune_and_visualize``：把对比结果放进剪枝摘要（控制台 + explain 提示词）；
* ``explain``：解释 LLM 要论证"剪枝合理"，必须有实测数值支撑（剪枝前后 MSE 变化、
  最大逐点偏差）。只凭公式长相论证"这些项可以去掉"是不可验证的空话——而用户明确
  要求 explain.md 解释剪枝过程与合理性。

除了拟合对比，本模块还提供"剪枝到底做了什么"的判定（``classify_pruning``）：
``nodes_pruned`` 是"是否真剪枝"的唯一硬判据，而 ``simplify`` 只做通分/展开时公式
**数学上没变**——这种情况必须如实说明并沿用原式，否则 explain.md 会被逼着解释一次
并不存在的剪枝（见 ``max_relative_difference`` 关于假阴性的说明）。

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
import re

import numpy as np
import sympy as sp

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: 实验目录名 ``<问题名>_<YYYYMMDD-HHMMSS>``：没有 config_snapshot.json 的历史目录
#: 靠它反推数据集（见 :func:`infer_data_csv`）。
_RUN_DIR_RE = re.compile(r"^(?P<problem>.+)_\d{8}-\d{6}$")

#: 判定"剪枝是否只是换了写法"（而非真的删掉项）的相对误差阈值。
#: 实测 36 次真实运行：仅形式变化的样本最大相对差 ≤ 9.6e-13，真剪枝的样本 ≥ 7.6e-3
#: —— 中间有 10 个数量级的安全间隔，1e-6 两侧都留足余量。
FORM_ONLY_RTOL = 1e-6

#: 形式等价校验的采样点数与随机种子（种子固定，同一实验可复现）。
VERIFY_SAMPLES = 200
VERIFY_SEED = 20260919

#: 已提示过的"数据源/列名兜底"信息：一次收尾里 load_training_data / compare_fits /
#: 曲线 / 样本外验证会各调一遍，不去重的话同一句话会刷 4 遍（run.out 已经很长）。
_warned: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        print(f"[WARN] {message}")


def resolve_csv(data_csv: str, results_root: str = "") -> str | None:
    """data_csv 依次按 results_root、项目根、cwd 解析；兼容绝对路径。

    config_snapshot 里通常存项目根相对路径（./data/...），自包含实验目录
    （如测试夹具）则是 results_root 相对路径——两处都要试。

    返回值统一 ``normpath``：快照里的 "./data/X/train.csv" 与目录拼接后会得到
    "…\\./data/X\\train.csv" 这种混合分隔符，直接进日志/产物说明很难看。
    """
    if os.path.isabs(data_csv):
        return os.path.normpath(data_csv) if os.path.isfile(data_csv) else None
    for base in (results_root, _REPO_ROOT, os.getcwd()):
        p = os.path.join(base, data_csv)
        if os.path.isfile(p):
            return os.path.normpath(p)
    return None


def infer_data_csv(results_root: str) -> str | None:
    """没有 config_snapshot.json 时按目录名推断训练数据：``<问题名>_<时间戳>`` → ``data/<问题名>/train.csv``。

    历史实验目录（用户从别处整理进来的那批）没有 config_snapshot.json，旧实现只能
    放弃——剪枝前后拟合对比与曲线图全部静默缺失。目录名里的问题名唯一，且 ``data/``
    下的数据集是**约定命名**，因此按它兜底能把这批目录重新变成可分析的。

    推断结果只作兜底并会在日志里明说：它终究是"猜"的，不如快照里的记录可靠。
    """
    m = _RUN_DIR_RE.match(os.path.basename(os.path.normpath(results_root)))
    if not m:
        return None
    candidate = os.path.join(_REPO_ROOT, "data", m.group("problem"), "train.csv")
    if os.path.isfile(candidate):
        _warn_once(f"目录里没有 config_snapshot.json，按目录名推断训练数据: {candidate}")
        return candidate
    return None


def load_training_data(results_root: str) -> np.ndarray | None:
    """按 config_snapshot.json 的 data_csv 读取训练数据（结构化数组）；失败返回 None。

    快照缺失或没写 data_csv 时退回 :func:`infer_data_csv` 按目录名推断（历史目录）。
    """
    snap_path = os.path.join(results_root, "config_snapshot.json")
    data_csv = None
    try:
        with open(snap_path, "r", encoding="utf-8") as f:
            data_csv = json.load(f).get("data_csv")
    except Exception as e:
        _warn_once(f"读取 config_snapshot.json 失败: {e}")

    if not data_csv:
        data_csv = infer_data_csv(results_root)
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


def resolve_columns(data: np.ndarray, dependent: str,
                    sym_names: list[str]) -> tuple[str, list[str], str]:
    """把函数头里的变量名对齐到 CSV 的实际列名。

    函数头里的名字是 LLM 写的，未必与 CSV 表头逐字相同（实测历史运行的因变量写成
    ``um``，而 CSV 列是 ``miu``）——旧实现直接按名字取列，取不到就静默跳过拟合对比与
    曲线（``compare_fits`` 返回空字典、``plot_data_curves`` 直接 return）。

    匹配顺序：

    1. 精确匹配；
    2. 忽略大小写（``Sigma`` / ``sigma``）；
    3. 因变量兜底：自变量都能对上时，取"自变量之外的唯一一列"（CSV 约定最后一列是因变量）。

    三档都不成立时抛 ``KeyError``（调用方只告警跳过，不猜）。

    Returns:
        ``(因变量列名, 自变量列名列表, 说明)``；说明非空表示发生了兜底匹配，供日志明说
        用了哪一列（不能悄悄换列）。
    """
    names = list(getattr(getattr(data, "dtype", None), "names", None) or ())
    lower = {n.lower(): n for n in names}
    notes: list[str] = []

    def _match(name: str, kind: str) -> str | None:
        if name in names:
            return name
        hit = lower.get(str(name).lower())
        if hit is not None:
            notes.append(f"{kind} {name!r} → CSV 列 {hit!r}（忽略大小写）")
        return hit

    ind_cols: list[str] = []
    for sym in sym_names:
        col = _match(sym, "自变量")
        if col is None:
            raise KeyError(f"数据里没有自变量列 {sym!r}（现有列：{names}）")
        ind_cols.append(col)

    dep_col = _match(dependent, "因变量")
    if dep_col is None:
        rest = [n for n in names if n not in ind_cols]
        if len(rest) == 1:
            dep_col = rest[0]
            notes.append(f"因变量 {dependent!r} → CSV 列 {dep_col!r}"
                         f"（按位置兜底：自变量之外的唯一一列）")
        else:
            raise KeyError(f"数据里没有因变量列 {dependent!r}（现有列：{names}）")
    return dep_col, ind_cols, "；".join(notes)


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
    # 样本内的最大误差：explain.md 的「样本外验证」小节要用它与 held-out 同口径对比
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
                            seed: int = VERIFY_SEED) -> float | None:
    """两条表达式在采样网格上的最大相对差；无有效采样点（全非有限）时返回 None。

    **这是"公式到底变没变"的权威判据。** ``sp.simplify(a - b) == 0`` 与 ``a.equals(b)``
    在含浮点指数的表达式上都会给出**假阴性**：实测某次 0 项剪枝的样本
    ``simplify(差) == 0`` 与 ``equals()`` 都是 False，而它在 8 个训练数据点与 100 个
    采样点上的相对差恰为 0（``equals`` 会在负实数/复数域上取样，浮点指数的分支不同）。
    本函数只在与剪枝决策**同一个有效定义域**（``sample_range``）上比较。
    """
    if not sym_names:
        return None
    args = sample_points(list(sym_names), sample_range, num_samples, seed)
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
                     sample_range=(1.0, 14.0)) -> dict:
    """判定本次"剪枝"的实质，返回可直接写进日志/解释提示词的证据字典。

    判定分层（``nodes_pruned`` 是唯一硬判据，数值比较只用来论证"形式变化"）::

        nodes_pruned > 0                        → 'pruned'（真剪枝，公式用剪枝结果）
        nodes_pruned == 0 且 simplify 没改写形式 → 'none'（什么都没变）
        nodes_pruned == 0 且 simplify 改写了形式 → 'form_only'（通分/重排，公式已回退原式）
        同上但数值等价性无法确认/不成立           → 'form_only_unverified'（同样回退，需人工看一眼）

    为什么需要它：``SensitivityPruner.prune`` 在 0 项剪枝时会返回原表达式，但**日志与
    explain.md 仍要如实说明**"simplify 本来会把它改写成什么形式、那只是通分"，
    否则读者会以为公式被剪枝改变了（也可能反过来怀疑剪枝没生效）。

    返回的 ``max_rel_diff`` 是**判定所依据的那一次比较**：真剪枝时是"剪枝结果 vs 原式"
    （剪枝改变了多少模型，应当明显非零）；0 项剪枝时是"simplify 会给出的形式 vs 原式"
    （只是换写法，应当≈0）。

    Args:
        expr: 剪枝前（参数已代入）的表达式。
        published: ``SensitivityPruner.prune`` 返回、即将对外发布的表达式。
        stats: 同一次剪枝的 ``PruneStats``。
        sym_names / sample_range: 形式等价校验的变量与采样区间，应与剪枝参数一致。
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
            expr, compared, list(sym_names), sample_range)
    if out["max_rel_diff"] is not None:
        out["numerically_equivalent"] = bool(out["max_rel_diff"] <= FORM_ONLY_RTOL)

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
