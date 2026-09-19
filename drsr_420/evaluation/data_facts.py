"""数据事实表：把"可判定的量"从 LLM 手里收回到代码。

角色
----
评测层的**确定性诊断计算**。给定一份数据集（inputs/outputs）算出：

* 逐列统计（n、min/max/mean/std）与相关结构（线性 / 秩 / 对数空间）；
* 因变量的极值点（全局最大/最小落在哪一行）——这是"峰在哪"的唯一权威答案；
* 一组机械生成的候选骨架各自的 NMSE（用与评估器**完全相同**的拟合口径，
  即 :func:`drsr_420.evaluation.problems.evaluate` 的多起点有界 least_squares）；
* 自变量之间的共线性/可辨识性告警。

为什么需要它
------------
实测中模型会自行改写数据（把 ``(lambda12=2, lambda23=8.933)`` 复述成
``(2, 2)``）、会把先验当结论（断言"压缩应力由 lambda12*lambda23 支配"，
而实测乘积骨架 NMSE 是分别幂律的 8 倍）、还会把全局极值说错（说峰在
``lambda12=2``，实际最大值在 ``lambda12=1``）。这些都是**代码一算就知道**的量，
不该交给 LLM 自由发挥。本模块产出的文本块注入分析提示词，让模型的每个数值
断言都有出处，也让"物理先验"必须与实测基线对质。

产物只进**分析阶段**（初次分析 + 每轮残差分析），不进每条采样提示——那张表
约几百字符，进采样提示会按样本数线性放大 token 消耗。
"""
from __future__ import annotations

import json
import os

import numpy as np

from drsr_420.evaluation.problems import evaluate


#: 行数不超过它就把**完整数据表**写进事实表（模型引用数字时的唯一合法出处）。
#: 超过则不写表，只给统计量与极值点，避免把"抽样出来的一部分"伪装成全部数据。
MAX_TABLE_ROWS = 40

#: 判为"近乎共线/不可辨识"的相关系数阈值（线性或对数空间任一命中即告警）。
COLLINEAR_R = 0.98

#: 子集脊检测要求保留的最少点数：少于它相关系数失去意义。
_MIN_SUBSET_ROWS = 4


# ── 取值与相关 ──────────────────────────────────────────────
def _rank(values: np.ndarray) -> np.ndarray:
    """平均秩（Spearman 用）。不引入 scipy.stats，保持本模块只依赖 numpy。"""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # 并列值取平均秩，避免秩相同却算出不同的相关系数
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """皮尔逊相关系数；任一列方差为 0（常数）时返回 None。"""
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """秩相关（对单调非线性关系比皮尔逊更合适，且不受量纲影响）。"""
    return _pearson(_rank(a), _rank(b))


def _round(value, digits: int = 4):
    """数字保留位数；None/非数字原样返回（便于直接渲染与 JSON 落盘）。"""
    if isinstance(value, (int, float)) and np.isfinite(value):
        return round(float(value), digits)
    return value


# ── 候选骨架 ────────────────────────────────────────────────
def _linear_all(*args):
    """a*x1 + b*x2 + ... + c（只依赖自变量的一次项）。"""
    xs, p = args[:-1], args[-1]
    return sum(p[i] * xs[i] for i in range(len(xs))) + p[len(xs)]


def _single_power(j: int):
    """a*xj^b + c（只看第 j 个自变量）。"""

    def fn(*args):
        xs, p = args[:-1], args[-1]
        return p[0] * xs[j] ** p[1] + p[2]

    return fn


def _product_power(*args):
    """a*(x1*x2*...)^b + c（整体乘积支配：即"长细比 L1/L3"型先验）。"""
    xs, p = args[:-1], args[-1]
    prod = xs[0]
    for x in xs[1:]:
        prod = prod * x
    return p[0] * prod ** p[1] + p[2]


def _separate_power(*args):
    """a*x1^b*x2^c + d（两个指数各自独立，与乘积型先验形成对照）。"""
    xs, p = args[:-1], args[-1]
    return p[0] * xs[0] ** p[1] * xs[1] ** p[2] + p[3]


def _ratio_power(*args):
    """a*(x1/x2)^b + c（比值型）。"""
    xs, p = args[:-1], args[-1]
    return p[0] * (xs[0] / xs[1]) ** p[1] + p[2]


def _product_plus_sum(*args):
    """a*(x1*x2) + b*(x1+x2) + c（乘积之外还有加性分离项）。"""
    xs, p = args[:-1], args[-1]
    return p[0] * (xs[0] * xs[1]) + p[1] * (xs[0] + xs[1]) + p[2]


def _skeleton_candidates(names: list[str]):
    """机械生成候选骨架列表（label, callable）。

    只放"约 7 条"的固定小批：覆盖乘积 / 分别幂律 / 单变量幂律 / 线性 / 比值 /
    乘积+求和，正好能反驳"乘积支配"这类先验。刻意不做组合爆炸——每条骨架都要
    用评估器的口径真跑一遍拟合，表越长提示词越贵，收益却递减。
    二元变量时才生成比值/分别幂律/乘积+求和（这些形状对三元变量没有明确含义）。
    """
    n = len(names)
    letters = "abcde"
    linear_label = " + ".join(f"{letters[i]}*{name}" for i, name in enumerate(names)) + " + const"
    cands = [(linear_label, _linear_all)]
    for j, name in enumerate(names):
        cands.append((f"a*{name}^b + c", _single_power(j)))
    if n >= 2:
        cands.append(("a*({})^b + c".format("*".join(names)), _product_power))
    if n == 2:
        cands.append((f"a*{names[0]}^b*{names[1]}^c + d", _separate_power))
        cands.append((f"a*({names[0]}/{names[1]})^b + c", _ratio_power))
        cands.append((f"a*({names[0]}*{names[1]}) + b*({names[0]}+{names[1]}) + c",
                      _product_plus_sum))
    return cands


def skeleton_baselines(inputs, outputs, feature_names, dependent_name,
                       *, seed: int | None = None) -> list[dict]:
    """用评估器同口径拟合候选骨架，返回 ``[{expression, nmse, r2}]``（NMSE 升序）。

    口径必须与 :func:`problems.evaluate` 一致：同 bounds、多起点、least_squares、
    同样的残差清洗。这样"骨架基线"与实验里真实打分的分数可比——若另起一套拟合，
    表里的 NMSE 就无法用来质疑模型的先验。
    """
    X = np.asarray(inputs, dtype=float)
    y = np.asarray(outputs, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    var_y = float(np.var(y)) if y.size else 0.0

    rows = []
    for label, fn in _skeleton_candidates(list(feature_names)):
        entry = {"expression": label, "nmse": None, "r2": None}
        try:
            score, _matrix, _params = evaluate(
                {"inputs": X, "outputs": y}, fn, seed=seed, verbose=False)
        except Exception as exc:            # 单条骨架失败不影响整张表
            entry["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(entry)
            continue
        if score is not None and var_y > 0:
            nmse = -float(score) / var_y
            entry["nmse"] = _round(nmse, 4)
            entry["r2"] = _round(1.0 - nmse, 4)
        rows.append(entry)

    rows.sort(key=lambda r: (r["nmse"] is None, r["nmse"] if r["nmse"] is not None else 0.0))
    return rows


# ── 事实表 ──────────────────────────────────────────────────
def compute_facts(inputs, outputs, feature_names, dependent_name,
                  *, max_table_rows: int = MAX_TABLE_ROWS,
                  seed: int | None = None,
                  with_skeletons: bool = True) -> dict:
    """计算完整事实表（逐列统计、相关、极值点、骨架基线、可辨识性）。

    Args:
        inputs: ``(n_samples, n_features)`` 数组。
        outputs: ``(n_samples,)`` 因变量。
        feature_names: 自变量名（长度须与列数一致，用于渲染与相关性对名）。
        dependent_name: 因变量名。
        max_table_rows: 行数不超过它才写完整数据表。
        with_skeletons: 是否跑候选骨架基线（跑一次要 N 次 least_squares，
            大数据集上可按需关掉）。

    Returns:
        dict（可直接 ``json.dump``）。
    """
    X = np.asarray(inputs, dtype=float)
    y = np.asarray(outputs, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    names = list(feature_names)
    dep = dependent_name or "y"

    columns = {}
    for j, name in enumerate(names):
        col = X[:, j]
        columns[name] = {"min": _round(np.min(col)), "max": _round(np.max(col)),
                         "mean": _round(np.mean(col)), "std": _round(np.std(col))}
    columns[dep] = {"min": _round(np.min(y)), "max": _round(np.max(y)),
                    "mean": _round(np.mean(y)), "std": _round(np.std(y))}

    # 相关结构：自变量两两（线性/秩/对数）+ 每个自变量与因变量
    all_positive = bool(np.all(X > 0)) and bool(np.all(y > 0))
    correlations = []
    pairs = [(names[i], names[j], X[:, i], X[:, j])
             for i in range(len(names)) for j in range(i + 1, len(names))]
    pairs += [(name, dep, X[:, j], y) for j, name in enumerate(names)]
    for a_name, b_name, a, b in pairs:
        entry = {
            "a": a_name, "b": b_name,
            "pearson": _round(_pearson(a, b)),
            "spearman": _round(_spearman(a, b)),
            "log_pearson": _round(_pearson(np.log(a), np.log(b))) if all_positive else None,
        }
        correlations.append(entry)

    # 全局极值点：直接回应"峰在哪"这类断言（实测模型把峰说在 lambda12=2，
    # 而全局最大值在 lambda12=1 —— 代码一算就知道）
    extremes = {}
    if y.size:
        i_max = int(np.argmax(y))
        i_min = int(np.argmin(y))
        extremes = {
            "max": {"value": _round(y[i_max]),
                    "at": {n: _round(X[i_max, j]) for j, n in enumerate(names)}},
            "min": {"value": _round(y[i_min]),
                    "at": {n: _round(X[i_min, j]) for j, n in enumerate(names)}},
        }

    facts = {
        "n_rows": int(X.shape[0]),
        "features": names,
        "dependent": dep,
        "columns": columns,
        "correlations": correlations,
        "monotonicity": _monotonicity(names, X, y),
        "extremes": extremes,
        "table_included": int(X.shape[0]) <= int(max_table_rows),
        "table_columns": names + [dep],
        "table_rows": [[_round(v) for v in row] for row in np.column_stack((X, y))] 
        if int(X.shape[0]) <= int(max_table_rows) else [],
        "skeletons": skeleton_baselines(X, y, names, dep, seed=seed) if with_skeletons else [],
        "identifiability": _identifiability(names, correlations, X),
    }
    return facts


def _monotonicity(names: list[str], X: np.ndarray, y: np.ndarray) -> list[dict]:
    """逐个自变量检查因变量是否单调；非单调时给出第一处反转的具体数据点。

    为什么必须由代码判定：实测分析文本把 λ23 说成"Monotone increase with lambda23"，
    而它自己列出的数字里 σ(λ23=3.9174)=306.577 > σ(λ23=4.8446)=296.651 就是一处反转
    ——相关系数高（0.83）不等于单调。这类断言会被注入每条采样提示，必须在源头拦住。
    """
    report = []
    for j, name in enumerate(names):
        order = np.argsort(X[:, j], kind="mergesort")
        xs, ys = X[order, j], y[order]
        direction = 0
        reversals = 0
        first = None
        for i in range(len(ys) - 1):
            delta = np.sign(ys[i + 1] - ys[i])
            if delta == 0:
                continue
            if direction == 0:
                direction = delta
            elif delta != direction:
                reversals += 1
                if first is None:
                    first = {"from": {name: _round(xs[i]), "dependent": _round(ys[i])},
                             "to": {name: _round(xs[i + 1]), "dependent": _round(ys[i + 1])}}
        entry = {
            "feature": name,
            "monotone": reversals == 0,
            "direction": ("increasing" if direction > 0 else
                          "decreasing" if direction < 0 else "flat"),
            "reversals": reversals,
        }
        if first is not None:
            entry["first_reversal"] = first
        report.append(entry)
    return report


def _identifiability(names: list[str], correlations: list[dict], X=None) -> list[dict]:
    """自变量之间近乎共线时给出"指数不可辨识"告警。

    数据设计常把自变量沿一条一维曲线采样（实测 MRFCompress-Cuboid 的 7 个点
    在 ln 空间 r=-0.9996）。此时两个指数的**分配**在数学上不可辨识，最终公式里
    谁大谁小不该被解释成独立发现——必须在提示词里说清楚，否则 explain.md 会把
    拟合出来的一对指数当成物理结论。

    判据分两层：全样本相关（线性/对数空间）命中即告警；全样本不命中时再看
    "剔除某列端点组后的子集"（``_subset_collinearity``）。第二层是必需的——
    实测本数据集全样本 log_pearson 只有 0.0924（被两个 lambda12=1 的杠杆点稀释），
    而剔除它们后的 6 个点在 ln 空间 |r|=0.9996，只看全样本就漏报。
    """
    warnings = []
    for c in correlations:
        if c["a"] == c["b"] or c["b"] not in names or c["a"] not in names:
            continue
        r = c.get("pearson")
        r_log = c.get("log_pearson")
        hit = [x for x in (r, r_log) if isinstance(x, (int, float)) and abs(x) >= COLLINEAR_R]
        if hit:
            space = "log space" if isinstance(r_log, (int, float)) and abs(r_log) >= COLLINEAR_R \
                else "linear space"
            warnings.append({
                "a": c["a"], "b": c["b"], "pearson": r, "log_pearson": r_log,
                "message": (
                    f"{c['a']} and {c['b']} are nearly collinear on this dataset "
                    f"(|r|={abs(max(hit, key=abs)):.4f} in {space}). Their exponents are NOT "
                    "separately identifiable: do not present their split as an independent "
                    "physical finding, and do not claim one drives the response more than the "
                    "other unless a single-variable baseline shows it."
                ),
            })
            continue
        sub = _subset_collinearity(names.index(c["a"]), names.index(c["b"]), X, names)
        if not sub:
            continue
        warnings.append({
            "a": c["a"], "b": c["b"], "pearson": r, "log_pearson": r_log, "subset": True,
            "message": (
                f"{c['a']} and {c['b']} are NOT collinear over the full dataset, but the "
                f"{sub['rows']} points left after removing {sub['column']}={sub['value']} lie on "
                f"a nearly one-dimensional ridge (|r|={sub['abs_r']:.4f} in {sub['space']}). On "
                "that subset their exponents are NOT separately identifiable: do not present "
                "their split as an independent physical finding, and do not claim one drives the "
                "response more than the other unless a single-variable baseline shows it."
            ),
        })
    return warnings


def _subset_collinearity(i: int, j: int, X, names: list[str]) -> dict | None:
    """剔除某一列的端点取值组后，剩余点是否反而几乎共线（"子集脊"）。

    全样本相关系数会被"被单独扫描的那一列取值"稀释：本数据集 8 行里两个 lambda12=1
    的点把 lambda23 从 1 扫到 14.12（乘积分别是 1 和 14.12，远离其余 6 点的 17.9–19.6），
    于是整样本对数空间相关只有 0.0924；去掉这两行后其余 6 点在 ln 空间 |r|=0.9996，
    指数分配在该子集上完全不可辨识。

    只枚举"某列取最小值/最大值的全部行"这一种剔除（端点组通常正是那个单独扫描组），
    既覆盖实测情形，也避免任意子集组合的爆炸。
    """
    if X is None:
        return None
    best = None
    for col in (i, j):
        colv = X[:, col]
        lo, hi = float(np.min(colv)), float(np.max(colv))
        if lo == hi:                      # 常数列：剔除会删空，没有子集可言
            continue
        for value in (lo, hi):
            mask = colv != value
            if int(mask.sum()) < _MIN_SUBSET_ROWS:
                continue
            a, b = X[mask, i], X[mask, j]
            cands = []
            r = _pearson(a, b)
            if r is not None:
                cands.append((abs(r), r, "linear space"))
            if np.all(a > 0) and np.all(b > 0):
                r_log = _pearson(np.log(a), np.log(b))
                if r_log is not None:
                    cands.append((abs(r_log), r_log, "log space"))
            for mag, r_val, space in cands:
                if mag >= COLLINEAR_R and (best is None or mag > best["abs_r"]):
                    best = {"column": names[col], "value": _round(value),
                            "rows": int(mask.sum()), "abs_r": mag,
                            "correlation": _round(r_val), "space": space}
    return best


# ── 渲染 ────────────────────────────────────────────────────
#: 事实表区块标题（注入分析提示词；"measured by code" 是给模型的权威性信号）。
FACTS_BLOCK_TITLE = (
    "\n\n### The following data facts were measured by code (same optimizer as the "
    "evaluator). Treat them as authoritative: this is the only source you may quote "
    "numbers from. ###\n\n"
)


def render_facts(facts: dict) -> str:
    """把事实表渲染成紧凑文本块（供提示词注入）。"""
    if not facts:
        return ""
    lines = []
    dep = facts.get("dependent", "y")
    lines.append(f"rows: {facts.get('n_rows')} | features: {', '.join(facts.get('features', []))} "
                 f"| dependent: {dep}")
    # 单调性判定放在最前面：实测模型会写"Monotone increase with lambda23"，而数据里
    # 明明有一处反转。相关系数高不等于单调，这条必须由代码给出结论。
    for m in facts.get("monotonicity") or []:
        if m.get("monotone"):
            lines.append(f"monotonicity: {dep} is monotone {m.get('direction')} in "
                         f"{m.get('feature')} on this dataset")
            continue
        rev = m.get("first_reversal") or {}
        frm, to = rev.get("from", {}), rev.get("to", {})
        lines.append(
            f"monotonicity: {dep} is NOT monotone in {m.get('feature')} "
            f"({m.get('reversals')} reversal(s); first at {m.get('feature')}="
            f"{frm.get(m.get('feature'))}->{to.get(m.get('feature'))}: "
            f"{frm.get('dependent')}->{to.get('dependent')}). Do not describe it as "
            "a monotone/saturating trend without acknowledging this.")
    lines.append("column stats:")
    for name, st in (facts.get("columns") or {}).items():
        lines.append(f"  {name}: min={st['min']} max={st['max']} mean={st['mean']} std={st['std']}")

    ex = facts.get("extremes") or {}
    if ex.get("max"):
        at = ", ".join(f"{k}={v}" for k, v in ex["max"]["at"].items())
        lines.append(f"global {dep} maximum: {ex['max']['value']} at ({at})")
    if ex.get("min"):
        at = ", ".join(f"{k}={v}" for k, v in ex["min"]["at"].items())
        lines.append(f"global {dep} minimum: {ex['min']['value']} at ({at})")

    lines.append("correlations (pearson / spearman / log-space pearson):")
    for c in facts.get("correlations") or []:
        lines.append(f"  {c['a']} vs {c['b']}: {c['pearson']} / {c['spearman']} / {c['log_pearson']}")

    if facts.get("table_included"):
        lines.append(f"complete data table (all {facts.get('n_rows')} rows -- do NOT paraphrase, "
                     "re-round, or invent rows beyond this table):")
        lines.append("  " + ", ".join(facts.get("table_columns") or []))
        for row in facts.get("table_rows") or []:
            lines.append("  " + ", ".join(str(v) for v in row))

    if facts.get("skeletons"):
        lines.append("candidate skeleton baselines (fitted with the evaluator's own optimizer; "
                     "lower NMSE is better, so a physical prior that contradicts this ranking "
                     "must be reported as a conflict, not restated as fact):")
        for s in facts["skeletons"]:
            nmse = s.get("nmse")
            shown = "failed" if nmse is None else f"NMSE={nmse} R2={s.get('r2')}"
            lines.append(f"  {s['expression']} -> {shown}")

    for w in facts.get("identifiability") or []:
        lines.append(f"identifiability warning: {w['message']}")

    return FACTS_BLOCK_TITLE + "\n".join(lines) + "\n"


# ── 落盘 ────────────────────────────────────────────────────
#: 事实表在实验目录里的文件名（写入方在 agents 层，读取方见 prompt_injection / explain）。
FACTS_FILENAME = "data_facts.json"


def facts_path(results_root: str | None) -> str:
    """实验目录下事实表的路径。"""
    return os.path.join(results_root or ".", FACTS_FILENAME)


def load_facts(results_root: str | None) -> dict:
    """读取事实表；缺失或损坏时返回 ``{}``（调用方静默降级，不阻塞主流程）。"""
    try:
        with open(facts_path(results_root), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def extract_xy(data_instance) -> tuple[np.ndarray, np.ndarray] | None:
    """从数据实例里取出 (inputs, outputs)；不支持的结构返回 ``None``。

    兼容 ``{'data': {'inputs':..., 'outputs':...}}``（pipeline 传给 agent 的实例）
    与 ``{'inputs':..., 'outputs':...}``（problems.evaluate 的入参）两种形状。
    """
    if not isinstance(data_instance, dict):
        return None
    payload = data_instance.get("data") if isinstance(data_instance.get("data"), dict) \
        else data_instance
    if not isinstance(payload, dict) or "inputs" not in payload or "outputs" not in payload:
        return None
    try:
        X = np.asarray(payload["inputs"], dtype=float)
        y = np.asarray(payload["outputs"], dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if X.size == 0 or y.size == 0:
        return None
    return X, y