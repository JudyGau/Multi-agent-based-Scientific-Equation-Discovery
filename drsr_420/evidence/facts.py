"""数据事实表：把"可判定的量"从 LLM 手里收回到代码。

角色
----
评测层的**确定性诊断计算**。给定一份数据集（inputs/outputs）算出：

* 逐列统计（n、min/max/mean/std）与相关结构（线性 / 秩 / 对数空间）；
* 因变量的极值点（全局最大/最小落在哪一行）——这是"峰在哪"的唯一权威答案；
* 一组机械生成的候选骨架各自的 NMSE（用与评估器**完全相同**的拟合口径，
  即 :func:`drsr_420.execution.problems.evaluate` 的多起点有界 least_squares）
  与**体检标记**（该形式的最优拟合本身是否靠角点门控/尖峰取得）；
* 自变量之间的共线性/可辨识性告警。

为什么需要它
------------
实测中模型会自行改写数据（把 ``(lambda12=2, lambda23=8.933)`` 复述成
``(2, 2)``）、会把先验当结论（断言"压缩应力由 lambda12*lambda23 支配"，
而实测乘积骨架 NMSE 是分别幂律的 8 倍）、还会把全局极值说错（说峰在
``lambda12=2``，实际最大值在 ``lambda12=1``）。这些都是**代码一算就知道**的量，
不该交给 LLM 自由发挥。本模块产出的文本块注入分析提示词，让模型的每个数值
断言都有出处，也让"物理先验"必须与实测基线对质。

基线 NMSE 的口径（两类数字不许混）
----------------------------------
``nmse`` = **拟合本身**的残差平方均值（残差列直接算），不含任何"选择惩罚"；
``flagged`` + ``note`` = 该最优拟合是不是靠局部化器件（角点门控/尖峰）取得的。
两者分开的理由：罚分是**选择用的偏好**，不是**拟合质量**。混在一起会出现
"某个先验形式看起来能拟合到 X"的假象——实测 ``a*(λ12λ23)^b+c`` 的最小二乘最优
是 ``b=-877``（把输出钉在 λ12=1 的尖峰），``a*λ12^b+c`` 的最优是 ``b=-2739``：
它们的 NMSE 是数值器件的上限，不代表该形式的能力。因此拟合调用显式
``range_check=False``，表中**不保留**含罚分的旧口径数字（那种混合口径正是本表
此前的缺陷，历史值只在本注释里留档）。

产物只进**分析阶段**（初次分析 + 每轮残差分析），不进每条采样提示——那张表
约几百字符，进采样提示会按样本数线性放大 token 消耗。
"""
from __future__ import annotations

import json
import os

import numpy as np

from drsr_420.equations.pathology import dynamic_range_check
from drsr_420.execution.problems import evaluate


#: 行数不超过它就把**完整数据表**写进事实表（模型引用数字时的唯一合法出处）。
#: 超过则不写表，只给统计量与极值点，避免把"抽样出来的一部分"伪装成全部数据。
MAX_TABLE_ROWS = 40

#: 判为"近乎共线/不可辨识"的相关系数阈值（线性或对数空间任一命中即告警）。
COLLINEAR_R = 0.98

#: 子集脊检测要求保留的最少点数：少于它相关系数失去意义。
_MIN_SUBSET_ROWS = 4

#: 基线拟合的随机起点种子。事实表是"代码实测的权威数字"，必须**可复现**：
#: 多起点随机起点会让同一条形式在不同实验里给出不同 NMSE——实测
#: ``a*λ12^b*λ23^c+d`` 偶尔收敛到与乘积形式重合的退化解（b≈c，把乘积当作
#: 整体长细比），NMSE 由 0.0307 跳到 0.1698，而这张表正是用来对质"乘积支配"
#: 先验的。数字随实验抖动会让该论证失去意义，故固定种子；调用方仍可显式覆盖。
BASELINE_SEED = 0


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


def _baseline_note(patho: dict) -> str:
    """被体检判为病理的基线的提示语（英文，注进提示词的候选骨架表）。

    措辞必须让模型无法把它当成"该形式的能力"：说清 NMSE 是**数值器件**换来的
    上限，并给出是哪条判据、数值多少——否则模型会拿它去论证先验被数据证实/否证。
    """
    why = []
    if patho.get("span_penalty"):
        why.append(f"output span {patho['span_ratio']:.4g}× the data range "
                   f"(limit {patho['limit']})")
    if patho.get("slope_penalty"):
        why.append(f"local slope {patho['slope_max']:.4g} (limit "
                   f"{patho['slope_limit']})")
    if patho.get("coef_penalty"):
        why.append(f"coefficient scale {patho['coef_ratio']:.4g}× the data range "
                   f"(limit {patho['coef_limit']})")
    return ("FLAGGED: this NMSE was reached by a localized/gating device ("
            + "; ".join(why) + "), not by a legitimate instance of the form — "
            "treat it as an artifact ceiling, not as this form's capability.")


def skeleton_baselines(inputs, outputs, feature_names, dependent_name,
                       *, seed: int | None = None) -> list[dict]:
    """用评估器同口径拟合候选骨架，返回 ``[{expression, nmse, r2, ...}]``（NMSE 升序）。

    口径必须与 :func:`problems.evaluate` 一致：同 bounds、多起点、least_squares、
    同样的残差清洗。这样"骨架基线"与实验里真实打分的分数可比——若另起一套拟合，
    表里的 NMSE 就无法用来质疑模型的先验。

    但 ``nmse`` **取拟合本身的均方误差**（由返回矩阵的残差列直接算），不取
    ``evaluate`` 的分数：分数自体检接入后含病理罚分，而罚分是"选择偏好"不是
    "拟合质量"，混进来会把"某形式能拟合到 X"凭空抬高（见模块 docstring 的口径
    说明）。旧口径留在 ``score_nmse``；最优拟合本身若是局部化器件（角点门控/
    尖峰），额外给出 ``flagged`` 与 ``note``——那是这张表最容易被误用的一点。

    每条骨架额外做一次体检求值（向量化的网格求值，相对 least_squares 的开销
    可忽略；``evaluate`` 不返回体检详情，故此处显式复算保持口径自明）。

    Args:
        seed: 多起点随机种子；``None``（含调用方未给）解析为 :data:`BASELINE_SEED`
            ——事实表要可复现，随机起点会让同一形式的 NMSE 在不同实验里抖动。
    """
    X = np.asarray(inputs, dtype=float)
    y = np.asarray(outputs, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    var_y = float(np.var(y)) if y.size else 0.0
    seed = BASELINE_SEED if seed is None else int(seed)

    rows = []
    for label, fn in _skeleton_candidates(list(feature_names)):
        entry = {"expression": label, "nmse": None, "r2": None}
        try:
            # range_check=False：这里只要**拟合**结果，体检由下面显式做一次
            # （不重复求值、也不把罚分混进这条基线的任何数字）。
            score, matrix, params = evaluate(
                {"inputs": X, "outputs": y}, fn, seed=seed, verbose=False,
                range_check=False)
            if score is not None and matrix is not None and params is not None:
                mse = float(np.mean(np.square(np.asarray(matrix[:, -1], dtype=float))))
                patho = dynamic_range_check(
                    X, y, lambda *cols: fn(*cols, np.asarray(params)),
                    params=np.asarray(params),
                    probe_fn=lambda *args: fn(*args[:-1], np.asarray(args[-1])))
                entry["pathology"] = {
                    "penalty": _round(patho["penalty"]),
                    "span_ratio": _round(patho["span_ratio"]),
                    "limit": patho["limit"],
                    "slope_max": _round(patho["slope_max"]),
                    "slope_limit": patho["slope_limit"],
                    "coef_ratio": _round(patho["coef_ratio"]),
                    "coef_limit": patho["coef_limit"],
                }
                entry["flagged"] = bool(patho["penalty"] > 0)
                if entry["flagged"]:
                    entry["note"] = _baseline_note(patho)
                if var_y > 0:
                    nmse = mse / var_y
                    entry["nmse"] = _round(nmse, 4)
                    entry["r2"] = _round(1.0 - nmse, 4)
        except Exception as exc:            # 单条骨架失败不影响整张表
            entry["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(entry)
            continue
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
        seed: 基线拟合的多起点种子；``None`` 解析为 :data:`BASELINE_SEED`（可复现）。
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

    **重复取值口径**（20260926-110809 的实测教训）：一个自变量上可以有两个不同的
    因变量值——本数据 λ12=1.0 同时有 193.0543 与 352.1991，即 σ **不是** λ12 的单值
    函数。旧实现按 x 排序后逐对比较，把"同一 x 的竖直跳变"当成一次真实上升，于是报
    「2 处反转、首个在 1.0->2.0」；而任何沿采样路径的读法都只看到 1 处，分析文本与
    事实表因此互相矛盾（提示词还规定事实表是唯一依据）。现在把同一 x 的取值按**区间**
    参与比较：下一点落在该区间内记 ``undetermined_steps``（方向不可判定、不计反转），
    并把该 x 的全部取值记进 ``duplicate_x_groups`` 供下游显式报告。

    注意 ``direction`` 是**最后一段**的方向（历史口径，渲染器只在 ``monotone`` 为真时
    才打印它）：非单调项的 ``direction`` 不代表整体趋势，判定请只看 ``monotone`` /
    ``reversals`` / ``first_reversal``（实测 λ12 非单调但 direction="increasing"，
    因为最后一段是 4.0->5.0 的回升）。
    """
    report = []
    for j, name in enumerate(names):
        order = np.argsort(X[:, j], kind="mergesort")
        xs, ys = X[order, j], y[order]
        # 稳定排序后同一 x 必然相邻，故一次线性扫描即可分组
        groups: list[tuple[float, list[float]]] = []
        for xi, yi in zip(xs, ys):
            if groups and float(xi) == groups[-1][0]:
                groups[-1][1].append(float(yi))
            else:
                groups.append((float(xi), [float(yi)]))
        direction = 0
        reversals = 0
        first = None
        undetermined = 0
        for (x1, g1), (x2, g2) in zip(groups, groups[1:]):
            lo1, hi1 = min(g1), max(g1)
            lo2, hi2 = min(g2), max(g2)
            if lo2 > hi1:
                delta = 1
            elif hi2 < lo1:
                delta = -1
            else:
                # 两组取值区间重叠：这一步的方向取决于取组内哪个值，不敢判定
                undetermined += 1
                continue
            if direction == 0:
                direction = delta
            elif delta != direction:
                reversals += 1
                direction = delta   # 必须锁存新方向：否则一段持续下行/上行会被逐增量重复计数
                if first is None:
                    # 只用**真实数据点**做见证，且取最保守的一对（升：左组最高 → 右组最低）
                    left = hi1 if delta > 0 else lo1
                    right = lo2 if delta > 0 else hi2
                    first = {"from": {name: _round(x1), "dependent": _round(left)},
                             "to": {name: _round(x2), "dependent": _round(right)}}
        entry = {
            "feature": name,
            "monotone": reversals == 0 and undetermined == 0,
            "direction": ("increasing" if direction > 0 else
                          "decreasing" if direction < 0 else "flat"),
            "reversals": reversals,
        }
        if first is not None:
            entry["first_reversal"] = first
        if undetermined:
            entry["undetermined_steps"] = undetermined
        dup = [(x, vals) for x, vals in groups if len(vals) > 1]
        if dup:
            entry["duplicate_x_groups"] = [
                {name: _round(x), "dependent": [_round(v) for v in vals]}
                for x, vals in dup
            ]
        report.append(entry)
    return report


def _identifiability(names: list[str], correlations: list[dict], X=None) -> list[dict]:
    """自变量之间近乎共线时给出"指数不可辨识"告警。

    数据设计常把自变量沿一条一维曲线采样（实测 MRFCompress-Cuboid 的 7 个点
    在 ln 空间 r=-0.9996）。此时两个指数的**分配**在数学上不可辨识，最终公式里
    谁大谁小不该被解释成独立发现——必须在提示词里说清楚，否则 report.md 会把
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
        feature = m.get("feature")
        if m.get("monotone"):
            lines.append(f"monotonicity: {dep} is monotone {m.get('direction')} in "
                         f"{feature} on this dataset")
            continue
        rev = m.get("first_reversal") or {}
        frm, to = rev.get("from", {}), rev.get("to", {})
        extra = ""
        if m.get("undetermined_steps"):
            extra += (f" {m['undetermined_steps']} step(s) are UNDETERMINED because two "
                      f"different {dep} values share one {feature} value (see below).")
        lines.append(
            f"monotonicity: {dep} is NOT monotone in {feature} "
            f"({m.get('reversals')} reversal(s); first at {feature}="
            f"{frm.get(feature)}->{to.get(feature)}: "
            f"{frm.get('dependent')}->{to.get('dependent')}).{extra} Do not describe it as "
            "a monotone/saturating trend without acknowledging this, and do NOT re-derive, "
            "re-count or re-word this verdict yourself: quote these numbers and this count.")
        # 重复 x（同一自变量取值上有多个因变量值）必须显式给出：这等价于"因变量不是该
        # 自变量的单值函数"，本身是重要事实，也是上面 undetermined 步的成因。
        for dup in m.get("duplicate_x_groups") or []:
            vals = ", ".join(str(v) for v in dup.get("dependent") or [])
            lines.append(
                f"  same {feature} value carries different {dep} values: {feature}="
                f"{dup.get(feature)} -> [{vals}] (so {dep} is NOT a single-valued function "
                f"of {feature})")
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
        # FLAGGED 的约束必须写成**双向**的：只说"不得用来论证该形式有能力"，模型会
        # 反过来把它当"先验被否证"的证据。实测 20260925-134149 的初次分析就引用了
        # FLAGGED 行 a*λ12^b+c（NMSE 0.8649，靠门控取得）支撑"σ 不是 λ12 的函数"。
        lines.append("candidate skeleton baselines (fitted with the evaluator's own optimizer; "
                     "NMSE is the fit's own mean-square error with no selection penalty mixed "
                     "in, lower NMSE is better, so a physical prior that contradicts this "
                     "ranking must be reported as a conflict, not restated as fact; a row "
                     "marked FLAGGED reached its NMSE through a localized/gating device and "
                     "must NOT be used as evidence in EITHER direction -- neither that the "
                     "form is capable, nor that the prior behind it is refuted):")
        for s in facts["skeletons"]:
            nmse = s.get("nmse")
            shown = "failed" if nmse is None else f"NMSE={nmse} R2={s.get('r2')}"
            note = f" [{s['note']}]" if s.get("note") else ""
            lines.append(f"  {s['expression']} -> {shown}{note}")

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