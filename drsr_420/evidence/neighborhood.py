"""未试邻域的**代表元参数化与实测**：把"还没试过的骨架"变成可拟合的方程并真跑一遍。

角色归属
--------
``evaluation`` 层"架构地形"子系统的**测量内核**：把 :mod:`~drsr_420.equations.text_algebra`
给出的**项集合（term set）**还原成一条可执行方程，用评估器同口径真拟合一次，得到实测
NMSE 与体检罚分。

为什么需要它（实测依据）
----------------------
``MRFCompress-Cuboid`` 的同数据 A/B（``110809`` → ``151008``）显示：提示里只列"未试邻域"
时，点名的 term set **一次都没被采纳**（88 个样本里 0 个用 ``drop λ12²`` 的非对称二次），
因为模型把"结构性事实"读成了"别回那个家族"。同时最优分从 NMSE 2.1e-4 退化到 8.4e-3。
故这里补的是**证据**而不是**建议**：对每个未试邻域用评估器同口径（同 bounds / 多起点
least_squares / 同残差清洗）真拟合一遍，把实测 NMSE 与该参数化的体检罚分写进提示
（"这个 term set 从没评估过；确定性拟合 NMSE=2.65e-3，你当前最好干净解 8.4e-3"）。

无法机械还原的标签（``higher`` / ``power(...)`` 这类把多个形状归并在一起的）**显式降级**，
绝不静默当成"没试过"——那是把"测不出来"说成"没测"。

与 :mod:`~drsr_420.equations.text_algebra` 的分工
--------------------------------------------------
那边只跟字符串与 AST 打交道（记号化、指纹、编译）；本模块负责"拿指纹/项集合去**拟合**"，
因此需要 :func:`drsr_420.execution.problems.evaluate` 与
:func:`drsr_420.equations.pathology.dynamic_range_check`。两条注入通道（采样提示 / 残差分析）
共用本模块的 :func:`measure_term_set`。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from drsr_420.equations.pathology import dynamic_range_check
from drsr_420.evidence.facts import BASELINE_SEED
from drsr_420.equations.text_algebra import representative_from_text
from drsr_420.execution.problems import evaluate

#: 代表元拟合的固定随机种子：同一实验的实测 NMSE 必须可复现
#: （``BASELINE_SEED`` 与 ``data_facts`` 的基线拟合同源，两者不可漂移）。
FIT_SEED = BASELINE_SEED

# ── 未试邻域的代表元参数化与实测 NMSE ──────────────────────────
def _plan_terms(terms: Sequence[str], names: Sequence[str], texts=None):
    """把标签序列变成"代表元"构造计划；无法唯一还原时返回 ``(None, 原因)``。

    计划项是 ``{kind, coef, exp, fmt}``：``coef`` 是该片段消费的**系数**参数下标，
    ``exp`` 是它消费的**指数**参数下标（带参数指数的 ``power(...)`` 才用第二个）；
    ``kind == "verbatim"`` 的项额外带 ``used``（消费的参数个数，含系数）、``text``
    与 ``fn``。

    为什么是"代表元"而不是原式：标签对记号不变（平移/取对数后的二次型与原坐标
    二次型同标签），故从标签还原必然要挑一个代表。多项式族挑恒等记号是精确的
    （仿射重参数化张成同一线性空间 → 最小二乘最优完全相同）；``log`` 记号与
    ``higher`` / ``power(λ12,λ23)`` 不是——后者把多个形状归并成一个标签。

    ``texts``（见 :func:`term_texts`）给出"该标签在**目标方程里**的实际项文本"。
    白名单没有的标签**优先走它**：那不是猜标签，而是读模型自己写的那一项，于是
    ``higher`` / ``power(...)`` 也能量出实数（缺陷 2）。文本越界（用了白名单外的
    构造）时仍降级，并在原因里写明是"文本不可求值"而不是"标签不可还原"。
    """
    n0, n1 = names
    kinds = {
        "const": ("const", 1, "p{i}"),
        n0: ("linear0", 1, f"p{{i}}*{n0}"),
        n1: ("linear1", 1, f"p{{i}}*{n1}"),
        f"{n0}^2": ("square0", 1, f"p{{i}}*{n0}**2"),
        f"{n1}^2": ("square1", 1, f"p{{i}}*{n1}**2"),
        f"{n0}*{n1}": ("cross", 1, f"p{{i}}*{n0}*{n1}"),
        f"{n0}^2*{n1}": ("mixed01", 1, f"p{{i}}*{n0}**2*{n1}"),
        f"{n0}*{n1}^2": ("mixed10", 1, f"p{{i}}*{n0}*{n1}**2"),
        f"{n0}^3": ("cubic0", 1, f"p{{i}}*{n0}**3"),
        f"{n1}^3": ("cubic1", 1, f"p{{i}}*{n1}**3"),
        f"power({n0})": ("power0", 2, f"p{{i}}*{n0}**p{{e}}"),
        f"power({n1})": ("power1", 2, f"p{{i}}*{n1}**p{{e}}"),
    }
    plan: list[dict] = []
    index = 0
    for label in terms:
        entry = kinds.get(label)
        if entry is None:
            text = (texts or {}).get(label)
            if text is None:
                return None, (f"label '{label}' merges several different forms, so its term "
                              "set cannot be reconstructed unambiguously")
            compiled, bound, used, reason = representative_from_text(text, names, offset=index)
            if compiled is None:
                return None, (f"label '{label}' was read from the equation's own term text, "
                              f"but that text cannot be evaluated here: {reason}")
            plan.append({"kind": "verbatim", "coef": index, "exp": None, "used": used,
                         "text": bound, "fn": compiled, "fmt": "p{i}"})
            index += used
            continue
        kind, used, fmt = entry
        plan.append({"kind": kind, "coef": index,
                     "exp": index + 1 if used == 2 else None, "fmt": fmt})
        index += used
    if not plan:
        return None, "the term set is empty"
    return plan, index


def template_from_terms(terms: Sequence[str], names: Sequence[str],
                        texts=None) -> tuple[str, int] | None:
    """由 term set 标签机械构造参数化模板与参数个数；无法唯一还原时返回 ``None``。

    例：``{const, λ12, λ23, λ23^2, λ12·λ23}`` → ``p0 + p1*λ12 + p2*λ23 + p3*λ23**2
    + p4*λ12*λ23``（5 个参数）。

    ``texts`` 见 :func:`_plan_terms`：给了它，``higher`` / ``power(...)`` 就按**目标方程
    里的实际项文本**写出模板，而不是因为"标签归并了多个形状"降级。
    """
    plan, extra = _plan_terms(terms, names, texts)
    if plan is None:
        return None
    parts = []
    for item in plan:
        if item["kind"] == "verbatim":
            # 该项文本已经带着自己的系数（编号是全局连续的），再加前缀会读成两个系数
            parts.append(item["text"])
        else:
            parts.append(item["fmt"].format(i=item["coef"], e=item["exp"]))
    return " + ".join(parts), extra


def _equation_from_terms(terms: Sequence[str], names: Sequence[str], texts=None):
    """由标签构造 ``equation(*columns, params)``；返回 ``(equation, n_params, 原因)``。

    指数是**连续参数**（不用 :func:`eval`/``exec``）：白名单片段按 kind 分派计算；
    ``verbatim`` 片段由 :func:`representative_from_text` 的 AST 白名单求值器提供，
    同样不执行任意代码。
    """
    plan, n_params = _plan_terms(terms, names, texts)
    if plan is None:
        return None, None, n_params

    def equation(*args):
        columns, params = args[:-1], np.asarray(args[-1], dtype=float)
        col0, col1 = columns[0], columns[1]
        total = None
        for item in plan:
            coef = params[item["coef"]]
            kind = item["kind"]
            if kind == "verbatim":
                # 编号已全局连续（见 _bind_params 的 offset），故吃完整参数向量
                value = item["fn"](*columns, params)
            elif kind == "const":
                value = np.ones_like(col0) * coef
            elif kind == "linear0":
                value = coef * col0
            elif kind == "linear1":
                value = coef * col1
            elif kind == "square0":
                value = coef * col0 ** 2
            elif kind == "square1":
                value = coef * col1 ** 2
            elif kind == "cross":
                value = coef * col0 * col1
            elif kind == "mixed01":                 # λ12²·λ23
                value = coef * col0 ** 2 * col1
            elif kind == "mixed10":                 # λ12·λ23²
                value = coef * col0 * col1 ** 2
            elif kind == "cubic0":
                value = coef * col0 ** 3
            elif kind == "cubic1":
                value = coef * col1 ** 3
            else:                                   # power0 / power1：带参数指数的幂律
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    value = coef * (col0 if kind == "power0" else col1) ** params[item["exp"]]
            total = value if total is None else total + value
        return total

    return equation, n_params, None


def _criteria(info: dict) -> list[str]:
    """体检详情 → 命中的判据短语（英文，直接进提示词）。"""
    hits = []
    if info.get("span_penalty"):
        hits.append(f"output span {info['span_ratio']:.4g}x the data range "
                    f"(limit {info['limit']})")
    if info.get("slope_penalty"):
        hits.append(f"local slope {info['slope_max']:.4g} (limit {info['slope_limit']})")
    if info.get("coef_penalty"):
        hits.append(f"coefficient scale {info['coef_ratio']:.4g}x the data range "
                    f"(limit {info['coef_limit']})")
    return hits


def _as_xy(data):
    """从事实表 dict 取出 ``(X, y)``；表不存在/列数不符时返回 ``(None, None)``。

    只认 ``data_facts.json`` 的形状（``table_included`` + ``table_columns`` +
    ``table_rows``）：行数超过 :data:`~drsr_420.evidence.facts.MAX_TABLE_ROWS`
    时事实表**故意不写全表**（避免把抽样出来的部分伪装成全部数据），此时也无从
    拟合，同样降级。
    """
    if not isinstance(data, dict):
        return None, None
    columns = data.get("table_columns") or []
    rows = data.get("table_rows") or []
    if not data.get("table_included") or len(columns) < 3 or not rows:
        return None, None
    try:
        table = np.asarray(rows, dtype=float)
    except (TypeError, ValueError):
        return None, None
    if table.ndim != 2 or table.shape[1] != len(columns):
        return None, None
    return table[:, :-1], table[:, -1]


def measure_term_set(terms: Sequence[str], names: Sequence[str], facts,
                     *, seed: int = FIT_SEED, texts=None) -> dict:
    """用**评估器同口径**拟合某个 term set 的代表元，返回实测 NMSE 与体检。

    口径与 :func:`drsr_420.execution.problems.evaluate` 完全一致（同 bounds、多起点、
    同残差清洗），拟合调用显式 ``range_check=False``：``nmse`` 只反映**拟合质量**，
    体检罚分单独给在 ``penalty``/``criteria`` 里——两类数字混在一起会把"某形式能拟合
    到 X"凭空抬高（见 :mod:`drsr_420.evidence.facts` 的口径说明）。

    失败/不可还原一律落在 ``reason`` 上（调用方必须显式披露），不抛异常。
    ``texts``（见 :func:`term_texts`）让 ``higher`` / ``power(...)`` 这类"标签归并了多个
    形状"的项按**目标方程里的实际文本**被测——这是缺陷 2 的修法，缺它时最前沿的那个族
    一条实测都给不出。

    Returns:
        dict：``terms`` / ``template`` / ``n_params`` / ``mse`` / ``nmse`` /
        ``penalty`` / ``score`` / ``flagged`` / ``criteria`` / ``reason``。
        ``score`` 与实验里的评分同口径（``-(mse + penalty)``），故可直接与
        :func:`sampling_terrain` 的 ``best_score`` 比较——这正是"把未试邻域变成
        可验证的改进方向"所需的那个数。
    """
    entry = {"terms": [str(t) for t in terms], "template": None, "n_params": None,
             "mse": None, "nmse": None, "penalty": None, "score": None, "flagged": None,
             "criteria": [], "reason": None}
    equation, n_params, reason = _equation_from_terms(list(terms), list(names), texts)
    if equation is None:
        entry["reason"] = reason
        return entry
    entry["n_params"] = n_params
    entry["template"] = template_from_terms(terms, names, texts)[0]

    inputs, outputs = _as_xy(facts)
    if inputs is None:
        entry["reason"] = ("this run has no full data table in data_facts.json "
                           "(too many rows or no table), so the fit cannot be reproduced")
        return entry
    if inputs.shape[1] != 2:
        entry["reason"] = f"expected 2 independent variables, got {inputs.shape[1]}"
        return entry

    try:
        score, matrix, params = evaluate(
            {"inputs": inputs, "outputs": outputs}, equation,
            n_params=n_params, seed=seed, verbose=False, range_check=False)
        if score is None or matrix is None or params is None:
            entry["reason"] = "the least-squares fit did not converge"
            return entry
        mse = float(np.mean(np.square(np.asarray(matrix[:, -1], dtype=float))))
        variance = float(np.var(outputs))
        entry["mse"] = mse
        entry["nmse"] = (mse / variance) if variance > 0 else None
        info = dynamic_range_check(
            inputs, outputs, lambda *cols: equation(*cols, np.asarray(params)),
            params=np.asarray(params),
            probe_fn=lambda *args: equation(*args[:-1], np.asarray(args[-1])))
        entry["penalty"] = float(info.get("penalty") or 0.0)
        entry["score"] = -(entry["mse"] + entry["penalty"])
        entry["flagged"] = entry["penalty"] > 0
        entry["criteria"] = _criteria(info)
    except Exception as exc:                        # 单条邻域失败不影响整份地形
        entry["reason"] = f"{type(exc).__name__}: {exc}"
    return entry
