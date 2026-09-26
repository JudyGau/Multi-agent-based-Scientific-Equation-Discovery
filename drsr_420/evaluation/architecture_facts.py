"""历史采样架构地形：把"已经试过哪些函数结构"从 LLM 的自述收回到代码。

角色
----
评测层的**确定性诊断计算**（与 :mod:`drsr_420.evaluation.data_facts` 同类），但对象
不是数据集而是**本实验已经评估过的方程**。它回答三个只有代码能回答的问题：

* 当前最好分由哪个架构取得、此后多少个样本没再改进；
* 已评估样本里有多少是**同一架构的重新参数化**（换个记号/平移不算新架构）；
* 当前架构的"删一项"邻域里，哪些**从未被评估过**。

为什么需要它
------------
实测 ``MRFCompress-Cuboid_20260926-110809``（同数据第二次运行）的 87 个有评分样本：
52 个（60%）是**同一个完整二阶响应面**（两个变量的平方项 + 交叉项），彼此只差记号
与参数化（``(u,v)`` 平移、``(log u, log v)``、``(λ-1)/(λ+1)``…）；而**删掉一个平方项**
的 5 参数非对称形式一次都没被提出——它恰是该实验里分数最高的干净骨架。采样器把
"换个记号"当成了探索，整期锁死在同一架构。（post-mortem 时用文本模式匹配的粗估计是
45 个，偏低；漏项的来源正是下面"和式别名"那条——多行右值只截到第一行。）

注入"已试过哪些架构 + 最小的未试邻域"这类机器事实，比只暴露罚分更准：罚分只说明
当前解不好，不说明该往哪个结构走。

结论必须由代码给出
------------------
"从未评估过"是对**已评估集合**的陈述，必须逐样本核对。本模块对每个样本做**结构
指纹**（表达式拆成加性项，归到"输入变量的一维变换的幂次"），再对指纹集合做差集。
解析不出的样本计入 ``n_unparsed`` 并显式披露，绝不静默当成"没试过"。

指纹口径（刻意粗、但对记号不变）
--------------------------------
* 先把函数体里的**别名**解析成三类原子：``U0``/``U1``（只依赖某一个输入变量的一维
  变换，含平移 ``λ-1``、缩放、``log``、``(λ-1)/(λ+1)``）与 ``W``（同时依赖两者）；
  含两个以上含变量项的**和式**别名（``sigma = p0 + p1*λ12 + …`` 再 ``return sigma``）
  保留文本结构、不压成一个原子——否则 6 个项会被记成 1 个；
* 每个加性项按"幂次对"归到标签：``(1,0)``→线性、``(2,0)``→平方、``(1,1)``→交叉、
  其余高阶→``higher``；指数是**参数**（``x**params[1]``）或 ``np.power(x, p)`` 的项
  另记 ``power(...)``（那是幂律，不是多项式）；
* 于是"平移/缩放/取对数后的二次型"与"原坐标二次型"落进**同一指纹**——这正是要抓的
  东西（换个记号不是换架构）；而"只对乘积 ``λ12·λ23`` 做二次"落进另一个指纹
  （那是一维山脊模型，不是双变量响应面）。

范围
----
只支持**两个自变量**（本项目当前的多变量实验都落在二元）：更多自变量时无法用
"平方项/交叉项"这一对概念刻画邻域，函数直接返回 ``None``/空串（调用方不注入）。

分层：只依赖标准库，供 agents 层的采样提示注入与残差分析提示共用。
"""
from __future__ import annotations

import re
from typing import Sequence

#: 判定"已触底"所需的最少"自最优以来的样本数"。更短时该判断证据不足，不注入。
MIN_STAGNANT_SAMPLES = 5

#: 判定"已试过足够多架构"所需的最少已解析样本数。
MIN_PARSED_SAMPLES = 10

#: 汇总块最多列出的"删一项"未试邻域个数（按被删项由高阶到低阶排序）。
MAX_UNTRY_DELETIONS = 3

#: 结构指纹的原子记号：U0/U1 = 只依赖某个输入变量的一维变换，W = 同时依赖两者，
#: P = 不含输入变量的量（参数/常数）。
_FEATURE_TOKENS = ("U0", "U1")
_COMPOSITE_TOKEN = "W"
_CONST_TOKEN = "P"
_RESERVED = set(_FEATURE_TOKENS) | {_COMPOSITE_TOKEN, _CONST_TOKEN}

_DOCSTRING_RE = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')
_COMMENT_RE = re.compile(r"#[^\n]*")
_ASSIGN_LHS_RE = re.compile(r"^[ \t]*([A-Za-z_]\w*)[ \t]*=(?!=)[ \t]*", re.M)
_RETURN_RE = re.compile(r"\breturn\b")
_DEF_RE = re.compile(r"^[ \t]*def\b", re.M)
_PARAM_INDEX_RE = re.compile(r"params\s*\[\s*\d+\s*\]")
_PARAM_NAME_RE = re.compile(r"params\b")
_POWER_CALL_RE_TMPL = r"np\.\s*power\s*\(\s*{}\s*,\s*([^,)]+)"
_SIGNATURE_RE = re.compile(r"def\s+\w+\s*\(([^)]*)\)")


def _bound(token: str) -> str:
    """匹配一个独立标识符（不落在更长标识符或属性名内部）。"""
    return r"(?<![\w.])" + re.escape(token) + r"(?![\w])"


def features_from_equation(text: str) -> list[str] | None:
    """从方程文本的 ``def ...(...)`` 签名里取自变量名（去注解、去 ``params``）。

    兜底用途：调用方拿不到 :class:`PromptContext` 时（例如离线复算），仍能定出
    变量名做指纹。取不到签名返回 ``None``（调用方按"无法判定"处理，不注入）。
    """
    match = _SIGNATURE_RE.search(str(text or ""))
    if not match:
        return None
    names = []
    for raw in match.group(1).split(","):
        name = raw.split(":")[0].split("=")[0].strip()
        if not name or name == "params" or not name.isidentifier():
            continue
        names.append(name)
    return names or None


# ── 记号化：别名 → 原子 ──────────────────────────────────────
def _apply(expr: str, mapping: dict) -> str:
    """按名字长度倒序替换（长名优先，避免 ``l1`` 吃掉 ``l12`` 的前缀）。

    替换值用 lambda 返回字面文本：展开式里可能带 ``\\`` 之类的字符，直接当
    ``re.sub`` 的替换串会被当成转义序列。
    """
    for name in sorted(mapping, key=len, reverse=True):
        expr = re.sub(_bound(name), lambda _m, value=mapping[name]: value, expr)
    return expr


def _substitution(resolved: dict) -> dict:
    """已解析别名 → 替换文本：原子记号原样用，和式展开式加括号后再代入。"""
    return {name: value if value in _RESERVED else f"({value})"
            for name, value in resolved.items()}


def _has_feature(term: str) -> bool:
    return bool(re.search(_bound(_COMPOSITE_TOKEN), term)) or any(
        re.search(_bound(tok), term) for tok in _FEATURE_TOKENS)


def _resolve_alias(rhs: str, features: Sequence[str], resolved: dict) -> str:
    """把一段右值归到原子（``U0``/``U1``/``W``/``P``）或原样返回和式展开式。

    为什么不能一律压成原子：实测有样本写成 ``sigma = p0 + p1*λ12 + … + p5*λ12*λ23``
    再 ``return sigma``——把整个和式压成一个原子，6 个项就只剩 1 个，架构统计整体
    失真（这些样本会被记成"只有一个交叉项的形式"而其实是完整二次型）。故只有
    **单块**（单变量的一维变换、单乘积/比值、单参数）才压缩成原子；含两个以上
    含变量项的和式保留文本结构。
    """
    text = _apply(rhs, {f: t for f, t in zip(features, _FEATURE_TOKENS)})
    text = _PARAM_INDEX_RE.sub(_CONST_TOKEN, text)
    text = _apply(text, _substitution(resolved))
    features_hit = {j for j, tok in enumerate(_FEATURE_TOKENS)
                    if re.search(_bound(tok), text)}
    if re.search(_bound(_COMPOSITE_TOKEN), text):
        features_hit = {0, 1}
    if len([t for t in _split_terms(text) if _has_feature(t)]) > 1:
        return text                      # 和式：保留结构，由调用方括号化后代入
    if not features_hit:
        return _CONST_TOKEN
    if len(features_hit) == 1:
        return _FEATURE_TOKENS[next(iter(features_hit))]
    return _COMPOSITE_TOKEN


def _assignments(code: str) -> dict:
    """取出 ``name = <右值>`` 的**整条逻辑语句**右值（跨行按括号配平）。

    行内正则取不到跨行右值：实测 ``sigma = (params[0] + params[1]*λ12 …`` 的右值
    继续在后续行，只截到第一行会留下未闭合的 ``(`` → 括号深度非 0 → 整个和式被当成
    一个项（这正是"完整二次型被记成单个交叉项"的另一个成因）。故按括号配平一直读到
    括号归零后的换行；顶层出现 ``return`` 也立即截止（表达式里不可能有 return）。
    """
    found: dict = {}
    for match in _ASSIGN_LHS_RE.finditer(code):
        start = match.end()
        depth = 0
        index = start
        while index < len(code):
            ch = code[index]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth < 0:
                    break
            elif ch == "\n" and depth == 0:
                break
            elif depth == 0 and code.startswith("return", index) and \
                    not re.match(r"\w", code[index + 6:index + 7] or ""):
                break
            index += 1
        found[match.group(1)] = code[start:index].strip()
    return found


def _alias_map(code: str, features: Sequence[str]) -> dict:
    """解析 ``name = expr`` 别名并迭代到不动点（别名可以引用另一个别名）。"""
    raw = _assignments(code)
    resolved: dict = {}
    for _ in range(6):
        changed = False
        for name, rhs in raw.items():
            if name in _RESERVED or name in features:
                continue
            guess = _resolve_alias(rhs, features, resolved)
            if resolved.get(name) != guess:
                resolved[name] = guess
                changed = True
        if not changed:
            break
    return resolved


# ── 项分类 ──────────────────────────────────────────────────
def _token_power(term: str, token: str) -> int | str:
    """项里某个原子的幂次；参数指数（``x**p``/``np.power(x, p)``）返回 ``"param"``。

    重复相乘（``x*x``）按出现次数计幂，显式 ``x**k`` 取较大者——这样
    ``U0*U0`` 与 ``U0**2`` 都算平方。
    """
    pattern = _bound(token)
    occurrences = len(re.findall(pattern, term))
    if not occurrences:
        return 0
    power: int | str = occurrences
    for match in re.finditer(pattern + r"\s*\*\*\s*(\d+)", term):
        power = max(int(power), int(match.group(1)))
    if re.search(pattern + r"\s*\*\*\s*\(?\s*(?![\d\s])", term):
        return "param"
    # 括号化的底数：``(U1 + P*U0)**P`` / ``(U0 - 1.0)**2``——指数的语义取决于底数里
    # 有谁，故必须看**含本记号的括号组**，而不是紧邻的 ``**``。
    for match in re.finditer(r"\(([^()]*)\)\s*\*\*\s*\(?\s*([^)\s]+)", term):
        group, exponent = match.groups()
        if not re.search(pattern, group):
            continue
        if exponent.isdigit():
            power = max(int(power), int(exponent))
        else:
            return "param"
    for match in re.finditer(_POWER_CALL_RE_TMPL.format(pattern), term):
        arg = match.group(1).strip()
        if arg.isdigit():
            power = max(int(power), int(arg))
        else:
            return "param"
    return power


def _term_label(term: str, names: Sequence[str]) -> str:
    """一个加性项的标签（见模块 docstring 的指纹口径）。"""
    p0 = _token_power(term, _FEATURE_TOKENS[0])
    p1 = _token_power(term, _FEATURE_TOKENS[1])
    pw = _token_power(term, _COMPOSITE_TOKEN)
    n0, n1 = names
    if p0 == 0 and p1 == 0 and pw == 0:
        return "const"
    if "param" in (p0, p1, pw):
        if pw or (p0 and p1):
            return f"power({n0},{n1})"
        return f"power({n1})" if p1 else f"power({n0})"
    deg0, deg1 = int(p0) + int(pw), int(p1) + int(pw)
    if (deg0, deg1) == (1, 0):
        return n0
    if (deg0, deg1) == (0, 1):
        return n1
    if (deg0, deg1) == (2, 0):
        return f"{n0}^2"
    if (deg0, deg1) == (0, 2):
        return f"{n1}^2"
    if (deg0, deg1) == (1, 1):
        return f"{n0}*{n1}"
    return "higher"


def _unwrap(expr: str) -> str:
    """剥掉**包住整个表达式**的外层括号（``return (a + b)`` 的 ``+`` 才算顶层）。

    实测：多数样本把返回式写成 ``return (p0 + p1*x1 + ...)``，不剥外层括号时所有
    加号都在深度 1，整个表达式被当成一个项。
    """
    text = expr.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        for i, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0 or (depth == 0 and i != len(text) - 1):
                    return text
        if depth != 0:
            return text
        text = text[1:-1].strip()
    return text


def _split_terms(expr: str) -> list[str]:
    """按顶层 ``+``/``-`` 切成加性项（括号内的符号不算；``**-1``、``1e-3`` 不切）。"""
    expr = _unwrap(expr)
    terms: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in expr:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if depth == 0 and ch in "+-" and "".join(current).strip():
            head = "".join(current).rstrip()
            if head.endswith(("*", "/", "(")) or re.search(r"[0-9][eE]$", head):
                current.append(ch)
                continue
            terms.append(head)
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        terms.append(tail)
    return terms


def architecture_fingerprint(equation_text: str, features: Sequence[str]) -> tuple[str, ...] | None:
    """结构指纹：加性项标签的排序去重元组；解析不出返回 ``None``。

    只支持两个自变量（见模块 docstring 的"范围"）。
    """
    if not equation_text or len(list(features)) != 2:
        return None
    code = _COMMENT_RE.sub(" ", _DOCSTRING_RE.sub(" ", str(equation_text)))
    match = _RETURN_RE.search(code)
    if not match:
        return None
    tail = _DEF_RE.split(code[match.end():], maxsplit=1)[0]
    mapping = {f: t for f, t in zip(features, _FEATURE_TOKENS)}
    mapping.update(_substitution(_alias_map(code, features)))
    text = _apply(tail, mapping)
    text = _PARAM_INDEX_RE.sub(_CONST_TOKEN, text)
    text = _PARAM_NAME_RE.sub(_CONST_TOKEN, text)
    labels = [_term_label(term, list(features)) for term in _split_terms(text)]
    labels = [label for label in labels if label]
    return tuple(sorted(set(labels))) or None


# ── 汇总 ────────────────────────────────────────────────────
def family_of(terms: Sequence[str], names: Sequence[str]) -> str:
    """粗粒度族标签：用于"同一架构被反复重试"的计数。

    判序刻意先看**平方项的个数**再看交叉项：只保留一个平方项的 5 参数非对称形式
    （本实验里分数最高的干净骨架）必须与"只有一个交叉项"的乘积模型分开计——前者
    是"二阶不足以支撑两个平方项"的候选，后者是一维山脊模型。
    """
    tags = set(terms)
    n0, n1 = names
    square0, square1 = f"{n0}^2", f"{n1}^2"
    cross = f"{n0}*{n1}"
    if {square0, square1, cross} <= tags:
        return "second_order_surface"
    if square0 in tags or square1 in tags:
        return "partial_second_order"
    if cross in tags:
        return "coupled_not_second_order"
    if "higher" in tags:
        return "higher_order"
    if any(tag.startswith("power(") for tag in tags):
        return "power_law"
    return "other"


def _family_phrase(family: str) -> str:
    return {
        "second_order_surface": ("a second-order surface with BOTH squared terms and "
                                 "the cross term"),
        "partial_second_order": ("a second-order surface that keeps only ONE of the two "
                                 "squared terms"),
        "coupled_not_second_order": "a coupled but not full second-order form",
        "higher_order": "a purely higher-order form",
        "power_law": "a power-law form (at least one parametric exponent)",
        "other": "a form with no squared or coupled terms",
    }.get(family, family)


def _term_rank(label: str, names: Sequence[str]) -> int:
    """删项建议的优先级：先动高阶/结构性项，最后才动线性项与常数项。"""
    if label in ("higher",) or label.startswith("power("):
        return 3
    if label.endswith("^2") or label == f"{names[0]}*{names[1]}":
        return 2
    if label == "const":
        return 0
    return 1


def sampling_terrain(entries: Sequence[dict], features: Sequence[str],
                     *, target: str | None = None) -> dict:
    """汇总已评估样本的架构地形。

    Args:
        entries: 经验条目（``experiences.json`` 的元素），需要 ``equation``、
            ``score``（无数字分值的条目被跳过）与可选的 ``sample_order``。
        features: 自变量名（长度必须为 2，否则返回空地形）。
        target: 要刻画其"删一项邻域"的方程文本；``None`` 时用**分数最高**的样本
            （采样通道用默认，残差通道传当前被分析的方程）。

    Returns:
        dict（结构见 :func:`render_terrain`；缺数据时各项为空/0，渲染器自行决定是否注入）。
    """
    names = list(features or [])
    empty = {"ok": False, "n_scored": 0, "n_parsed": 0, "n_unparsed": 0,
             "target_terms": None, "untried_deletions": []}
    if len(names) != 2:
        return empty

    records = []
    for index, entry in enumerate(entries or []):
        if not isinstance(entry, dict):
            continue
        score = entry.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        order = entry.get("sample_order")
        order = int(order) if isinstance(order, (int, float)) and not isinstance(order, bool) \
            else index + 1
        records.append({"score": float(score), "order": order,
                        "terms": architecture_fingerprint(entry.get("equation"), names)})
    parsed = [r for r in records if r["terms"]]
    if not parsed:
        return empty

    best = max(parsed, key=lambda r: r["score"])
    current_order = max(r["order"] for r in records)
    since = [r for r in parsed if r["order"] > best["order"]]
    best_family = family_of(best["terms"], names)

    target_terms = tuple(architecture_fingerprint(target, names)) if target else best["terms"]
    target_is_best = bool(target_terms) and tuple(target_terms) == tuple(best["terms"])
    if not target_terms:
        target_terms = best["terms"]
        target_is_best = True
    target_terms = tuple(target_terms)
    target_family = family_of(target_terms, names)

    tried = {r["terms"] for r in parsed}
    deletions = []
    for label in sorted(target_terms, key=lambda x: (-_term_rank(x, names), x)):
        candidate = tuple(sorted(t for t in target_terms if t != label))
        if candidate in tried:
            continue
        deletions.append({"dropped": label, "terms": list(candidate)})
        if len(deletions) >= MAX_UNTRY_DELETIONS:
            break

    return {
        "ok": True,
        "features": names,
        "n_scored": len(records),
        "n_parsed": len(parsed),
        "n_unparsed": len(records) - len(parsed),
        "best_score": round(best["score"], 6),
        "best_order": best["order"],
        "current_order": current_order,
        "stagnant_samples": current_order - best["order"],
        "since_samples": len(since),
        "since_same_family": sum(1 for r in since if family_of(r["terms"], names) == best_family),
        "since_term_sets": len({r["terms"] for r in since}),
        "best_terms": list(best["terms"]),
        "best_family": best_family,
        "best_family_count": sum(1 for r in parsed
                                 if family_of(r["terms"], names) == best_family),
        "best_family_variants": len({r["terms"] for r in parsed
                                     if family_of(r["terms"], names) == best_family}),
        "target_terms": list(target_terms),
        "target_is_best": target_is_best,
        "target_family": target_family,
        "target_family_count": sum(1 for r in parsed
                                   if family_of(r["terms"], names) == target_family),
        "target_exact_count": sum(1 for r in parsed if r["terms"] == target_terms),
        "untried_deletions": deletions,
    }


# ── 渲染 ────────────────────────────────────────────────────
def render_terrain(terrain: dict, features: Sequence[str], title: str) -> str:
    """把地形渲染成注入提示词的文本块；证据不足或无法判定时返回空串。

    两个闸门（任一不满足就不注入）：已解析样本数 ≥ :data:`MIN_PARSED_SAMPLES`、
    自最优以来 ≥ :data:`MIN_STAGNANT_SAMPLES` 个样本。"该架构已触底"只在有足够
    样本时才说得出口。
    """
    if not terrain or not terrain.get("ok"):
        return ""
    if terrain.get("n_parsed", 0) < MIN_PARSED_SAMPLES:
        return ""
    if terrain.get("stagnant_samples", 0) < MIN_STAGNANT_SAMPLES:
        return ""
    names = list(features)
    parsed = terrain["n_parsed"]
    since = ""
    if terrain["since_samples"]:
        since = (f" ({terrain['since_same_family']} of them in the best sample's term-set "
                 f"family, {terrain['since_term_sets']} distinct term sets in total)")

    lines = [f"scored samples: {terrain['n_scored']} (structurally parsed: {parsed}"
             + (f", unparsed: {terrain['n_unparsed']}" if terrain["n_unparsed"] else "")
             + f") | best score: {terrain['best_score']} (higher is better) at sample_order "
             f"{terrain['best_order']} | samples since that best: "
             f"{terrain['stagnant_samples']}{since}"]

    where = ("the best sample's term set" if terrain["target_is_best"]
             else "the term set of the equation under analysis")
    lines.append(f"{where}: {{{', '.join(terrain['target_terms'])}}}")
    lines.append(
        f"{terrain['target_family_count']} of {parsed} scored samples "
        f"({100.0 * terrain['target_family_count'] / parsed:.0f}%) are in that family - "
        f"{_family_phrase(terrain['target_family'])}; this exact term set has already been "
        f"evaluated {terrain['target_exact_count']} time(s), and a different notation or "
        "parameterization of it is the SAME architecture, not a new one")

    if terrain["untried_deletions"]:
        lines.append("term sets NEVER evaluated that are ONE term away from it:")
        for item in terrain["untried_deletions"]:
            lines.append(f"  drop {item['dropped']} -> {{{', '.join(item['terms'])}}}")
        lines.append(
            "Another re-parameterization of the term set above stays at the same score "
            "floor; the deletions listed above are structurally different and unexplored.")
    else:
        lines.append(
            "Every one-term deletion of that term set has already been evaluated: the "
            "next structural step must remove or replace TWO terms at once, or change the "
            "functional form of an existing one.")
    return title + "\n".join(lines) + "\n"