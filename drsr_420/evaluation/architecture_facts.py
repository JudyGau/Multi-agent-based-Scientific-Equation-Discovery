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

光说"没试过"还不够（2026-09-26 实测补充）
-----------------------------------------
``MRFCompress-Cuboid`` 的同数据 A/B（``110809`` → ``151008``）显示：只列"未试邻域"
时，点名的 term set **一次都没被采纳**（88 个样本里 0 个用 ``drop λ12²`` 的非对称二次），
因为模型把"结构性事实"读成了"别回那个家族"。同时最优分从 NMSE 2.1e-4 退化到 8.4e-3。
故本模块补两件事（都由代码算出、固定种子可复现）：

* :func:`measure_term_set` —— 对每个未试邻域用**评估器同口径**（同 bounds / 多起点
  least_squares / 同残差清洗）真拟合一遍，把实测 NMSE 与该参数化的体检罚分写进提示
  （"这个 term set 从没评估过；确定性拟合 NMSE=2.65e-3，你当前最好干净解 8.4e-3"）。
  无法机械还原的标签显式降级，绝不静默当成"没试过"；
* 分数分解 —— 提示里给出"最好干净解（罚分=0）MSE=…"与"全局最好拟合 MSE=… 但罚分=…
  → 分被吃掉"的对照，避免模型把"拟合 MSE 0.197 + 罚分 36.06"当成胜利（本轮 order 34
  就是这种样本，观测到的"低 MSE 高罚分 ↔ 高 MSE 零罚分"两模态振荡即源于此）。

2026-09-28 补充：闸门被抖动锁死 + 加项方向缺失
-----------------------------------------------
多种子协议（6 跑，``ab-terrain-{only,measured}``）暴露本模块两处会让处置**在最需要它的
时候静默失效**的设计：

* **"自最优以来"用错了最优**。``max(records, key=score)`` 会让 refit 抖动（同一模型
  重新拟合出的 1e-10 级差异）成为全局最大，把 ``best_order`` 顶到样本前沿。实测
  T2-s33 撞地板后 order 49/62/65/73 的 4 次"刷新"累计只改善 8.6e-10，于是块内自报的
  停滞被重置成 ``samples since that best: 8``（若锚在真正的最后进步应为 33）——闸门
  恰好在该注入时关闭。现在最优由 :func:`significant_best` 定：改善必须超过
  :data:`SIGNIFICANT_IMPROVEMENT_REL` / :data:`SIGNIFICANT_IMPROVEMENT_ABS`。
* **未试邻域只枚举"删项"**。实测唯一成功的逃逸动作是**加项**（非对称二次 + λ23³，
  T2-s22 的 order 79 → MSE 0.271565），而只列删项时该方向既不在列表里、也不可能被
  点名——同一臂的 T2-s33 全程 85 个样本里 ``λ23³`` 一次都没出现，卡在地板上。
  现在 :func:`addition_candidates` 从目标自己的项**升一阶**产生候选，与删项同口径
  实测、同段渲染（``add λ23^3 -> …``）。为此结构指纹把三次单项式**单列标签**
  （``λ23^3`` / ``λ12^2·λ23`` / ``λ12·λ23^2`` …），不再并进 ``higher``——否则
  "这个 term set 从未试过"既认不出来、也无法从标签还原代表元去实测。

⚠️ 闭环约束**不得**写成"禁止该维引入 |指数|>k 的幂律"：本轮达到干净地板 NMSE 2.65e-3
的形式 ``a*(λ23+p·λ12)^b+d·λ12^e+f`` 本身带参数指数，一刀切会把最优干净形式一起禁掉
（且本仓已记录该手段不可泛化）。约束只能落在**可验证的量**上：输出跨度 / 局部斜率 /
系数比 / 实测 NMSE——即本模块与 :mod:`drsr_420.core.range_check` 给出的东西。

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

代表元拟合的局限（必须在提示里说清）
------------------------------------
未试邻域是从**标签**还原的（标签对记号不变，故还原必然丢掉"平移/取对数"这一层）。
多项式族无妨：仿射重参数化（``u = λ12-1``）与恒等记号张成同一线性空间，最小二乘最优
NMSE 精确相同；但 ``log`` 记号不是同一空间，且 ``higher`` / ``power(λ12,λ23)`` 这类标签
把多个不同形状归并在一起、无法唯一还原——这些一律**不猜**，显式记为未测量。

分层
----
依赖 core（:mod:`~drsr_420.core.range_check`、:mod:`~drsr_420.core.sample_records`）与
同层的 :mod:`~drsr_420.evaluation.problems` / :mod:`~drsr_420.evaluation.data_facts`
（拟合口径必须与打分完全一致，另起一套就不可比）。供 agents 层的采样提示注入与残差
分析提示共用。
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from drsr_420.core.range_check import dynamic_range_check
from drsr_420.core.sample_records import load_sample_records, score_breakdown
from drsr_420.evaluation.data_facts import BASELINE_SEED, load_facts
from drsr_420.evaluation.problems import evaluate

#: 判定"已触底"所需的最少"自最优以来的样本数"。更短时该判断证据不足，不注入。
MIN_STAGNANT_SAMPLES = 5

#: 判定"已试过足够多架构"所需的最少已解析样本数。
MIN_PARSED_SAMPLES = 10

#: 汇总块最多列出的"删一项"未试邻域个数（按被删项由高阶到低阶排序）。
#: 每个邻域都要真拟合一遍，故这也是"每份地形最多几次 least_squares"的上界。
MAX_UNTRY_DELETIONS = 3

#: 汇总块最多列出的"加一项"未试邻域个数（按被加项由高阶到低阶排序）。
#: 加项与删项是**两个方向**，不能只做一个：实测本仓唯一成功的逃逸动作就是**加项**
#: （``{const, λ12, λ12·λ23, λ23, λ23²}`` 再加 ``λ23³``，T2-s22 的 order 79 →
#: MSE 0.271565），而只枚举删项时该方向既不在列表里、也不可能被点名（同一臂的
#: T2-s33 全程 85 个样本里 ``λ23³`` 一次都没出现，卡在 0.867473 的地板上）。
#: 取 4 而不是 3：二次目标"升一阶"的全部候选正好是 4 个三次单项式
#: （``λ12³`` / ``λ12²·λ23`` / ``λ12·λ23²`` / ``λ23³``），取 3 会按标签字典序
#: 砍掉其中一个——那正是"按实现细节而不是按结构决定看哪个方向"的老毛病。
MAX_UNTRY_ADDITIONS = 4

#: "实质改善"的容差（相对 / 绝对，取二者较大者）。
#: refit 微抖动不是进步：实测 T2-s33（``20260928-092926``）撞地板后 order 49/62/65/73
#: 的 4 次"刷新"累计只改善 **8.6e-10**（相邻两点差 8.2e-10 / 2.1e-11 / 1.3e-11 /
#: 2.5e-12），却把"最优"一路顶到样本前沿，于是地形块**自报的**停滞被重置成
#: ``samples since that best: 8``（真值应为 33）——闸门恰好在该注入时关闭。
#: 容差取 1e-6，与渲染精度（``round(..., 6)``）同量级：显示上完全相同的两个分数
#: 不构成一次"刷新"。
SIGNIFICANT_IMPROVEMENT_REL = 1e-6
SIGNIFICANT_IMPROVEMENT_ABS = 1e-6

#: 未试邻域实测 NMSE 的随机起点种子。与 :data:`~drsr_420.evaluation.data_facts.BASELINE_SEED`
#: 同源：事实表里的数字必须可复现，多起点随机会让同一个 term set 在不同轮次给出不同
#: NMSE，而这段提示正是用来让模型"照数字去试"的。
FIT_SEED = BASELINE_SEED

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
    # 三次单项式单列标签，不并进 ``higher``：实测唯一成功的逃逸解正是"非对称二次 +
    # λ23³"，并进 ``higher`` 会让它与别的形状同标签 → 既认不出"这个 term set 从未
    # 试过"，也无法从标签还原出代表元去实测（见模块 docstring 的 2026-09-28 补充）。
    if (deg0, deg1) == (3, 0):
        return f"{n0}^3"
    if (deg0, deg1) == (0, 3):
        return f"{n1}^3"
    if (deg0, deg1) == (2, 1):
        return f"{n0}^2*{n1}"
    if (deg0, deg1) == (1, 2):
        return f"{n0}*{n1}^2"
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


def _cubic_labels(names: Sequence[str]) -> set[str]:
    """三次单项式标签集合（供"高阶项"排序与加项候选共用）。"""
    n0, n1 = names
    return {f"{n0}^3", f"{n1}^3", f"{n0}^2*{n1}", f"{n0}*{n1}^2"}


def _term_rank(label: str, names: Sequence[str]) -> int:
    """删项/加项建议的优先级：先动高阶/结构性项，最后才动线性项与常数项。"""
    if label in ("higher",) or label.startswith("power(") or label in _cubic_labels(names):
        return 3
    if label.endswith("^2") or label == f"{names[0]}*{names[1]}":
        return 2
    if label == "const":
        return 0
    return 1


def _monomial_degree(label: str, names: Sequence[str]) -> tuple[int, int] | None:
    """标签对应的 ``(λ12 的次数, λ23 的次数)``；不是单项式标签时返回 ``None``。

    ``higher`` / ``power(...)`` 本身归并了多个形状，"升一阶"无从谈起 → 不产生加项候选。
    """
    n0, n1 = names
    return {
        "const": (0, 0),
        n0: (1, 0), n1: (0, 1),
        f"{n0}^2": (2, 0), f"{n1}^2": (0, 2), f"{n0}*{n1}": (1, 1),
        f"{n0}^3": (3, 0), f"{n1}^3": (0, 3),
        f"{n0}^2*{n1}": (2, 1), f"{n0}*{n1}^2": (1, 2),
    }.get(label)


def _monomial_label(degree: tuple[int, int], names: Sequence[str]) -> str | None:
    """次数对 → 标签；三次以上没有独立标签，返回 ``None``（不猜）。"""
    n0, n1 = names
    return {
        (0, 0): "const",
        (1, 0): n0, (0, 1): n1,
        (2, 0): f"{n0}^2", (0, 2): f"{n1}^2", (1, 1): f"{n0}*{n1}",
        (3, 0): f"{n0}^3", (0, 3): f"{n1}^3",
        (2, 1): f"{n0}^2*{n1}", (1, 2): f"{n0}*{n1}^2",
    }.get(degree)


def addition_candidates(terms: Sequence[str], names: Sequence[str]) -> list[str]:
    """目标 term set 的"升一阶"加项候选，按次数降序 + 标签排序。

    只从**目标自己的项**长候选（``λ23²`` → ``λ23³``、``λ12`` → ``λ12²``、
    ``λ12·λ23`` → ``λ12²·λ23`` / ``λ12·λ23²``），不去枚举任意高阶单项式：实测唯一
    成功的逃逸动作正是"把已有的 λ23² 再升一阶"，而全枚举三次单项式会让真正该看的
    那一项被前几名挤掉。常数项在次数上无从升，故不产生候选。

    排序按次数降序（高次优先）→ 与本仓"逃逸动作 = 给最高次项升阶"的实测一致。
    """
    found: list[str] = []
    for label in terms:
        degree = _monomial_degree(label, names)
        if degree is None or degree == (0, 0):
            continue
        for raised in ((degree[0] + 1, degree[1]), (degree[0], degree[1] + 1)):
            sibling = _monomial_label(raised, names)
            if sibling and sibling not in terms and sibling not in found:
                found.append(sibling)
    return sorted(found, key=lambda label: (-sum(_monomial_degree(label, names)), label))


# ── 未试邻域的代表元参数化与实测 NMSE ──────────────────────────
def _plan_terms(terms: Sequence[str], names: Sequence[str]):
    """把标签序列变成"代表元"构造计划；无法唯一还原时返回 ``(None, 原因)``。

    计划项是 ``{kind, coef, exp, fmt}``：``coef`` 是该片段消费的**系数**参数下标，
    ``exp`` 是它消费的**指数**参数下标（带参数指数的 ``power(...)`` 才用第二个）。

    为什么是"代表元"而不是原式：标签对记号不变（平移/取对数后的二次型与原坐标
    二次型同标签），故从标签还原必然要挑一个代表。多项式族挑恒等记号是精确的
    （仿射重参数化张成同一线性空间 → 最小二乘最优完全相同）；``log`` 记号与
    ``higher`` / ``power(λ12,λ23)`` 不是——后者把多个形状归并成一个标签，
    **不猜**，直接降级（见模块 docstring）。
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
            return None, (f"label '{label}' merges several different forms, so its term set "
                          "cannot be reconstructed unambiguously")
        kind, used, fmt = entry
        plan.append({"kind": kind, "coef": index,
                     "exp": index + 1 if used == 2 else None, "fmt": fmt})
        index += used
    if not plan:
        return None, "the term set is empty"
    return plan, index


def template_from_terms(terms: Sequence[str], names: Sequence[str]) -> tuple[str, int] | None:
    """由 term set 标签机械构造参数化模板与参数个数；无法唯一还原时返回 ``None``。

    例：``{const, λ12, λ23, λ23^2, λ12·λ23}`` → ``p0 + p1*λ12 + p2*λ23 + p3*λ23**2
    + p4*λ12*λ23``（5 个参数）。
    """
    plan, extra = _plan_terms(terms, names)
    if plan is None:
        return None
    return " + ".join(p["fmt"].format(i=p["coef"], e=p["exp"]) for p in plan), extra


def _equation_from_terms(terms: Sequence[str], names: Sequence[str]):
    """由标签构造 ``equation(*columns, params)``；返回 ``(equation, n_params, 原因)``。

    指数是**连续参数**（不用 :func:`eval`/``exec``）：片段由上面的白名单生成、
    变量名来自方程签名（调用方已校验是标识符），故直接按 kind 分派计算即可。
    """
    plan, n_params = _plan_terms(terms, names)
    if plan is None:
        return None, None, n_params

    def equation(*args):
        columns, params = args[:-1], np.asarray(args[-1], dtype=float)
        col0, col1 = columns[0], columns[1]
        total = None
        for item in plan:
            coef = params[item["coef"]]
            kind = item["kind"]
            if kind == "const":
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
    ``table_rows``）：行数超过 :data:`~drsr_420.evaluation.data_facts.MAX_TABLE_ROWS`
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
                     *, seed: int = FIT_SEED) -> dict:
    """用**评估器同口径**拟合某个 term set 的代表元，返回实测 NMSE 与体检。

    口径与 :func:`drsr_420.evaluation.problems.evaluate` 完全一致（同 bounds、多起点、
    同残差清洗），拟合调用显式 ``range_check=False``：``nmse`` 只反映**拟合质量**，
    体检罚分单独给在 ``penalty``/``criteria`` 里——两类数字混在一起会把"某形式能拟合
    到 X"凭空抬高（见 :mod:`drsr_420.evaluation.data_facts` 的口径说明）。

    失败/不可还原一律落在 ``reason`` 上（调用方必须显式披露），不抛异常。

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
    equation, n_params, reason = _equation_from_terms(list(terms), list(names))
    if equation is None:
        entry["reason"] = reason
        return entry
    entry["n_params"] = n_params
    entry["template"] = template_from_terms(terms, names)[0]

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


# ── 输入装载（两条注入通道共用） ────────────────────────────────
def load_terrain_inputs(results_root: str | None) -> dict:
    """读地形计算所需的两个外部输入：事实表与样本记录（缺失即降级为空）。

    * ``facts``：``data_facts.json``（提供训练数据表，未试邻域的实测 NMSE 要用它复现
      评估器口径）；
    * ``records``：``samples/*.json``（提供 ``mse``/``penalty``/``score`` 三元组，
      分数分解要用）。

    任一缺失时下游只输出纯结构事实（这正是旧行为），注入本身绝不能让主流程失败。
    """
    root = results_root or "."
    try:
        facts = load_facts(root)
    except Exception:
        facts = {}
    try:
        records = load_sample_records(root)
    except Exception:
        records = []
    return {"facts": facts if isinstance(facts, dict) else {}, "records": records}


def with_score_breakdown(terrain: dict, records: Sequence[dict]) -> dict:
    """返回带上"分数是怎么来的"分解的**地形副本**（不就地改：地形本身会被整轮缓存复用）。

    单独一步的原因：结构地形（含未试邻域的拟合）只随**已试集合**变化、可以整轮缓存；
    而分数分解随每个新样本变化，必须每轮刷新。
    """
    merged = dict(terrain or {})
    merged["breakdown"] = score_breakdown(list(records or []))
    return merged


def significant_improvement_tolerance(reference_score: float) -> float:
    """判定"实质改善"的容差：``max(绝对, 相对 × |参考分|)``。

    取二者较大者：本仓实测的分数跨 0.27~10³，纯绝对容差在小分上偏松、纯相对容差
    在 0 附近恒为 0。与渲染精度（6 位小数）同量级 ⇒ 显示上分不出差别的两个分数
    不会被当成一次"刷新"。
    """
    try:
        magnitude = abs(float(reference_score))
    except (TypeError, ValueError):
        magnitude = 0.0
    return max(SIGNIFICANT_IMPROVEMENT_ABS, SIGNIFICANT_IMPROVEMENT_REL * magnitude)


def significant_best(records: Sequence[dict]) -> dict:
    """按 ``order`` 走一遍，只在**实质**改善时推进"最优"（不做全局 ``max``）。

    为什么不能用 ``max(records, key=score)``：refit 抖动会让**同一个模型**以 1e-10 的
    优势成为全局最大，于是 ``best_order`` 被顶到样本前沿、"自最优以来" 归零——恰好在
    真正卡住时把地形块关掉（实测见 :data:`SIGNIFICANT_IMPROVEMENT_REL` 的文档）。
    """
    ordered = sorted(records, key=lambda r: (r["order"], -r["score"]))
    best = ordered[0]
    for record in ordered[1:]:
        if record["score"] > best["score"] + significant_improvement_tolerance(best["score"]):
            best = record
    return best


def sampling_terrain(entries: Sequence[dict], features: Sequence[str],
                     *, target: str | None = None, facts=None) -> dict:
    """汇总已评估样本的架构地形。

    Args:
        entries: 经验条目（``experiences.json`` 的元素），需要 ``equation``、
            ``score``（无数字分值的条目被跳过）与可选的 ``sample_order``。
        features: 自变量名（长度必须为 2，否则返回空地形）。
        target: 要刻画其"删一项邻域"的方程文本；``None`` 时用**分数最高**的样本
            （采样通道用默认，残差通道传当前被分析的方程）。
        facts: ``data_facts.json`` 的内容（可选）。给了就用评估器同口径把每个未试
            邻域真拟合一遍并把实测 NMSE 写进地形（见 :func:`measure_term_set`）；
            没给/表不可用则只列 term set、附上显式降级原因。

    Returns:
        dict（结构见 :func:`render_terrain`；缺数据时各项为空/0，渲染器自行决定是否注入）。
    """
    names = list(features or [])
    empty = {"ok": False, "n_scored": 0, "n_parsed": 0, "n_unparsed": 0,
             "target_terms": None, "untried_deletions": [], "untried_additions": []}
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

    best = significant_best(parsed)
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
        # 每个未试邻域真拟合一遍：光说"没试过"实测会被读成"别回那个家族"，
        # 配上可复现的 NMSE 才成为"可验证的改进方向"（见模块 docstring）。
        deletions.append({"dropped": label, "terms": list(candidate),
                          "measurement": measure_term_set(candidate, names, facts)})
        if len(deletions) >= MAX_UNTRY_DELETIONS:
            break

    additions = []
    for label in addition_candidates(target_terms, names):
        candidate = tuple(sorted(list(target_terms) + [label]))
        if candidate in tried:
            continue
        # 与删项同口径：加项邻域也要真拟合一遍，"加这项会变成什么分数"必须可验证。
        additions.append({"added": label, "terms": list(candidate),
                          "measurement": measure_term_set(candidate, names, facts)})
        if len(additions) >= MAX_UNTRY_ADDITIONS:
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
        "untried_additions": additions,
    }


# ── 渲染 ────────────────────────────────────────────────────
def _fmt_number(value) -> str:
    """4 位有效数字；``None``/非数值 → ``n/a``（跨量级的数字必须留有效位数，不取整）。"""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return "n/a"


def _render_measurement(measurement: dict, best_score) -> list[str]:
    """未试邻域的"实测口径 + 实测分数"（或显式降级原因）。

    为什么比较的是**分数**而不是 NMSE：NMSE 只反映拟合质量，"拟合得好但被罚分吃掉"
    与"拟合一般却干净"在评分层是同一条轴上的两个解。实测 110809 的完整二次型拟合
    NMSE 2.11e-4 却带 0.6877 罚分（score −1.113），而"删 λ12²"的非对称二次测下来
    拟合 NMSE 4.31e-4、罚分 0（score −0.867）——**后者分数更好**；只报 NMSE 会把
    这个可验证的改进方向说反。

    只说"没试过"实测会被读成"别回那个家族"（110809→151008 的 88 个样本里被点名的
    term set 0 次采纳），配上"它比你现在最好的分还高"才成为可执行的方向。
    """
    if not measurement:
        return []
    if measurement.get("nmse") is None:
        reason = measurement.get("reason") or "no NMSE was produced"
        return [f"      NOT measured here ({reason}) — still never evaluated"]
    head = (f"      measured on THIS run's training data with the same evaluator "
            f"(deterministic fit, seed {FIT_SEED}, representative parameterization "
            f"`{measurement.get('template')}` using {measurement.get('n_params')} parameters): "
            f"fit MSE {_fmt_number(measurement.get('mse'))} "
            f"(NMSE {_fmt_number(measurement['nmse'])})")
    if measurement.get("flagged"):
        hits = "; ".join(measurement.get("criteria") or []) or "dynamic-range pathology"
        return [head + f", BUT penalty {_fmt_number(measurement.get('penalty'))} ({hits})"
                f" -> implied score {_fmt_number(measurement.get('score'))}",
                "      -> that fit is a localized device: its low NMSE would be paid for by "
                "the penalty, so do not chase the fit quality alone"]
    lines = [head + ", penalty 0 (clean)"]
    score = measurement.get("score")
    if score is not None and best_score is not None:
        if score > best_score:
            lines.append(f"      -> implied score {_fmt_number(score)} WOULD BEAT your current "
                         f"best score ({_fmt_number(best_score)}): an unexplored term set with "
                         "a verifiable improvement on the same training data")
        else:
            lines.append(f"      -> implied score {_fmt_number(score)} is below your current "
                         f"best score ({_fmt_number(best_score)}): still unexplored, but it is "
                         "not the missing step")
    return lines


def _penalty_share(row: dict) -> float:
    """该样本的分数里罚分占的比例（``|score| = mse + penalty``）；算不出时返回 0。"""
    total = (row.get("mse") or 0.0) + (row.get("penalty") or 0.0)
    return (row.get("penalty") or 0.0) / total if total else 0.0


def _render_breakdown(terrain: dict) -> list[str]:
    """分数分解段：把"拟合 MSE"与"体检罚分"拆开。

    实测 20260926-151008 的 order 34 拟合 MSE 0.197 却带 36.06 罚分，模型会把 0.197
    当胜利——观测到的"低 MSE 高罚分 ↔ 高 MSE 零罚分"两模态振荡正是这么来的。
    故这里点名两个"看起来很好但分被吃掉"的样本：带罚分里**分数最高**的那个（干净解
    最强的对手）与**拟合 MSE 最低**的那个（"拟合得好"的真正上界）。

    计数只说"**已落盘**的样本记录"：旧实验只留 top-K（``persist_all_samples`` 之前），
    此时记录数远小于头行的 ``scored samples``，两者不混。
    """
    breakdown = terrain.get("breakdown") or {}
    if breakdown.get("n_scored", 0) <= 0:
        return []
    if not breakdown.get("penalty_known"):
        return ["score decomposition unavailable for this run: its sample records do not "
                "carry the pathology penalty separately from the fit (older format, where "
                "the stored MSE already contained the penalty)"]
    best = breakdown.get("best") or {}
    clean = breakdown.get("best_clean")
    penalized = breakdown.get("best_penalized")
    lowest = breakdown.get("best_fit")
    shown = {best.get("sample_order")} | {other.get("sample_order")
                                         for other in (clean, penalized) if other}
    lines = ["how the score is built - every number below was measured by this run's own "
             "evaluator: score = -(fit MSE + pathology penalty), so a high fit quality "
             "with a penalty still loses"]
    lines.append(f"  best score overall: sample_order {best.get('sample_order')}, fit MSE "
                 f"{_fmt_number(best.get('mse'))}, penalty {_fmt_number(best.get('penalty'))}, "
                 f"score {_fmt_number(best.get('score'))}")
    if clean and clean.get("sample_order") != best.get("sample_order"):
        lines.append(f"  best sample with ZERO penalty: sample_order {clean.get('sample_order')}, "
                     f"fit MSE {_fmt_number(clean.get('mse'))}, "
                     f"score {_fmt_number(clean.get('score'))}")
    if penalized and penalized.get("sample_order") != best.get("sample_order"):
        tail = (": this is the strongest penalized competitor"
                + (", and its score is worse than the clean one" if clean else ""))
        lines.append(f"  best sample that still carries a penalty: sample_order "
                     f"{penalized.get('sample_order')}, fit MSE "
                     f"{_fmt_number(penalized.get('mse'))}, penalty "
                     f"{_fmt_number(penalized.get('penalty'))}, score "
                     f"{_fmt_number(penalized.get('score'))}{tail}")
    if lowest and lowest.get("sample_order") not in shown:
        lines.append(f"  lowest fit MSE seen: sample_order {lowest.get('sample_order')}, fit MSE "
                     f"{_fmt_number(lowest.get('mse'))} but penalty "
                     f"{_fmt_number(lowest.get('penalty'))} -> score "
                     f"{_fmt_number(lowest.get('score'))} "
                     f"({100.0 * _penalty_share(lowest):.0f}% of the score is penalty), so the "
                     "fit MSE alone overstates how good that solution is")
    summary = (f"  Of the {_fmt_number(breakdown.get('n_penalty_known'))} sample records "
               f"persisted for this run, {_fmt_number(breakdown.get('n_penalized'))} carry a "
               "nonzero penalty")
    if not clean:
        summary += " and NONE is clean"
    summary += (". The penalty is charged by the dynamic-range check on verifiable quantities "
                "(output span vs the data range, local slope inside the box, coefficient "
                "cancellation); a low fit MSE WITH a large penalty is a WORSE solution than a "
                "higher fit MSE with zero penalty.")
    lines.append(summary)
    return lines


def render_terrain(terrain: dict, features: Sequence[str], title: str) -> str:
    """把地形渲染成注入提示词的文本块；证据不足或无法判定时返回空串。

    两个闸门（任一不满足就不注入）：已解析样本数 ≥ :data:`MIN_PARSED_SAMPLES`、
    自最优以来 ≥ :data:`MIN_STAGNANT_SAMPLES` 个样本（"自最优"由
    :func:`significant_best` 定，refit 抖动不算进步）。"该架构已触底"只在有足够
    样本时才说得出口。

    四段内容（各段缺数据时自行省略）：结构事实（已试家族 / 这个 term set 被评估过几次）、
    未试邻域的**实测分数**（**两个方向**：删项与加项），以及分数的分解（拟合 MSE ↔ 体检罚分）。
    """
    if not terrain or not terrain.get("ok"):
        return ""
    if terrain.get("n_parsed", 0) < MIN_PARSED_SAMPLES:
        return ""
    if terrain.get("stagnant_samples", 0) < MIN_STAGNANT_SAMPLES:
        return ""
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

    lines.extend(_render_breakdown(terrain))

    deletions = terrain["untried_deletions"]
    additions = terrain.get("untried_additions") or []
    if deletions or additions:
        lines.append("term sets NEVER evaluated that are ONE term away from it:")
        for item in deletions:
            lines.append(f"  drop {item['dropped']} -> {{{', '.join(item['terms'])}}}")
            lines.extend(_render_measurement(item.get("measurement"),
                                             terrain.get("best_score")))
        for item in additions:
            lines.append(f"  add {item['added']} -> {{{', '.join(item['terms'])}}}")
            lines.extend(_render_measurement(item.get("measurement"),
                                             terrain.get("best_score")))
        lines.append(
            "Another re-parameterization of the term set above stays at the same score "
            "floor; the deletions and additions listed above are structurally different and "
            "unexplored.")
    else:
        lines.append(
            "Every one-term deletion AND every one-term addition of that term set has "
            "already been evaluated: the next structural step must change TWO terms at "
            "once, or change the functional form of an existing one.")
    return title + "\n".join(lines) + "\n"