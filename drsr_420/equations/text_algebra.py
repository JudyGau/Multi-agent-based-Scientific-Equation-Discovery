"""样本方程的**文本代数**：把 LLM 写的公式文本规范化、抽取特征、编译成可调用对象。

角色归属
--------
``evaluation`` 层"架构地形"子系统的**词法/语法内核**。它只跟**字符串与 AST** 打交道
——不读盘、不拟合、不渲染——因此可以脱离实验目录单独测试。

为什么单独成模块
----------------
原先这些函数与"采样策略常量 / 未试邻域拟合 / 地形渲染"同住一个 1464 行的文件里。
按职责看，这里做的是**文本代数**：把别名链展开成原子项（``记号化``）、把加性项按幂次
对分类（``项分类``）、再给出对记号不变的**架构指纹**与代表元编译。这些与"该看哪个
未试邻域""汇总块长什么样"是两件事。

在外面的消费方只用**公开名**（:func:`architecture_fingerprint` / :func:`term_texts` /
:func:`representative_from_text` / :func:`features_from_equation` / :func:`atom_meanings`），
所以这次搬迁没有把私有名变成跨模块接口。

设计要点（搬运自原模块文档，仍逐条成立）
--------------------------------------
* 归一化的对象是 LLM 写出的中间变量链：``pow = lambda12**2``、``f12 = 1/(l1*l2)`` 这类
  别名先展开成"自变量 + 常数 + 复合token"的原子式，否则"换个中间变量名"会被误读成
  "换了架构"。
* 每个加性项按幂次对归到标签：``(1,0)``→线性、``(2,0)``→平方、``(1,1)``→交叉、
  其余高阶→``higher``；指数是**参数**（``x**params[1]``）或 ``np.power(x, p)`` 的项
  另记 ``power(...)``（那是幂律，不是多项式）。
* 于是"平移/缩放/取对数后的二次型"与"原坐标二次型"落进**同一指纹**——这正是要抓的
  东西（换个记号不是换架构）；而"只对乘积 λ12·λ23 做二次"落进另一个指纹
  （那是一维山脊模型，不是双变量响应面）。
* 只支持**两个自变量**（本项目当前的多变量实验都落在二元）：更多自变量时函数直接
  返回 ``None``/空串（调用方不注入）。
"""
from __future__ import annotations

import ast
import itertools
import operator
import re
from typing import Sequence

import numpy as np

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


def _alias_texts(code: str, features: Sequence[str]) -> dict[str, str]:
    """别名 → **展开后的文本**（只用自变量名与 ``params[k]`` 表示）；迭代到不动点。

    与 :func:`_alias_map`（别名 → 原子）相对：判标签只需要原子，但"从实际项文本还原
    代表元"需要**能求值的原式**（见 :func:`term_texts`）。展开不完整的（循环引用等）
    会留下别名名，下游 :func:`representative_from_text` 按越界降级，不会误执行。
    """
    raw = _assignments(code)
    names = list(features)
    texts: dict[str, str] = {}
    for _ in range(6):
        changed = False
        for name, rhs in raw.items():
            if name in _RESERVED or name in names:
                continue
            others = {n: f"({t})" for n, t in texts.items() if n != name}
            candidate = _apply(rhs, others)
            if texts.get(name) != candidate:
                texts[name] = candidate
                changed = True
        if not changed:
            break
    return texts


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


def _atom_terms(equation_text: str, features: Sequence[str]):
    """把方程拆成加性项，返回 ``([标签], [原子文本], [原样文本])``；解析不出返回 ``None``。

    "原子文本"是 :func:`_term_label` 判定标签时用的那段（自变量成 ``U0``/``U1``、复合别名
    成 ``W``、参数成 ``P``）；"原样文本"只做别名展开、**参数保留 ``params[k]``**，用于
    :func:`term_texts`（显示与求值都要保留原下标：同一个 ``params[1]`` 出现两次仍是同一个
    参数，压成 ``P`` 之后就分不出来了）。

    两条文本**共用同一次遍历与同一套切分**（切分结果的项数不一致时整体放弃，而不是
    返回错位的项）；指纹与代表元还原因此不会各切一次、最终对不上。
    """
    names = list(features)
    if not equation_text or len(names) != 2:
        return None
    code = _COMMENT_RE.sub(" ", _DOCSTRING_RE.sub(" ", str(equation_text)))
    match = _RETURN_RE.search(code)
    if not match:
        return None
    tail = _DEF_RE.split(code[match.end():], maxsplit=1)[0]
    mapping = {f: t for f, t in zip(names, _FEATURE_TOKENS)}
    mapping.update(_substitution(_alias_map(code, names)))
    expanded = _apply(tail, mapping)
    collapsed = _PARAM_INDEX_RE.sub(_CONST_TOKEN, expanded)
    collapsed = _PARAM_NAME_RE.sub(_CONST_TOKEN, collapsed)
    atom_parts = [_term for _term in _split_terms(collapsed) if _term_label(_term, names)]
    raw_parts = _split_terms(expanded)
    if not atom_parts or len(atom_parts) != len(raw_parts):
        return None
    return [_term_label(_term, names) for _term in atom_parts], atom_parts, raw_parts


def architecture_fingerprint(equation_text: str, features: Sequence[str]) -> tuple[str, ...] | None:
    """结构指纹：加性项标签的排序去重元组；解析不出返回 ``None``。

    只支持两个自变量（见模块 docstring 的"范围"）。
    """
    prepared = _atom_terms(equation_text, features)
    if prepared is None:
        return None
    return tuple(sorted(set(prepared[0])))


def atom_meanings(equation_text: str, features: Sequence[str]) -> dict[str, str]:
    """``U0`` / ``U1`` / ``W`` 在这份方程里各自代表什么（可求值文本）。

    没有别名指向某个原子时它就是自变量本身（``W`` 是两者的乘积）。多个别名指向同一
    原子时取**源码里最先出现的那个**：那是这个方程的写法，代表元只需与它一致——而且
    渲染时会把代表元一并写出来（``representative parameterization``），不是偷偷替换。
    """
    names = list(features)
    meanings = {_FEATURE_TOKENS[0]: names[0] if names else _FEATURE_TOKENS[0],
                _FEATURE_TOKENS[1]: names[1] if len(names) > 1 else _FEATURE_TOKENS[1],
                _COMPOSITE_TOKEN: (f"({names[0]}*{names[1]})" if len(names) == 2
                                   else _COMPOSITE_TOKEN)}
    if len(names) != 2:
        return meanings
    code = _COMMENT_RE.sub(" ", _DOCSTRING_RE.sub(" ", str(equation_text or "")))
    texts = _alias_texts(code, names)
    taken: set[str] = set()
    for alias, atom in _alias_map(code, names).items():
        if atom in meanings and atom not in taken and alias in texts:
            meanings[atom] = f"({texts[alias]})"
            taken.add(atom)
    return meanings


def term_texts(equation_text: str, features: Sequence[str]) -> dict[str, str]:
    """目标方程里每个标签对应的**实际项文本**（还原成自变量与 ``params`` 的可求值形式）。

    这是缺陷 2 的修法。``higher`` / ``power(λ12,λ23)`` 这类标签把多个形状归并在一起，
    **从标签猜代表元是不诚实的**；但它们的项在目标方程里写得明明白白。于是这里把该项
    自己的文本取出来（``U0`` → 它的别名式、``W`` → 复合式、``P`` → ``params[k]``），
    交给 :func:`representative_from_text` 求值——**不是猜，是读**。

    同一标签出现多次时取第一次出现的那个项——与 :func:`architecture_fingerprint` 的
    去重口径一致（标签相同即视为同一架构）。
    """
    names = list(features or [])
    prepared = _atom_terms(equation_text, names)
    if prepared is None:
        return {}
    labels, _atoms, raw = prepared
    meanings = atom_meanings(equation_text, names)
    out: dict[str, str] = {}
    for label, term in zip(labels, raw):
        if label in out:
            continue
        expanded = term
        for token in (_COMPOSITE_TOKEN, _FEATURE_TOKENS[1], _FEATURE_TOKENS[0]):
            expanded = re.sub(_bound(token), lambda _m, t=token: meanings[t], expanded)
        out[label] = expanded.strip()
    return out


#: 代表元求值允许的函数（只认 ``np.<name>`` 形式；越界一律降级）。
_ALLOWED_FUNCS = {
    "log": np.log, "log10": np.log10, "exp": np.exp, "sqrt": np.sqrt,
    "abs": np.abs, "sign": np.sign, "tanh": np.tanh, "power": np.power,
}

_ALLOWED_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                   ast.Div: operator.truediv, ast.Pow: operator.pow}


def _bind_params(text: str, *, offset: int = 0) -> tuple[str, int]:
    """把文本里的参数改写成 ``p{offset}/p{offset+1}/...``，返回 ``(改写后的文本, 个数)``。

    下标记号从 ``offset`` 起：多个项拼进同一个参数向量时（:func:`_plan_terms` 的
    ``verbatim`` 片段）编号必须全局连续，否则求值时会读到别人的参数。

    两种来源分开处理：

    * ``params[k]``（:func:`term_texts` 给出的原样文本）：**按原文去重**——同一个
      ``params[1]`` 出现两次仍是同一个参数，下标信息在这里是可信的；
    * ``P``（原子化留下的记号，多个不同下标已被压成同一个）：只能**按出现次序**各给
      一个。对常见写法（``P*U0``、``P*U0**P``、``P*W**2``）这次序正好是"系数 + 形状
      参数"，个数是对的；只有"同一项里把同一个 ``params[k]`` 写两次"会多一个自由度
      （罕见，且不影响"哪个方向更值得试"这一判断）。
    """
    seen: dict[str, str] = {}
    counter = itertools.count(offset)

    def repl(match):
        key = match.group(0)
        if key.startswith("params"):
            if key not in seen:
                seen[key] = f"p{next(counter)}"
            return seen[key]
        return f"p{next(counter)}"

    bound = _PARAM_INDEX_RE.sub(repl, str(text))
    bound = re.sub(_bound(_CONST_TOKEN), repl, bound)
    return bound.strip(), next(counter) - offset


def _compile_node(node, names: Sequence[str]):
    """AST 节点 → 求值闭包 ``f(params, columns)``；越界返回 ``None``。

    刻意**不用** :func:`eval` / ``exec``：文本来自 LLM 写的方程，虽然只取**一项**、且已
    经别名展开，仍按白名单逐节点分派（数字 / 自变量 / ``p{k}`` / ``+ - * / **`` / 有限几个
    ``np.*`` 函数）。下标、属性、其它调用、推导式、``lambda``、条件表达式等一律返回
    ``None``，由调用方降级成"未测量"——绝不执行。
    """
    names = list(names or [])
    if isinstance(node, ast.Expression):
        return _compile_node(node.body, names)
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return lambda params, columns, v=float(value): v
    if isinstance(node, ast.Name):
        if node.id in names:
            index = names.index(node.id)
            return lambda params, columns, i=index: columns[i]
        if len(node.id) > 1 and node.id[0] == "p" and node.id[1:].isdigit():
            index = int(node.id[1:])
            return lambda params, columns, i=index: params[i]
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _compile_node(node.operand, names)
        if operand is None:
            return None
        sign = 1.0 if isinstance(node.op, ast.UAdd) else -1.0
        return lambda params, columns, f=operand, s=sign: s * f(params, columns)
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_BINOPS.get(type(node.op))
        left = _compile_node(node.left, names)
        right = _compile_node(node.right, names)
        if op is None or left is None or right is None:
            return None
        return lambda params, columns, l=left, r=right, o=op: o(l(params, columns),
                                                                r(params, columns))
    if isinstance(node, ast.Call):
        func = node.func
        if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
            return None
        if func.value.id not in ("np", "numpy") or node.keywords:
            return None
        allowed = _ALLOWED_FUNCS.get(func.attr)
        if allowed is None or len(node.args) != 1:
            return None
        arg = _compile_node(node.args[0], names)
        if arg is None:
            return None
        return lambda params, columns, f=arg, g=allowed: g(f(params, columns))
    return None


def _compile_text(bound_text: str, names: Sequence[str]):
    """已绑好参数名的表达式 → 求值闭包；语法错或越界返回 ``None``。"""
    try:
        tree = ast.parse(bound_text, mode="eval")
    except SyntaxError:
        return None
    return _compile_node(tree, names)


def representative_from_text(text: str, names: Sequence[str], *, offset: int = 0):
    """把一段项文本编译成 ``f(*columns, params)``；越界即降级（绝不执行）。

    Returns:
        ``(fn, bound_text, n_params, reason)``。成功时 ``reason`` 为 ``None``；
        ``fn`` 吃**完整**参数向量，下标从 ``offset`` 起（便于把多个项拼进同一套编号）；
        ``bound_text`` 是编号改写后的文本，可直接当"代表元参数化"写进提示。
    """
    bound, n_params = _bind_params(text, offset=offset)
    if _compile_text(bound, names) is None:
        try:
            ast.parse(bound, mode="eval")
        except SyntaxError as exc:
            reason = f"the term text does not parse ({exc.msg})"
        else:
            reason = ("the term uses constructs outside the whitelist (numbers, the two "
                      "independents, params, + - * / ** and a few np.* functions)")
        return None, bound, n_params, reason
    evaluator = _compile_text(bound, names)

    def equation(*args):
        columns, params = args[:-1], np.asarray(args[-1], dtype=float)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return evaluator(params, columns)

    return equation, bound, n_params, None


# ── 汇总 ────────────────────────────────────────────────────
