"""表达式解析：把 LLM 生成的函数骨架字符串变成可求值的 SymPy 表达式。

角色归属
--------
收尾分析（analysis）阶段的**第一步**：参数代入 + 语法归一化 + 表达式解析。
无状态、不调用 LLM、不写磁盘（末尾的 :func:`audit_parse_failures` 是收尾自检，
只**读** ``samples/*.json`` 统计解析失败率，不落盘）。

为什么要单独一个模块
--------------------
LLM 写出来的"类 Python"骨架与 SymPy 的语义有三处系统性偏差，且**都会静默算错**：

1. ``where(cond, a, b)``（numpy 风格）不是合法 SymPy 调用，必须改写成 ``Piecewise``；
   ``where`` 参数个数不对时更不能"改成 Piecewise 交差"——``Piecewise(x1 > 0)`` 会被
   SymPy 解释成空 Piecewise 并返回 ``nan``，静默毒化整条表达式；
2. ``x1 == 0`` 在 SymPy 里会退化成 Python 的 ``False``（``Symbol.__eq__`` 对非 Basic
   返回 False），于是 ``where(x1 == 0, a, b)`` 会**静默**塌缩成 ``b``；必须包成 ``Eq``；
3. ``a = expr`` 形式的中间变量行要显式消解，否则 ``return a`` 会静默返回裸符号 ``a``，
   被下游当成"发现"的方程；
4. ``parse_expr`` 默认把**整个 sympy 命名空间**当全局名，中间变量一旦与 sympy 撞名
   （``poly`` 抛异常、``E`` / ``pi`` / ``gamma`` 静默换成常量）必须先注册进
   ``local_dict``——见 :func:`_parse_expr_with_symbols`。

这四处坑各自都有回归测试（``tests/test_expr_substitution.py``），逻辑集中在这里
便于逐条加固；``find_best_eq.py`` 只负责"调它、拿结果、做下一步"。

对外契约：``expr_substitution()`` 解析失败一律返回 ``None``（不抛异常）；
``rewrite_where_calls()`` 在 ``where`` 参数个数不是 3 时抛 :class:`WhereArityError`。
"""
from __future__ import annotations

import ast
import contextlib
import glob
import io
import json
import math
import os
import re

import sympy as sp

#: 参数代入表达式时保留的**有效数字**位数（不是小数点后位数）。
#:
#: 旧实现是 ``round(x, 2)``（小数点后 2 位），对"小系数 × 巨量项"的骨架是灾难：
#: 实测 experiments/MRFCompress-Cuboid/MRFCompress-Cuboid_20260918-195057 的最优样本，
#: params[3]=0.0074
#: 被舍成 0.01，而它乘的 (lambda23**3.59 - 1) 量级到 1e4 —— MSE 从 2.9e-4 飙到 186
#: （六个数数量级）。于是收尾产物解释的根本不是实验选出的那个模型：剪枝在错的
#: 表达式上做、曲线偏离数据、report.md 拿 186 去论证"剪枝合理"。
#: 换成 6 位有效数字后，同一组参数 MSE = 2.917e-4（+0.4%），既保住拟合，写进
#: 表达式里也不比 "0.01" 长多少（0.0074、193.054、8.3557）。
PARAM_SIG_DIGITS = 6


def _round_params(params, sig: int = PARAM_SIG_DIGITS) -> list:
    """按有效数字舍入参数；非数值/NaN/inf 原样保留（由后续解析决定成败）。"""
    out = []
    for x in params:
        try:
            value = float(x)
        except (TypeError, ValueError):
            out.append(x)
            continue
        if value == 0.0 or not math.isfinite(value):
            out.append(value)
        else:
            out.append(float(f"%.{sig}g" % value))
    return out


def find_matching_paren(text: str, open_idx: int) -> int:
    """返回 text[open_idx] 处 '(' 对应的 ')' 下标；括号不闭合时返回 -1。

    字符串字面量内部的括号会被跳过，反斜杠转义按两字符整体跳过。
    """
    depth = 0
    quote = None
    i = open_idx
    while i < len(text):
        ch = text[i]
        if quote is not None:
            if ch == '\\':
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def split_top_level(text: str, sep: str = ',') -> list:
    """按"括号深度为 0"的分隔符切分，括号/中括号/花括号及字符串内的分隔符不切分。

    这是修掉 `where(cond, maximum(a, b), c)` 这类"参数里自带逗号"问题的关键：
    旧的 `str.split(',')` 会把它切成 4 段，再按 `parameter_list[0..2]` 取值，
    cond / 两个分支全部错位，真值与假值被丢弃。
    """
    parts, depth, start, quote = [], 0, 0, None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote is not None:
            if ch == '\\':
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
        i += 1
    parts.append(text[start:])
    return parts


# 只匹配独立的 where 标识符调用；`(?<![\w.])` 防止命中 somewhere(...) / np.where(...)。
_WHERE_CALL_RE = re.compile(r'(?<![\w.])where\s*\(')

# 锚定的中间变量赋值行：`a = expr`。`(?!=)` 排除 `==`，前面的标识符要求保证
# 含比较运算符的行（`return where(x1 >= 0, ...)`）不会误判为赋值。
_ASSIGN_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*=(?!=)\s*(.+?)\s*$')


def _known_names(independent_list: list, inter_vars: dict) -> list:
    """自变量 + 已解析中间变量的名字，用作 :func:`_parse_expr_with_symbols` 的符号表。"""
    return [*independent_list, *(str(sym) for sym in inter_vars)]


def fold_constant_comparisons(text: str) -> str:
    """把"两侧都是数值常量"的比较折叠成 0/1（``(-0.219 == 0) * 1e-12`` → ``0 * 1e-12``）。

    模型为避免除零常写 ``(params[k] == 0) * 1e-12`` 这类保护；参数代入后比较两侧都成了
    数值，``parse_expr`` 会把 ``==`` 求值成 Python 的 ``bool``，于是 ``bool * Float``
    抛 ``TypeError``——整条中间变量行解析失败（实测：最优样本的三个 sat 项因此变成
    孤儿符号，剪枝又把它们全删掉，公式塌缩成常数）。两侧都是常量时结果本就确定，
    折叠成 ``int(bool)`` 语义不变。

    只折叠**常量**比较：``x1 == 0``（含符号）原样保留——它属于
    :func:`normalize_condition` 的职责，在这里折叠会静默丢掉分支。文本不是合法的
    Python 表达式时原样返回（交由下游按原样解析/报错）。
    """
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return text

    def _numeric(node) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float, bool))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return _numeric(node.operand)
        return False

    class _Fold(ast.NodeTransformer):
        def visit_Compare(self, node):
            self.generic_visit(node)
            if len(node.ops) != 1 or not all(
                    _numeric(o) for o in (node.left, *node.comparators)):
                return node
            try:
                # 只含数值常量，无名字/调用，eval 无副作用
                value = eval(compile(ast.Expression(node), "<fold>", "eval"),
                             {"__builtins__": {}}, {})
            except Exception:
                return node         # 例如常量除零：原样留给下游报错
            return ast.copy_location(ast.Constant(int(bool(value))), node)

    try:
        return ast.unparse(_Fold().visit(tree))
    except Exception:
        return text


def _parse_expr_with_symbols(text: str, names: list) -> sp.Expr:
    """用显式符号表解析表达式，避免与 SymPy 全局名撞名。

    ``sp.parse_expr`` 只在 ``local_dict`` 里找不到名字时才去查全局名，而它的
    ``global_dict`` 默认是**整个 sympy 命名空间**。模型把中间变量取成 sympy 已有的
    名字时就会出事，且两种后果都难排查：

    * ``poly`` 在 sympy 里是函数，``poly * f23`` 直接抛
      ``TypeError: unsupported operand type(s) for *: 'function' and 'Symbol'``，
      于是``expr_substitution`` 返回 None，物理解释 / 剪枝 / 预览图全部跳过；
    * ``E`` / ``pi`` / ``gamma`` 这类**不报错**：中间变量被静默替换成常量或函数，
      产出一个看着正常、其实错误的表达式，比抛异常更危险。

    因此把自变量与已知中间变量的名字都放进 ``local_dict``，令其优先于全局名；
    ``N``（样本数）按历史行为始终保留为符号。
    """
    local = {name: sp.Symbol(name) for name in names}
    local.setdefault('N', sp.Symbol('N'))
    return sp.parse_expr(fold_constant_comparisons(text), local)


def normalize_condition(cond: str) -> str:
    """把条件里的顶层 `==` / `!=` 换成 SymPy 的 `Eq` / `Ne`。

    SymPy 只把 `<` `<=` `>` `>=` 重载成关系式；`x1 == 0` 会退化成 Python 的
    `False`（`Symbol.__eq__` 对非 Basic 返回 False），于是 Piecewise 的第一个
    分支被静默丢弃、整条表达式塌缩成 else 分支——即
    `where(x1 == 0, a, b)` 会**静默**变成 `b`。这里在顶层把 == / != 包成
    Eq / Ne，使其成为真正可判定的关系式。

    仅处理顶层（括号深度为 0）的运算符，括号/字符串内的原样保留。
    `and` / `or` 不在此转换：`&` / `|` 的优先级与比较运算符不同，直接替换会
    改变结合顺序，宁可让 SymPy 解析失败（返回 None）也不要静默算错。
    """
    depth = 0
    quote = None
    i = 0
    while i < len(cond):
        ch = cond[i]
        if quote is not None:
            if ch == '\\':
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        elif depth == 0:
            if cond.startswith('==', i):
                left = normalize_condition(cond[:i]).strip()
                right = normalize_condition(cond[i + 2:]).strip()
                return f"Eq({left}, {right})"
            if cond.startswith('!=', i):
                left = normalize_condition(cond[:i]).strip()
                right = normalize_condition(cond[i + 2:]).strip()
                return f"Ne({left}, {right})"
        i += 1
    return cond


class WhereArityError(ValueError):
    """`where(...)` 参数个数不是 3（numpy 只有 where(cond, x, y) 这一种公式用法）。"""


def rewrite_where_calls(text: str) -> str:
    """把 numpy 风格 `where(cond, a, b)` 改写为 SymPy 的 `Piecewise((a, cond), (b, True))`。

    旧实现把 `Piecewise\\(.*?\\)\\n` 整体匹配出来后按 `split(',')` 取前三段，实测三处缺陷
    （均已写成测试固化）：
      1. 逗号不足 2 个时 `parameter_list[2]` 抛 IndexError，且该异常无人接管，
         会一路上抛打断最终结果整理流程；
      2. 参数自带逗号（`where(c, maximum(a, b), d)`）或嵌套 `where` 时切分错位，
         条件分支被静默丢弃、留下括号不配对的残句，只能退化为整条解析失败；
      3. 正则要求闭括号后紧跟换行，故 `where(...) + x`、函数串末尾无换行等写法漏改，
         残留的 `Piecewise(cond, a, b)` 被 SymPy 判为非法而失败。
    改为按括号配对定位整个调用、按顶层逗号切分参数，并递归处理嵌套调用。

    参数个数不是 3 时抛 `WhereArityError`（由调用方转成返回 None）；条件里的
    `==` / `!=` 一并转成 `Eq` / `Ne`，见 :func:`normalize_condition`。
    """
    out, pos = [], 0
    while True:
        m = _WHERE_CALL_RE.search(text, pos)
        if m is None:
            out.append(text[pos:])
            return ''.join(out)

        open_idx = m.end() - 1              # 正则末尾 '(' 的下标
        close_idx = find_matching_paren(text, open_idx)
        out.append(text[pos:m.start()])
        if close_idx < 0:                   # 括号不闭合：原样保留，交由下游解析报错
            out.append(text[m.start():])
            return ''.join(out)

        inner = text[open_idx + 1:close_idx]
        args = [a.strip() for a in split_top_level(inner)]
        while args and args[-1] == '':      # 容忍 `where(c, a, b,)` 这类尾随逗号
            args.pop()
        if len(args) != 3:
            # 不能只把名字改成 Piecewise 交差：`Piecewise(x1 > 0)` 会被 SymPy
            # 解释成空 Piecewise 并返回 nan（静默毒化表达式），
            # `Piecewise(c, a)` 才会报错。两种结果都不可接受，直接显式失败。
            raise WhereArityError(
                f"where(...) 需要 3 个参数 (cond, true_val, false_val)，实际 {len(args)} 个：{inner.strip()!r}"
            )
        cond, true_val, false_val = [rewrite_where_calls(a) for a in args]
        out.append(f"Piecewise(({true_val}, {normalize_condition(cond)}), ({false_val}, True))")
        pos = close_idx + 1


def _unwrap_outer_parens(expr_str: str) -> str:
    """剥掉多行括号式 return 的最外层括号并合并为单行：

    ``return (\\n  expr1\\n  + expr2\\n)`` → ``expr1 + expr2``。
    """
    if not expr_str.startswith('('):
        return ' '.join(expr_str.splitlines())
    depth = 0
    end_idx = len(expr_str)
    for i, ch in enumerate(expr_str):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    return ' '.join(expr_str[1:end_idx].splitlines())


_TUPLE_UNPACK_RE = re.compile(
    r"^\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*=\s*"
    r"((?:params(?:\[[^\]]*\])?|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s*,\s*(?:params(?:\[[^\]]*\])?|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))*)"
    r"\s*$")
#: 逐项解包的单项：params[整数] 或已数值化的字面量（标量替换先于本预处理执行）。
_TUPLE_ITEM_INDEX_RE = re.compile(r"^params\[(\d+)\]$")
_TUPLE_ITEM_NUMBER_RE = re.compile(
    r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
#: 通用多目标赋值（LHS 至少两个名字）：用于兜底识别 RHS 带包裹括号的逐项解包。
_TUPLE_ASSIGN_RE = re.compile(
    r"^\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)+)\s*=\s*(.+)$")
#: 生成器/列表推导式解包 ``p0, p1, ... = (params[i] for i in range(n))``（可带外壳括号）。
#: 与切片解包 ``= params[:n]`` 语义等价，但 RHS 既非单一 ``params[...]``（_TUPLE_UNPACK_RE
#: 不认括号），也非逐项逗号列表（_resolve_tuple_assignment 只得到 1 项）——两套既有识别
#: 全部落空，p0..pn 沦为自由符号（实测 ab-fix6-treatment/..._20260928-154609 samples_42）。
_COMPREHENSION_UNPACK_RE = re.compile(
    r"^params\s*\[\s*([A-Za-z_]\w*)\s*\]\s*for\s+\1\s+in\s+"
    r"range\s*\(\s*(?:(\d+)|len\s*\(\s*params\s*\))\s*\)$")
_PAREN_OPEN = re.compile(r"[(\[{]")
_PAREN_CLOSE = re.compile(r"[)\]}]")


def _paren_delta(line: str) -> int:
    """一行内未闭合的括号数（正 = 还有括号没关上）。"""
    return len(_PAREN_OPEN.findall(line)) - len(_PAREN_CLOSE.findall(line))


def _resolve_tuple_items(names: list[str], items: list[str],
                         params: list) -> dict[str, str] | None:
    """逐项解包映射：params[整数] 按下标取值、数值字面量原样代入。

    元组解包按 Python 语义要求两侧数目一致；出现识别不了的项或数目不齐时
    返回 None（安全失败，孤儿符号由函数尾部的自由符号护栏拒绝），不带病求值。
    """
    if len(names) != len(items):
        return None
    mapping: dict[str, str] = {}
    for name, item in zip(names, items):
        im = _TUPLE_ITEM_INDEX_RE.match(item)
        if im:
            k = int(im.group(1))
            if k >= len(params):
                return None
            mapping[name] = str(params[k])
        elif _TUPLE_ITEM_NUMBER_RE.match(item):
            mapping[name] = item
        else:
            return None
    return mapping


def _resolve_tuple_assignment(lhs: str, rhs: str, params: list) -> dict[str, str] | None:
    """兜底解析 RHS 带包裹括号的逐项解包 ``p0, p1 = (params[0], params[1])``。

    与 :data:`_TUPLE_UNPACK_RE` 的差异：RHS 允许整体包一层（或多层）括号——
    跨行括号元组（实测 MRFCompress-Cuboid_20260921-161549 order75 最优样本）
    合并成单行后旧正则依然不认（RHS 以 ``(`` 开头）。LHS 至少两个名字，
    普通单变量赋值（``sigma = (...)``）不会进入本函数。
    """
    rhs = rhs.strip()
    while rhs.startswith("("):
        close = find_matching_paren(rhs, 0)
        if close != len(rhs) - 1:
            break
        rhs = rhs[1:close].strip()
    names = [n.strip() for n in lhs.split(",")]
    items = [s.strip() for s in split_top_level(rhs, ",")]
    return _resolve_tuple_items(names, items, params)


def _resolve_comprehension_unpack(lhs: str, rhs: str, params: list) -> dict[str, str] | None:
    """``p0, p1, ... = (params[i] for i in range(n))``：按位置代参。

    生成器/列表推导式是 ``params[:n]`` 的等价写法，但 RHS 不是单一 ``params[...]``、
    也不是逐项逗号列表，既有两套识别都不命中 → p0..pn 全部沦为自由符号 → ``return``
    得不到替换目标 → 整条样本返回 None（实测 ab-fix6-treatment/..._20260928-154609
    samples_42 的 best 样本）。剥掉可选的外壳括号/中括号后，按 comprehension 的索引
    变量与 ``range`` 上界映射为 ``params[0..n-1]``。

    上界支持字面量 ``range(n)`` 与 ``range(len(params))``。名字数多于上界（Python 运行
    时会 ValueError 的不齐形态）时返回 None——安全失败，不猜。
    """
    rhs = rhs.strip()
    while rhs and rhs[0] in "([":
        closer = ")" if rhs[0] == "(" else "]"
        if not rhs.endswith(closer):
            return None
        rhs = rhs[1:-1].strip()
    m = _COMPREHENSION_UNPACK_RE.match(rhs)
    if m is None:
        return None
    names = [n.strip() for n in lhs.split(",")]
    upper = int(m.group(2)) if m.group(2) is not None else len(params)
    if len(names) > upper:
        return None
    mapping: dict[str, str] = {}
    for j, name in enumerate(names):
        if j >= len(params):
            return None
        mapping[name] = str(params[j])
    return mapping


#: 纯类型转换包装：``np.asarray(x, dtype=float)`` / ``np.array(x)`` / ``x.astype(float)``。
#: 符号层只关心**值**，dtype 不改变值 → 去掉包装是保语义的（只删这三类，不碰
#: ``where``/``maximum``/``power`` 这些会改变值的调用）。
_TYPE_CAST_CALL_RE = re.compile(
    r'(?<![\w.])(?:np\.|numpy\.)?(?:asarray|array)\s*\(\s*([^(),]+?)\s*(?:,\s*[^()]*)?\)')
_ASTYPE_RE = re.compile(r'([A-Za-z_]\w*|\([^()]+\))\s*\.astype\s*\(\s*[^()]*\)')


def strip_type_cast_wrappers(text: str) -> str:
    """去掉**纯类型转换**包装：``asarray(x, dtype=…)`` → ``x``、``x.astype(…)`` → ``x``。

    为什么必须做：sympy 1.14 的 ``parse_expr`` **不认 numpy 的关键字参数**，会抛
    ``ValueError: Unknown options: {'dtype': float}``。最小复现（2026-09-29 实测）::

        asarray(lambda12, dtype=float)      -> ValueError: Unknown options: {'dtype': float}
        asarray(lambda12)                   -> 通过
        asarray(lambda12, dtype=np.float64) -> AttributeError: 'Symbol' object has no attribute 'float64'

    于是一行中间变量解析失败被 ``continue`` 跳过 → 它定义的符号成了**孤儿** → ``return``
    无法求值 → 整条样本返回 ``None`` → ``find_best_eq`` 拿不到选解信息 → 报告缺
    「发布解选择」小节。实测 ``ab-iso6-no6/MRFCompress-Cuboid_20260929-091844`` 的 best
    样本正是这个链条（``l12 = np.asarray(lambda12, dtype=float)`` → 三次 WARN → None）。

    实测分布：四臂 **1024** 条样本里 **28** 条含这类包装、**全部带 ``dtype=``**（形态单一）。
    只做**保语义**的删除；括号内是复合表达式（如 ``asarray((a+b))``）时正则不命中、
    原样保留——宁可少删也不误改。连续/嵌套包装迭代到不动点。
    """
    out = str(text or "")
    for _ in range(4):
        new = _TYPE_CAST_CALL_RE.sub(r'\1', out)
        new = _ASTYPE_RE.sub(r'\1', new)
        if new == out:
            break
        out = new
    return out


#: 显式 numpy 幂调用：``np.power(a, b)`` / ``numpy.power(a, b)``。
_NP_POWER_RE = re.compile(r'(?<![\w.])(?:np|numpy)\s*\.\s*power\s*\(')


def rewrite_numpy_power_calls(text: str) -> str:
    """把显式 ``np.power(a, b)`` 改写为 ``((a)**(b))``。

    为什么必须在**剥离 ``np.`` 前缀之前**做：模型可能先用 ``power`` 命名一个中间变量
    （``power = np.power(aspect, p1)``），随后又写 ``np.power(lambda23, p4)``。一旦剥掉
    ``np.``，两处都成了裸 ``power``；而 :func:`_parse_expr_with_symbols` 会把已知中间
    变量名注入 ``local_dict``，于是后一处 ``power(...)`` 命中的是**那个符号**而不是 sympy
    的 ``power`` 函数 → ``'Symbol' object is not callable`` → 该行被跳过 → ``l23_term``
    成孤儿 → 整条样本返回 None（实测 ab-iso6-head/..._20260929-091848 samples_12）。

    numpy 的 ``power`` 就是逐元素幂，改写为 ``(a)**(b)`` 与 sympy 的 ``Pow`` 语义完全
    一致。参数个数不是 2 时（numpy 本就会报错）原样保留，交给下游按解析失败处理。
    """
    out, pos = [], 0
    while True:
        m = _NP_POWER_RE.search(text, pos)
        if m is None:
            out.append(text[pos:])
            return ''.join(out)
        open_idx = m.end() - 1               # 正则末尾 '(' 的下标
        close_idx = find_matching_paren(text, open_idx)
        out.append(text[pos:m.start()])
        if close_idx < 0:                    # 括号不闭合：原样保留，交由下游报错
            out.append(text[m.start():])
            return ''.join(out)
        args = split_top_level(text[open_idx + 1:close_idx])
        if len(args) == 2:
            a = rewrite_numpy_power_calls(args[0].strip())   # 递归处理嵌套 np.power
            b = rewrite_numpy_power_calls(args[1].strip())
            out.append(f"(({a})**({b}))")
        else:
            out.append(text[m.start():close_idx + 1])
        pos = close_idx + 1


#: 显式/裸 numpy 截断调用：``np.clip(a, a_min, a_max)`` / ``clip(...)``
#: （``a_min`` / ``a_max`` 可以是 ``None``，表示该侧不设界）。
_NP_CLIP_RE = re.compile(r'(?<![\w.])(?:(?:np|numpy)\s*\.\s*)?clip\s*\(')


def rewrite_numpy_clip_calls(text: str) -> str:
    """把 ``np.clip(a, a_min, a_max)`` 改写为 sympy 的 ``Max`` / ``Min`` 组合。

    numpy 语义：``clip(a, lo, hi)`` = 把 a 逐元素截到 ``[lo, hi]``；``lo``/``hi`` 为
    ``None`` 表示该侧不设界。sympy 没有 ``clip``：``clip`` 会被当成未知函数，
    而其中的 ``None`` 字面量更会让 ``parse_expr`` 抛
    ``'NoneType' object has no attribute 'is_Float'``——整行中间变量被跳过、符号成
    孤儿、``return`` 无法求值（实测 BPG0 benchmark 的 I.37.4_0_1 samples_8：
    ``radicand = np.clip(radicand, 0.0, None)``）。

    改写为等价的最小组合（与 ``np.clip`` 同义）::

        clip(a, None, None) -> a
        clip(a, lo,   None) -> Max(a, lo)
        clip(a, None, hi)   -> Min(a, hi)
        clip(a, lo,   hi)   -> Max(Min(a, hi), lo)

    参数个数不是 3（numpy 本就会报错）或括号不闭合时原样保留，交下游按失败处理。
    """
    out, pos = [], 0
    while True:
        m = _NP_CLIP_RE.search(text, pos)
        if m is None:
            out.append(text[pos:])
            return ''.join(out)
        open_idx = m.end() - 1               # 正则末尾 '(' 的下标
        close_idx = find_matching_paren(text, open_idx)
        out.append(text[pos:m.start()])
        if close_idx < 0:                    # 括号不闭合：原样保留
            out.append(text[m.start():])
            return ''.join(out)
        args = [a.strip() for a in split_top_level(text[open_idx + 1:close_idx])]
        if len(args) != 3:
            out.append(text[m.start():close_idx + 1])
        else:
            a, lo, hi = [rewrite_numpy_clip_calls(x) for x in args]  # 递归处理嵌套 clip
            lo_none, hi_none = (lo == "None"), (hi == "None")
            if lo_none and hi_none:
                out.append(f"({a})")
            elif lo_none:
                out.append(f"Min(({a}), ({hi}))")
            elif hi_none:
                out.append(f"Max(({a}), ({lo}))")
            else:
                out.append(f"Max(Min(({a}), ({hi})), ({lo}))")
        pos = close_idx + 1


def _normalize_statements(func: str, params: list) -> str:
    """语句级预处理，替 return 的符号消解扫清三种 LLM 常见写法：

    ① **元组解包** ``p0, p1, ..., p9 = params[:10]``：旧流程只会替换
       ``params[i]`` 形式，解包出来的名字全部留在表达式里成为自由符号，
       ``return sigma`` 的 sigma 无处可解，最终"表达式"退化为裸符号——
       剪枝率恒 0、曲线报 Cannot convert expression to float
       （实测 MRFCompress-Cuboid_20260917-194203）。按位置把每个名字
       替换为数值后，后续流程照常工作。
       **逐项形式** ``a, b, c = params[0], params[1], params[2]`` 同理必须
       处理（实测 MRFCompress-Cuboid_20260921-134921 的最优样本用的正是
       这种写法）：标量替换先执行，本函数看到的是 ``a, b, c = 0.048, 3.19,
       288.1``；元组解包按位置对应，逐项映射即可。旧正则只认 RHS 为单一
       ``params[...]``，逐项形式整行漏掉 → a..f 成为自由符号 → 解析返回
       None → **剪枝与 held-out 验证被整体跳过**（explain 退化为无剪枝版）。
    ② **多行赋值** ``sigma = (\\n  p0\\n  + p1 ...)``：赋值正则按单行匹配，
       只能看到 ``sigma = (`` 就 EOF 报错被跳过，同样退化为裸符号。
       按括号配对把续行合并回单行（对多行 ``return (`` 同样生效）。
    ④ **纯类型转换包装** ``l12 = np.asarray(lambda12, dtype=float)``：sympy 不认 numpy 的
       关键字参数，该行抛 ``Unknown options`` 被跳过 → ``l12`` 成孤儿 → ``return`` 无法
       求值 → 整条样本返回 None（实测 ``ab-iso6-no6/..._20260929-091844`` 的 best 样本，
       后果是报告缺「发布解选择」小节）。dtype 不改变值，去掉包装保语义，见
       :func:`strip_type_cast_wrappers`。
    """
    name_map: dict[str, str] = {}
    #: **整体别名**（``p = params`` / ``p = params[:10]``）：左侧只有一个名字、右侧是整个
    #: 数组。它不是解包，值要**按下标**取（见函数尾部的展开），不能当成"第一个参数"。
    array_aliases: set[str] = set()
    out_lines: list[str] = []
    # ③ **反斜杠续行**（Python 语义：行尾 ``\`` 后接换行等于一行）：
    #    ``p0, p1, p2 = params[0], params[1], \\n    params[2]`` 的第一行以
    #    ``\`` 结尾，_TUPLE_UNPACK_RE 的 ``\\s*$`` 永远匹配失败（``\`` 不是空白），
    #    第二行又是裸参数行——实测 MRFCompress-Cuboid_20260921-161549 的最优样本
    #    正是这种写法，p0..p7 全部沦为自由符号，剪枝与 held-out 被整体跳过。
    #    逐项/元组解包处理之前先把续行拼回单行；注释与 docstring 已在上游被
    #    移除，此处残留的行尾 ``\`` 只可能是续行符。
    joined: list[str] = []
    pending = ""
    for raw in func.splitlines():
        if pending:
            pending = pending + " " + raw.strip()
        else:
            pending = raw
        stripped = pending.rstrip()
        if stripped.endswith("\\"):
            pending = stripped[:-1]
            continue
        joined.append(pending)
        pending = ""
    if pending:
        joined.append(pending)
    # ④ 纯类型转换包装先去掉（np. 前缀在上游还没剥，故正则自带 np./numpy. 可选前缀）：
    # 不去掉的话 asarray(x, dtype=float) 会抛 Unknown options，该行被跳过、符号成孤儿。
    lines = [strip_type_cast_wrappers(line) for line in joined]
    i = 0
    while i < len(lines):
        line = lines[i]
        merged = line.strip()
        # 括号续行合并（见②），且每合并一行就重试一次解包匹配——④ 的跨行
        # 括号元组只有合并成单行才可能识别。
        m = _TUPLE_UNPACK_RE.match(line)
        while m is None and _paren_delta(merged) > 0 and i + 1 < len(lines):
            i += 1
            merged = merged + " " + lines[i].strip()
            m = _TUPLE_UNPACK_RE.match(merged)
        if m:
            names = [n.strip() for n in m.group(1).split(",")]
            items = [s.strip() for s in m.group(2).split(",")]
            if len(names) == 1 and (items[0] == "params"
                                    or re.fullmatch(r"params\s*\[[^\]]*:[^\]]*\]", items[0])):
                # ⑤ **整体/切片别名**（``p = params``、``p = params[:10]``）——**不是解包**：
                # 左侧只有一个名字、右侧是**整个数组**。旧逻辑把它按"名字按位置对应 params
                # 列表"处理，于是 p 被绑成 params[0] 的**值**，后续 ``p[3]`` 经名字替换变成
                # ``193.05[3]``（数字被索引）→ sympy 报内部错
                # ``Integer.__new__() missing 1 required positional argument: 'i'``。
                # 实测 ab-fix6-control/..._20260928-154613 的 best 样本正是这种写法
                # （``p = params`` + ``return p[0] + p[1]*a12 + …``）。
                # 注意与 ``p = params[0]``（**标量**，无冒号、走下面的逐项分支）区分。
                array_aliases.add(names[0])
                i += 1
                continue
            if len(items) == 1 and not _TUPLE_ITEM_NUMBER_RE.match(items[0]):
                # 单一 params / params[slice]：名字按位置对应 params 列表
                for j, name in enumerate(names):
                    if j < len(params):
                        name_map[name] = str(params[j])
            else:
                # 逐项形式（``a, b = params[0], params[1]``，或标量替换后
                # ``a, b = 10.0, 20.0``）：元组解包本就按位置，逐项对应。
                # 出现无法识别的项时整行放弃替换——孤儿符号由函数尾部的
                # 自由符号护栏拒绝（安全失败），不带病求值。
                ok = True
                for name, item in zip(names, items):
                    im = _TUPLE_ITEM_INDEX_RE.match(item)
                    if im:
                        k = int(im.group(1))
                        if k < len(params):
                            name_map[name] = str(params[k])
                    elif _TUPLE_ITEM_NUMBER_RE.match(item):
                        name_map[name] = item
                    else:
                        ok = False
                        break
                if not ok:
                    i += 1
                    continue
            i += 1
            continue
        # ④ **括号包裹的逐项解包**（可跨行）``p0, p1 = (params[0], params[1])``：
        #    RHS 以 ``(`` 开头，_TUPLE_UNPACK_RE 一律不认；合并成单行后走兜底。
        assign = _TUPLE_ASSIGN_RE.match(merged)
        if assign:
            resolved = _resolve_tuple_assignment(assign.group(1),
                                                 assign.group(2), params)
            if resolved is None:
                # 生成器/列表推导式 ``p0, p1, ... = (params[i] for i in range(n))``：
                # 与切片解包等价，见 _resolve_comprehension_unpack。
                resolved = _resolve_comprehension_unpack(assign.group(1),
                                                         assign.group(2), params)
            if resolved is not None:
                name_map.update(resolved)
                i += 1
                continue
        out_lines.append(merged)
        i += 1
    text = "\n".join(out_lines)
    # ⑤ 整体别名先按**下标**展开成对应参数值。顺序不能反：若先走下面的名字替换，
    # ``p`` 会变成"第一个参数的值"，``p[3]`` 就成了"数字[3]"（syntactically 合法但语义全错，
    # 且 sympy 只报内部 TypeError，极难反查）。
    for alias in array_aliases:
        def _expand_index(mm, _alias=alias):
            k = int(mm.group(1))
            return str(params[k]) if k < len(params) else mm.group(0)
        text = re.sub(rf"\b{re.escape(alias)}\s*\[\s*(\d+)\s*\]", _expand_index, text)
    # 展开后若仍残留裸的别名名字（例如 ``return p * 2`` 这种用法），留给函数尾部的
    # 自由符号护栏拒绝——**安全失败**，不带病求值。
    for name, value in name_map.items():
        text = re.sub(rf"\b{re.escape(name)}\b", value, text)
    return text


def expr_substitution(func: str, params: list) -> sp.Expr | None:
    """把 LLM 返回的函数骨架字符串替换为具体参数值，解析为 SymPy 表达式。

    失败（找不到自变量/return，或解析报错）时返回 None，由调用方兜底。

    已知边界：`==` / `!=` 只在 `where(...)` 生成的条件里被归一化为 `Eq` / `Ne`
    （见 :func:`normalize_condition`）。模型若直接手写 SymPy 风格的
    `Piecewise((a, x1 == 0), (b, True))`，其中的 `==` 仍会被 SymPy 折叠成
    Python 的 False 而丢掉该分支——该写法在本项目的提示词里并未出现，
    故未纳入改写范围。
    """
    params = _round_params(params or [])

    # 解析自变量列表：兼容逗号、中文逗号、空白分隔
    independent_match = re.search(r'Independents:\s*(.*)', func)
    if independent_match:
        independent = independent_match.group(1)
        independent_list = [
            v.strip() for v in re.split(r'[,，\s]+', independent) if v.strip()
        ]
    else:
        print("未找到自变量，返回 None")
        return None

    # 将具体数值代入参数（只替换实际存在的 params，避免越界）
    for i in range(len(params)):
        func = func.replace(f"params[{i}]", str(params[i]))

    # 去除注释部分 """...""" 和 #...
    pattern = "\"\"\"(.*?)\"\"\"|#[^\n]*"  # 非贪婪匹配
    func = re.sub(pattern, "", func, flags=re.DOTALL)  # 将匹配到的内容替换为""

    # 语句级预处理（两项写法在 MRFCompress-Cuboid_20260917-194203 双双踩中，
    # 后果都是 return 的符号无处可解、最终"表达式"退化为裸符号 sigma——
    # 剪枝率 0%、曲线报 Cannot convert expression to float）：
    func = _normalize_statements(func, params)

    # 显式 np.power(...) 必须在剥离 np. 前缀**之前**改写为 ((a)**(b))：模型可能已用
    # power 命名中间变量，剥掉前缀后裸 power(...) 会被解析成那个符号（Symbol 不可调用）。
    # 见 :func:`rewrite_numpy_power_calls`。
    func = rewrite_numpy_power_calls(func)
    # 同理，``np.clip(a, lo, None)`` 里的 ``None`` 会让 parse_expr 抛
    # ``'NoneType' object has no attribute 'is_Float'``；sympy 也没有 clip。见
    # :func:`rewrite_numpy_clip_calls`。
    func = rewrite_numpy_clip_calls(func)

    # 去掉 numpy 前缀，并把 maximum/minimum 别名映射到 SymPy 的 Max/Min。
    # 用 \b 限定标识符边界：原来的 str.replace 会把 `maximum_likelihood` 之类的
    # 名字一起改掉（`Max_likelihood`），静默产出错误的符号名。
    func = func.replace("np.", "").replace("numpy.", "")
    func = re.sub(r'\bmaximum\b', 'Max', func)
    func = re.sub(r'\bminimum\b', 'Min', func)

    # 把 where(cond, a, b) 改写为 Piecewise((a, cond), (b, True))
    # （直接写成 Piecewise 的表达式不受影响，SymPy 原生支持）
    try:
        func = rewrite_where_calls(func)
    except WhereArityError as e:
        print(f"[WARN] {e}，返回 None")
        return None

    inter_vars = {sp.Symbol(var_str): None for var_str in independent_list}

    # 识别中间变量赋值行。
    # 必须用锚定的赋值正则，不能用 `"=" in line` + `line.split("=")`：
    # `a = where(x1 >= 0, p0, p1)` 这类含比较运算符的行会被 split("=") 切成
    # eq_left=`a `、eq_right=`` 与残渣，解析必然失败并被 continue 跳过，
    # 于是 inter_vars 里没有 a；随后 `return a` 找不到替换目标，会**静默返回
    # 裸符号 a**（既不报错也不是 None），下游会把它当成"发现"的方程。
    for line in func.splitlines():
        assign_match = _ASSIGN_RE.match(line)
        if not assign_match:
            continue

        eq_left, eq_right = assign_match.group(1), assign_match.group(2)
        var = sp.Symbol(eq_left)
        try:
            expr = _parse_expr_with_symbols(
                eq_right, _known_names(independent_list, inter_vars))
        except Exception as e:
            # 跳过这一行本身可以（该变量可能没被 return 用到），但**不能因此交出
            # 含未定义符号的表达式**——函数尾部的自由符号校验会拦住那种情况。
            print(f"[WARN] 中间变量行解析失败（跳过）: {eq_left} = {eq_right} -> {e}")
            continue

        for symbol in expr.free_symbols:
            prev = inter_vars.get(symbol)   # 未定义变量安全跳过（不再 KeyError）
            if prev is not None:
                expr = expr.subs(symbol, prev)

        inter_vars[var] = expr

    match = re.search(r'return\s+(.*)', func, re.DOTALL)
    if not match:
        print("未找到 return，返回 None")
        return None

    expr_str = _unwrap_outer_parens(match.group(1).strip())
    try:
        expr = _parse_expr_with_symbols(
            expr_str, _known_names(independent_list, inter_vars))
    except Exception as e:
        print(f"[WARN] return 表达式解析失败，返回 None: {e}")
        return None

    for symbol in expr.free_symbols:
        prev = inter_vars.get(symbol)
        if prev is not None:
            expr = expr.subs(symbol, prev)

    # 不做 expr.n(2) 之类的有效数字舍入：那会把 357.8 压成 3.6e2，系数 ±0.5% 的
    # 失真在 ~2000 量级的项上放大成 ±30 的函数值偏差——精确拟合（NMSE 1e-29）
    # 的曲线会"神秘地"不穿过数据点（实测 MRFCompress-Cuboid_20260917-134427）。
    # 参数已在函数入口按 2 位小数舍入（±0.005，无损量级），此处保持全精度。
    # 自由符号校验：只允许自变量（外加历史约定保留的样本数符号 N）。
    # 中间变量行解析失败时它是被"跳过"的，于是 return 里的那个名字成了**孤儿符号**：
    # 表达式必然求值不出有限值，而"求值失败"在敏感度侧曾被当成"敏感度 0"，剪枝据此
    # 把每一项都删掉——实测最优样本因此从 NMSE 8.7e-07 塌缩成常数 193.054（NMSE 6.81，
    # 比"预测样本均值"的 1.0 还差 6.8 倍）。这种表达式交给下游只会静默产出错误结论，
    # 一律拒绝：返回 None，由调用方按"解析失败"处理（跳过剪枝，保留原式）。
    allowed = {str(name) for name in independent_list} | {"N"}
    unknown = sorted(str(s) for s in expr.free_symbols if str(s) not in allowed)
    if unknown:
        print(f"[WARN] 表达式含未定义符号 {unknown}（自变量只有 {independent_list}）"
              f"：无法求值，返回 None")
        return None

    print(f"代入中间变量后的表达式: {expr}")
    return expr


# ── 收尾自检：解析失败率（分「截断样本」与「解析器不支持的写法」两类） ──────

def _parse_capturing(func: str, params: list) -> tuple[sp.Expr | None, str]:
    """调用 :func:`expr_substitution` 并捕获其 stdout（WARN 文本），返回 (表达式, WARN)。

    自检要拿到"为什么解析不了"的那行 WARN，而它目前是 ``print`` 到 stdout 的——
    直接调用会让自检把几百行 WARN 刷进 run.out。这里把 stdout 重定向到缓冲，
    既保留诊断文本，又不污染日志。
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        expr = expr_substitution(func, params)
    return expr, buf.getvalue().strip()


def classify_sample(func: str | None, params: list) -> tuple[str, str]:
    """把一个样本函数体归为 ``ok`` / ``truncated`` / ``no_params`` / ``unsupported``。

    * ``"ok"``          —— :func:`expr_substitution` 得到表达式；
    * ``"truncated"``   —— 函数体没有 ``return``：多为 ``max_tokens`` 截断的**不完整
      样本**（评估器未打分，``score`` 为 None）。属于**采样侧**问题；
    * ``"no_params"``   —— 有 ``return`` 但样本**没有 params**（``params`` 为空/None）：
      评估器从未为它拟合出参数（``score`` 为 None），于是 ``params[k]`` 无从代换——
      实测 ``I.37.4_0_1`` 的 samples_7/9 即此形态，报错是"符号被下标"。这是
      **样本状态**问题，**不是**解析器缺陷；
    * ``"unsupported"`` —— 有 ``params`` 但 :func:`expr_substitution` 仍返回 None，
      即**解析器不支持的写法**，是解析器（本模块）要补的形态；WARN 说明卡在哪。

    后三类都算"解析失败"，但只有 ``unsupported`` 需要在解析器侧动手——这就是把
    它们分开计数、而不是给一个总百分比的原因。

    Returns:
        ``(kind, warn)``。
    """
    if not func or "return" not in func:
        return "truncated", ""
    expr, warn = _parse_capturing(func, params or [])
    if expr is not None:
        return "ok", warn
    # 解析失败且没有参数：先归因于"样本未被评估"（无 params），再谈解析器支持与否。
    # 注意先解析再判空——没有 params 的公式若本来就不含 params[k]，仍应算 ok。
    return ("no_params" if not params else "unsupported"), warn


def audit_parse_failures(results_root: str) -> dict:
    """扫 ``<results_root>/samples/*.json``，统计表达式**解析失败率**并分类计数。

    为什么要有它：收尾阶段要把选出的样本公式解析成 SymPy 表达式；解析失败时只在同目录
    ``run.out`` 里留一行 ``[WARN]``，``report.md`` 完全不提示——读者拿不到"这一次
    有多少样本根本解释不了"的总体数字，也看不出该修采样侧还是解析器侧。本函数把
    **全部已落盘样本**逐个过 :func:`expr_substitution`，按 :func:`classify_sample`
    拆成三类（截断样本 / 无参数样本 / 解析器不支持的写法）给出——只有最后一类需要在
    解析器侧动手。

    只读磁盘、不写盘、不调 LLM。文件名带 ``top`` 的 Top-K 副本与全量文件按
    ``sample_order`` 去重（优先全量），口径与
    :func:`drsr_420.core.sample_records.load_sample_records` 一致，但**不过滤
    ``score`` 为 None 的样本**——截断/未评估样本正是要计数的对象。

    Returns:
        dict：``n_total`` / ``n_ok`` / ``n_failed`` / ``n_truncated`` / ``n_no_params`` /
        ``n_unsupported`` / ``failure_rate``（失败数/总数；总数为 0 时 ``None``）/
        ``truncated`` / ``no_params`` / ``unsupported``（失败明细，各含 ``file`` /
        ``sample_order`` / ``score``，后两类的 ``unsupported`` 另含 ``warn``）。
    """
    pattern = os.path.join(results_root or ".", "samples", "*.json")
    unique: dict = {}
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] 解析自检：读取样本失败，跳过 {name}: {e}")
            continue
        order = data.get("sample_order")
        key = order if order is not None else name
        is_top = name.startswith("top")
        prev = unique.get(key)
        if prev is not None and not (prev["is_top"] and not is_top):
            continue
        unique[key] = {"file": name, "is_top": is_top, "sample_order": order,
                       "score": data.get("score"),
                       "function": data.get("function") or "",
                       "params": data.get("params") or []}

    truncated, no_params, unsupported = [], [], []
    for rec in unique.values():
        kind, warn = classify_sample(rec["function"], rec["params"])
        if kind == "truncated":
            truncated.append(rec)
        elif kind == "no_params":
            no_params.append(rec)
        elif kind == "unsupported":
            unsupported.append({**rec, "warn": warn})

    def _order_key(rec: dict):
        return (rec["sample_order"] is None, rec["sample_order"] or 0)

    truncated.sort(key=_order_key)
    no_params.sort(key=_order_key)
    unsupported.sort(key=_order_key)
    n_total = len(unique)
    n_failed = len(truncated) + len(no_params) + len(unsupported)
    return {
        "n_total": n_total,
        "n_ok": n_total - n_failed,
        "n_failed": n_failed,
        "n_truncated": len(truncated),
        "n_no_params": len(no_params),
        "n_unsupported": len(unsupported),
        "failure_rate": (n_failed / n_total) if n_total else None,
        "truncated": truncated,
        "no_params": no_params,
        "unsupported": unsupported,
    }
