"""表达式解析：把 LLM 生成的函数骨架字符串变成可求值的 SymPy 表达式。

角色归属
--------
收尾分析（analysis）阶段的**第一步**：参数代入 + 语法归一化 + 表达式解析。
无状态、不调用 LLM、不写磁盘。

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

import re

import sympy as sp


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
    return sp.parse_expr(text, local)


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
    r"^\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*=\s*params(?:\[[^\]]*\])?\s*$")
_PAREN_OPEN = re.compile(r"[(\[{]")
_PAREN_CLOSE = re.compile(r"[)\]}]")


def _paren_delta(line: str) -> int:
    """一行内未闭合的括号数（正 = 还有括号没关上）。"""
    return len(_PAREN_OPEN.findall(line)) - len(_PAREN_CLOSE.findall(line))


def _normalize_statements(func: str, params: list) -> str:
    """语句级预处理，替 return 的符号消解扫清两种 LLM 常见写法：

    ① **元组解包** ``p0, p1, ..., p9 = params[:10]``：旧流程只会替换
       ``params[i]`` 形式，解包出来的名字全部留在表达式里成为自由符号，
       ``return sigma`` 的 sigma 无处可解，最终"表达式"退化为裸符号——
       剪枝率恒 0、曲线报 Cannot convert expression to float
       （实测 MRFCompress-Cuboid_20260917-194203）。按位置把每个名字
       替换为数值后，后续流程照常工作。
    ② **多行赋值** ``sigma = (\\n  p0\\n  + p1 ...)``：赋值正则按单行匹配，
       只能看到 ``sigma = (`` 就 EOF 报错被跳过，同样退化为裸符号。
       按括号配对把续行合并回单行（对多行 ``return (`` 同样生效）。
    """
    name_map: dict[str, str] = {}
    out_lines: list[str] = []
    lines = func.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _TUPLE_UNPACK_RE.match(line)
        if m:
            names = [n.strip() for n in m.group(1).split(",")]
            for j, name in enumerate(names):
                if j < len(params):
                    name_map[name] = str(params[j])
            i += 1
            continue
        merged = line.strip()
        while _paren_delta(merged) > 0 and i + 1 < len(lines):
            i += 1
            merged = merged + " " + lines[i].strip()
        out_lines.append(merged)
        i += 1
    text = "\n".join(out_lines)
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
    params = [round(x, 2) for x in (params or [])]

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
    print(f"代入中间变量后的表达式: {expr}")
    return expr
