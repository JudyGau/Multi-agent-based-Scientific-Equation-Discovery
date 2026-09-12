import json, glob, os
import re
import threading
import sympy as sp
from sympy import nsimplify, dotprint
from sympy.parsing import sym_expr
from drsr_420.core.console import LineStreamPrinter, print_block

# graphviz 为可选依赖：未安装时 Source=None，表达式树渲染会跳过（已有 try/except 兜底）。
# 此前顶层硬导入会使 pipeline.py（依赖本模块）在缺 graphviz 环境下整体无法 import。
try:
    from graphviz import Source
except ImportError:  # pragma: no cover - 缺 graphviz 时降级
    Source = None

import drsr_420.llm as llm
from drsr_420.core import prompt_config as pc
from drsr_420.analysis.sensitivity_prune import SensitivityPruner
from drsr_420.knowledge.tool_runner import mcp_call_tool


def explain_re_act(client: llm.LLMClient, content: str) -> str | None:
    if client is not None:
        try:
            messages = [
                {"role": "system", "content": pc.sampling_system_prompt},
                {"role": "user", "content": content},
            ]

            while True:
                # 流式迭代：reasoning 与 content 按到达顺序实时打印增量（网络层已是 SSE 流式）
                resp = None
                stream = LineStreamPrinter()
                shown = 0  # 已实时打印的字符数（reasoning 在前、content 在后拼接）
                think_label_printed = False
                content_label_printed = False
                for chunk in client.chat_stream(messages):
                    if chunk.get('final'):
                        resp = {k: v for k, v in chunk.items() if k != 'final'}
                        break
                    reasoning = chunk.get('reasoning_content') or ''
                    content = chunk.get('content') or ''
                    text = reasoning + content
                    if len(text) > shown:
                        if shown < len(reasoning) and not think_label_printed:
                            stream.write("[思考]\n")
                            think_label_printed = True
                        elif not content_label_printed:
                            stream.write_line("[正文]")
                            content_label_printed = True
                        stream.write(text[shown:])
                        shown = len(text)
                stream.flush()
                if resp is None:
                    return None
                print("\n====================================================\n")

                tool_calls = resp.get('tool_calls', [])
                messages.append({"role": "assistant", "content": resp.get('content', ''), "tool_calls": tool_calls})

                # 如果调了 tool，执行后回传
                if tool_calls:
                    print("调用了工具：", tool_calls)

                    for tc in tool_calls:
                        fn_name = tc.get('function', {}).get('name', '')
                        args = json.loads(tc.get('function', {}).get('arguments', '{}'))
                        result = mcp_call_tool(fn_name, args)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get('id', ''),
                            "content": result
                        })
                # 如果未调用，则跳出循环
                else:
                    return resp.get('content', '')
        except Exception as e:
            print(f"API请求发生错误: {str(e)}")

def _find_matching_paren(text: str, open_idx: int) -> int:
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


def _split_top_level(text: str, sep: str = ',') -> list:
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


def _normalize_condition(cond: str) -> str:
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
                left = _normalize_condition(cond[:i]).strip()
                right = _normalize_condition(cond[i + 2:]).strip()
                return f"Eq({left}, {right})"
            if cond.startswith('!=', i):
                left = _normalize_condition(cond[:i]).strip()
                right = _normalize_condition(cond[i + 2:]).strip()
                return f"Ne({left}, {right})"
        i += 1
    return cond


class WhereArityError(ValueError):
    """`where(...)` 参数个数不是 3（numpy 只有 where(cond, x, y) 这一种公式用法）。"""


def _rewrite_where_calls(text: str) -> str:
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
    `==` / `!=` 一并转成 `Eq` / `Ne`，见 `_normalize_condition`。
    """
    out, pos = [], 0
    while True:
        m = _WHERE_CALL_RE.search(text, pos)
        if m is None:
            out.append(text[pos:])
            return ''.join(out)

        open_idx = m.end() - 1              # 正则末尾 '(' 的下标
        close_idx = _find_matching_paren(text, open_idx)
        out.append(text[pos:m.start()])
        if close_idx < 0:                   # 括号不闭合：原样保留，交由下游解析报错
            out.append(text[m.start():])
            return ''.join(out)

        inner = text[open_idx + 1:close_idx]
        args = [a.strip() for a in _split_top_level(inner)]
        while args and args[-1] == '':      # 容忍 `where(c, a, b,)` 这类尾随逗号
            args.pop()
        if len(args) != 3:
            # 不能只把名字改成 Piecewise 交差：`Piecewise(x1 > 0)` 会被 SymPy
            # 解释成空 Piecewise 并返回 nan（静默毒化表达式），
            # `Piecewise(c, a)` 才会报错。两种结果都不可接受，直接显式失败。
            raise WhereArityError(
                f"where(...) 需要 3 个参数 (cond, true_val, false_val)，实际 {len(args)} 个：{inner.strip()!r}"
            )
        cond, true_val, false_val = [_rewrite_where_calls(a) for a in args]
        out.append(f"Piecewise(({true_val}, {_normalize_condition(cond)}), ({false_val}, True))")
        pos = close_idx + 1


def expr_substitution(func: str, params: list) -> sp.Expr | None:
    """把 LLM 返回的函数骨架字符串替换为具体参数值，解析为 SymPy 表达式。

    失败（找不到自变量/return，或解析报错）时返回 None，由调用方兜底。

    已知边界：`==` / `!=` 只在 `where(...)` 生成的条件里被归一化为 `Eq` / `Ne`
    （见 `_normalize_condition`）。模型若直接手写 SymPy 风格的
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

    # 去掉 numpy 前缀，并把 maximum/minimum 别名映射到 SymPy 的 Max/Min。
    # 用 \b 限定标识符边界：原来的 str.replace 会把 `maximum_likelihood` 之类的
    # 名字一起改掉（`Max_likelihood`），静默产出错误的符号名。
    func = func.replace("np.", "").replace("numpy.", "")
    func = re.sub(r'\bmaximum\b', 'Max', func)
    func = re.sub(r'\bminimum\b', 'Min', func)

    # 把 where(cond, a, b) 改写为 Piecewise((a, cond), (b, True))
    # （直接写成 Piecewise 的表达式不受影响，SymPy 原生支持）
    try:
        func = _rewrite_where_calls(func)
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
            expr = sp.parse_expr(eq_right, {'N': sp.Symbol('N')})
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

    expr_str = match.group(1).strip()
    # 兼容多行括号式 return：
    #   return (
    #       expr1
    #       + expr2
    #   )
    # 取出最外层括号内的内容并合并为单行，再交给 parse_expr。
    if expr_str.startswith('('):
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
        expr_str = expr_str[1:end_idx]
    expr_str = ' '.join(expr_str.splitlines())
    try:
        expr = sp.parse_expr(expr_str, {'N': sp.Symbol('N')})
    except Exception as e:
        print(f"[WARN] return 表达式解析失败，返回 None: {e}")
        return None

    for symbol in expr.free_symbols:
        prev = inter_vars.get(symbol)
        if prev is not None:
            expr = expr.subs(symbol, prev)

    expr = expr.n(2)
    print(f"代入中间变量后的表达式: {expr}")
    return expr


def _safe_preview(expr, filename: str) -> None:
    """容错地保存表达式预览图；缺 latex/工具链时仅告警，不中断流程。"""
    try:
        sp.preview(expr, output='png', filename=filename, viewer='file')
    except Exception as e:
        print(f"[WARN] 保存表达式图片失败（{filename}）: {e}")


def _find_best_sample(results_root: str):
    """扫描 samples 目录，返回分数最高的样本 (score, path, func, params)；无则 None。"""
    best = None
    for p in glob.glob(os.path.join(results_root, "samples", "*_samples_*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            s = d.get("score")
            if s is None:
                continue
            if best is None or s > best[0]:
                best = (s, p, d.get("function", ""), d.get("params"))
        except Exception:
            continue
    return best


def _build_explain_content(func: str, exp: dict) -> str | None:
    """从样本函数与匹配的经验条目构造物理解释提示词；解析失败返回 None。"""
    thinking = exp.get("thinking_content", "")
    if not thinking:
        return None
    thinking = thinking.rsplit('\n', 1)[0]
    thinking = "以下是另一个LLM给出的公式推导（思考过程）:\n" + thinking

    return_eq = exp.get("equation", "")
    eq_match = re.search(r'return\s+(.*)', return_eq)
    if not eq_match:
        return None
    eq = "以下是另一个LLM给出的含参本构公式:\n" + eq_match.group(1)

    dep_match = re.search(r'Dependent:\s*(\w+)', func)
    ind_match = re.search(r'Independents:\s+(.*)', func)
    if not dep_match or not ind_match:
        return None
    dependent = dep_match.group(1)
    independent = ind_match.group(1)

    head = (f"你是一名力学工程师/应用力学家，对给定公式做逐项物理机理解释，以下是一个含参本构公式和这个公式的推导逻辑，"
            f"因变量是 {dependent}，自变量是 {independent}，请你据此对这个公式从力学角度进行详细的解释。"
            "具体的领域背景请参考下方提供的文献摘要。")
    tail = "请你根据以上内容对这个公式从力学角度进行详细的解释"

    # RAG 检索增强：注入相关文献背景（失败/库为空时静默跳过）
    rag_block = ""
    try:
        from drsr_420.knowledge.rag_kb import get_kb, load_config
        _rag_cfg = load_config()
        rag_block = get_kb().get_context(_rag_cfg.get('default_query') or independent, k=_rag_cfg.get('k', 5))
    except Exception as _e:
        print(f"[RAG] 解释阶段文献检索失败（跳过）: {_e}")

    content = head + "\n" + eq + "\n" + thinking \
        + ("\n\n### 以下是相关文献背景，供力学解释参考 ###\n\n" + rag_block if rag_block else "") \
        + "\n" + tail
    return content


def _explain_best_sample(results_root: str, func: str, sample_order: str) -> None:
    """按 sample_order 匹配 Good 经验条目，调用 LLM 生成物理解释并落盘 explain.txt。

    任意环节失败（无经验文件 / 无匹配条目 / 提示词构造失败 / LLM 初始化失败）
    均只告警并返回，不抛出，避免影响后续剪枝流程。
    """
    exp_path = os.path.join(results_root, "experiences.json")
    try:
        with open(exp_path, "r", encoding="utf-8") as f:
            exp_data = json.load(f)
    except Exception as e:
        print(f"[WARN] 读取经验文件失败，跳过物理解释: {e}")
        return

    matched = None
    for exp in exp_data.get("Good", []):
        if str(exp.get("sample_order")) == sample_order:
            matched = exp
            break
    if matched is None:
        print(f"[WARN] 未找到 sample_order={sample_order} 的 Good 经验，跳过物理解释。")
        return

    content = _build_explain_content(func, matched)
    if content is None:
        print("[WARN] 构造物理解释提示词失败，跳过。")
        return

    # 初始化 LLM 客户端（公式解释任务，由 ClientFactory 统一注入 provider/api_key/参数）
    client = None
    try:
        llm_config = llm.load_llm_config("deepseek_deepseek-v4-flash.config")
        client = llm.ClientFactory.from_config(llm_config)
        if client is not None:
            client = client.clone_for_task('explain')
        print(f"[INFO] LLM client initialized: provider={client._provider_name()}, model={client.model}, kwargs={client.kwargs}")
    except Exception as e:
        print(f"[WARN] Failed to init LLM client: {e}")

    explain = explain_re_act(client, content)
    print_block(explain if explain is not None else "")

    try:
        explain_out_path = os.path.join(results_root, "explain.txt")
        with open(explain_out_path, "w", encoding="utf-8") as f:
            f.write(explain or "")
        print(f"[INFO] Saved explain to: {explain_out_path}")
    except Exception as e:
        print(f"[WARN] Failed to save explain: {e}")


def _render_expr_trees(results_root: str, expr, pruned_expr) -> None:
    """表达式树可视化（依赖 graphviz，缺失或失败时仅告警，不中断流程）。"""
    for name, e in (("original_expr_tree", expr), ("pruned_expr_tree", pruned_expr)):
        if Source is None:
            print(f"[WARN] 跳过表达式树图（{name}）：未安装 graphviz")
            continue
        try:
            src = Source(dotprint(e))
            src.render(f'{results_root}/{name}', view=True)
        except Exception as ex:
            print(f"[WARN] 生成表达式树图失败（{name}，可能缺少 graphviz 环境）: {ex}")


def _prune_and_visualize(results_root: str, func: str, params,
                         threshold: float, sample_range: tuple) -> None:
    """基于敏感度分析剪枝最优公式，并保存表达式预览图与表达式树图。"""
    dependent_match = re.search(r'Dependent:\s*(\w+)', func)
    independent_match = re.search(r'Independents:\s*(.*)', func)
    if not dependent_match or not independent_match:
        print("[WARN] 无法从样本中解析 Dependent/Independents，跳过剪枝。")
        return
    dependent = dependent_match.group(1)
    independent_str = independent_match.group(1)

    # 解析自变量符号列表（兼容逗号/中文逗号/空白分隔）
    sym_names = [v.strip() for v in re.split(r'[,，\s]+', independent_str) if v.strip()]
    symbols = sp.symbols(sym_names)
    if not symbols:
        print("[WARN] 自变量列表为空，跳过剪枝。")
        return

    pruner = SensitivityPruner(symbols=symbols, threshold=threshold, sample_range=sample_range)
    expr = expr_substitution(func, params)
    if expr is None:
        print("[WARN] 表达式解析失败，跳过剪枝。")
        return

    print(f"剪枝前的表达式为 {dependent} =")
    sp.pprint(expr)
    _safe_preview(expr, f'{results_root}/expr.png')

    try:
        pruned_expr = pruner.prune(expr, verbose=True)
    except Exception as e:
        print(f"[WARN] 剪枝失败: {e}")
        return

    pruned_expr = pruned_expr.n(2)
    print(f"剪枝后的表达式为 {dependent} =")
    sp.pprint(pruned_expr)
    _safe_preview(pruned_expr, f'{results_root}/prunedExpr.png')

    _render_expr_trees(results_root, expr, pruned_expr)


def find_best_eq(results_root: str, threshold: float = 0.1,
                 sample_range: tuple = (1, 14)):
    """收尾：寻找最优样本 → 生成物理解释 → 敏感度剪枝与可视化。

    主函数仅做扁平编排，具体逻辑拆分到 _find_best_sample / _explain_best_sample /
    _prune_and_visualize 三个单一职责 helper，避免原先 try-with-for-if-try 的深嵌套。
    """
    best = _find_best_sample(results_root)
    if best is None:
        print("没有找到有效样本。")
        return

    score, path, func, params = best
    print(f"[BEST] score={score} file={path}")

    # 物理解释（按 sample_order 匹配 Good 经验，含 RAG 文献注入）
    order_match = re.search(r"samples_(\d+)", path)
    if not order_match:
        print("[WARN] 无法从样本文件名解析 sample_order，跳过物理解释。")
    else:
        _explain_best_sample(results_root, func, order_match.group(1))

    # 敏感度剪枝 + 表达式预览/树图
    _prune_and_visualize(results_root, func, params, threshold, sample_range)


if __name__ == "__main__":
    find_best_eq("..\\experiments\\MRFCompress-Ellipsoid_20260813-133229")