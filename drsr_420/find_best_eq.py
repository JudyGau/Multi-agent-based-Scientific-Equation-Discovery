import json, glob, os
import re
import sympy as sp
from graphviz import Source
from sympy import nsimplify, dotprint
from sympy.parsing import sym_expr

import llm
from drsr_420.sensitivity_prune import SensitivityPruner
from drsr_420.tool_runner import mcp_call_tool


def explain_re_act(client: llm.LLMClient, content: str) -> str | None:
    if client is not None:
        try:
            # responses = []
            # think_responses = []

            messages = []
            messages.append({"role": "user", "content": content})
            resp = client.chat([{"role": "user", "content": content}])

            while True:
                print("========================思考过程========================\n")
                print(resp.get('reasoning_content', ''))
                print("====================================================\n")

                tool_calls = resp.get('tool_calls', [])
                messages.append({"role": "assistant", "content": resp.get('content', ''), "tool_calls": tool_calls})

                # 如果调了 tool，执行后回传
                if tool_calls:
                    print("调用了工具：", tool_calls)

                    for tc in tool_calls:
                        fn_name = tc.get('function', {}).get('name', '')
                        args = json.loads(tc.get('function', {}).get('arguments', []))
                        result = mcp_call_tool(fn_name, args)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get('id', ''),
                            "content": result
                        })

                    resp = client.chat(messages)
                # 如果未调用，则跳出循环
                else:
                    return resp.get('content', '')
                    # responses.append(resp.get('content', ''))
                    # think_responses.append(resp.get('reasoning_content', ''))

            # return (responses, think_responses) if self._batch_inference else (responses[0], think_responses[0])
        except Exception as e:
            print(f"API请求发生错误: {str(e)}")
            # return ([""] * repeat_prompt, [""] * repeat_prompt) if self._batch_inference else ("", "")

def expr_substitution(func: str, params: list) -> sp.Expr | None:
    """把 LLM 返回的函数骨架字符串替换为具体参数值，解析为 SymPy 表达式。

    失败（找不到自变量/return，或解析报错）时返回 None，由调用方兜底。
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

    # 将np.替换为sp. 将maximum替换为Max 将where替换为Piecewise
    func = func.replace("np.","").replace("maximum", "Max").replace("minimum", "Min").replace("where", "Piecewise")

    # 更改 Piecewise 函数的调用格式
    def replfunc(m):
        parameter_str = m.group(0)
        parameter_list = parameter_str.split(',')
        cond, true_val, false_val  = parameter_list[0].strip().strip("Piecewise("), parameter_list[1].strip(), parameter_list[2].strip().strip(")\n")
        return f"Piecewise(({true_val}, {cond}), ({false_val}, True))\n"

    func = re.sub("Piecewise\\(.*?\\)\n",replfunc, func,flags=re.DOTALL)

    inter_vars = {sp.Symbol(var_str): None for var_str in independent_list}

    # 遍历func的每行
    for line in func.splitlines():
        equal_match = re.search(r'=', line)

        if equal_match:
            equation = line.split("=")

            eq_left = equation[0].strip()
            var = sp.Symbol(eq_left)

            eq_right = equation[1].strip()
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
        else:
            print("跳过该行")


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


def find_best_eq(results_root: str, threshold: float = 0.1,
                 sample_range: tuple = (1, 14)):
    # results_root = "experiments/MRFCompress-Cuboid_20260712-150715"  # 改为你的目录
    best = None
    for p in glob.glob(os.path.join(results_root, "samples", "*_samples_*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            s = d.get("score")
            if s is None:
                continue
            if best is None or s > best[0]:
                best = (s, p, d.get("function",""), d.get("params"))
        except Exception:
            continue

    if best is None:
        print("没有找到有效样本。")
    else:
        score, path, func, params = best

        sample_order = re.search(r"samples_(\d+)",path).group(1)

        try:
            with open(os.path.join(results_root,  "experiences.json"), "r", encoding="utf-8") as f:
                exp = json.load(f)
                good_exp = exp["Good"]
                for exp in good_exp:
                    if exp["sample_order"].__str__() == sample_order:

                        thinking_content = exp["thinking_content"]
                        thinking_content = thinking_content.rsplit('\n', 1)[0]
                        thinking_content ="以下是另一个LLM给出的公式推导（思考过程）:\n" + thinking_content

                        return_eq = exp["equation"]
                        eq=re.search(r'return\s+(.*)', return_eq).group(1)
                        eq = "以下是另一个LLM给出的含参本构公式:\n" + eq

                        dependent = re.search(r'Dependent:\s*(\w+)', func).group(1)
                        independent = re.search(r'Independents:\s+(.*)', func).group(1)

                        head = (f"你是一名力学工程师/应用力学家，对给定公式做逐项力学解释，以下是一个磁流变液的含参本构公式和这个公式的推导逻辑，因变量是磁流变效应 {dependent}，" +
                                f"自变量是 {independent}，其中lambda12＝L1/L2 和 lambda23=L2/L3，L1, L2, and L3 分别是颗粒的长轴，中轴和短轴,请你据此对这个公式从力学角度进行详细的解释")
                        tail = "请你根据以上内容对这个公式从力学角度进行详细的解释"
                        # RAG 检索增强：注入相关文献背景，帮助模型从物理机理角度解释（失败/库为空时静默跳过）
                        rag_block = ""
                        try:
                            from drsr_420.rag_kb import get_kb, load_config
                            _rag_cfg = load_config()
                            rag_block = get_kb().get_context(
                                _rag_cfg.get('default_query', independent), k=_rag_cfg.get('k', 5))
                        except Exception as _e:
                            print(f"[RAG] 解释阶段文献检索失败（跳过）: {_e}")
                        content = head + "\n" + eq + "\n" + thinking_content \
                            + ("\n\n### 以下是相关文献背景，供力学解释参考 ###\n\n" + rag_block if rag_block else "") \
                            + "\n" + tail

                        llm_config = llm.load_llm_config("llm_explain.config")
                        # 由 ClientFactory 统一完成 provider 解析、api_key 解析、base_url 与生成参数注入
                        client = None
                        try:
                            client = llm.ClientFactory.from_config(llm_config)
                            print(f"[INFO] LLM client initialized: provider={client._provider_name()}, model={client.model}, kwargs={client.kwargs}")
                        except Exception as e:
                            print(f"[WARN] Failed to init LLM client: {e}")

                        # resp = client.chat([{"role": "user", "content": content}])
                        # explain = resp.get("content","")

                        explain = explain_re_act(client, content)
                        print(explain)

                        # 将动态渲染的 explain 保存到本次实验目录，便于调试
                        try:
                            explain_out_path = os.path.join(results_root, "explain.txt")
                            with open(explain_out_path, "w", encoding="utf-8") as f:
                                f.write(explain)
                            print(f"[INFO] Saved dynamic spec to: {explain_out_path}")
                        except Exception as e:
                            print(f"[WARN] Failed to save dynamic spec: {e}")



                        break
        except Exception as e:
            print(f"[WARN] Failed to init LLM client: {e}")
            pass

        # params = [round(x, 2) for x in params]
        #
        # match = re.search(r'Dependent:\s*(\w+)', func)
        # if match:
        #     # print(match.group(1))
        #     dependent = match.group(1)
        #     # func=func.replace("return",f"return {dependent} =")
        # else:
        #     print("未找到因变量")
        #
        #
        # for i in range(len(params)):
        #     func=func.replace(f"params[{i}]", str(params[i]))
        #
        # match = re.search(r'return\s+(.*)', func)
        # if match:
        #     expr_str = match.group(1)
        #     print(expr_str)
        # else:
        #     print("未找到 return")

        # print(f"[BEST] score={score} file={path} params={params} function:{func}")

        # ── 基于敏感度分析剪枝：移除对输出影响小于阈值的项 ──
        dependent = re.search(r'Dependent:\s*(\w+)', func)
        independent = re.search(r'Independents:\s*(.*)', func)
        if not dependent or not independent:
            print("[WARN] 无法从样本中解析 Dependent/Independents，跳过剪枝。")
            return
        dependent = dependent.group(1)
        independent_str = independent.group(1)

        # 解析自变量符号列表（兼容逗号/中文逗号/空白分隔）
        sym_names = [
            v.strip() for v in re.split(r'[,，\s]+', independent_str) if v.strip()
        ]
        symbols = sp.symbols(sym_names)
        if not symbols:
            print("[WARN] 自变量列表为空，跳过剪枝。")
            return

        # 创建智能剪枝器（阈值/采样区间可配置）
        pruner = SensitivityPruner(
            symbols=symbols,
            threshold=threshold,
            sample_range=sample_range,
        )

        expr = expr_substitution(func, params)
        if expr is None:
            print("[WARN] 表达式解析失败，跳过剪枝。")
            return

        print(f"剪枝前的表达式为 {dependent} =")
        sp.pprint(expr)
        # 保存为PNG图片（无 latex/graphviz 环境时容错）
        _safe_preview(expr, f'{results_root}/expr.png')

        # 使用智能剪枝
        try:
            pruned_expr = pruner.prune(expr, verbose=True)
        except Exception as e:
            print(f"[WARN] 剪枝失败: {e}")
            return

        pruned_expr = pruned_expr.n(2)

        print(f"剪枝后的表达式为 {dependent} =")
        sp.pprint(pruned_expr)
        _safe_preview(pruned_expr, f'{results_root}/prunedExpr.png')

        # 表达式树可视化（依赖 graphviz，失败时仅告警，不中断流程）
        print(".......")
        for name, e in (("original_expr_tree", expr), ("pruned_expr_tree", pruned_expr)):
            try:
                src = Source(dotprint(e))
                src.render(f'{results_root}/{name}', view=True)
            except Exception as ex:
                print(f"[WARN] 生成表达式树图失败（{name}，可能缺少 graphviz 环境）: {ex}")


if __name__ == "__main__":
    find_best_eq("..\\experiments\\MRFCompress-Ellipsoid_20260813-133229")