import json, glob, os
import re
import sympy as sp
from graphviz import Source
from sympy import nsimplify, dotprint
from sympy.parsing import sym_expr

import llm
from drsr_420.evaluate_on_problems import MAX_NPARAMS

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

def expr_substitution(func: str, params: list) ->  sp.Expr | None:

    params = [round(x, 2) for x in params]


    independent_match = re.search(r'Independents:\s+(.*)', func)
    if independent_match:
        independent = independent_match.group(1)
        independent_list = independent.split(', ')
    else:
        print("未找到自变量")

    # 将具体数值代入参数
    for i in range(MAX_NPARAMS):
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

    inter_vars = {}
    for var_str in independent_list:
        var = sp.Symbol(var_str)
        inter_vars[var] = None

    # 遍历func的每行
    for line in func.splitlines():
        equal_match = re.search(r'=', line)

        if equal_match:
            # independent = equal_match.group(1)

            equation = line.split("=")

            eq_left = equation[0].strip()
            var = sp.Symbol(eq_left)

            eq_right = equation[1].strip()
            expr = sp.parse_expr(eq_right,{'N': sp.Symbol('N')})

            for symbol in expr.free_symbols:
                if inter_vars[symbol] is not None:
                    expr = expr.subs(symbol, inter_vars[symbol])

            inter_vars[var] = expr

            # expr = expr.n(2)
        else:
            print("跳过该行")


    match = re.search(r'return\s+(.*)', func, re.DOTALL)
    if match:
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
        expr = sp.parse_expr(expr_str, {'N': sp.Symbol('N')})
        for symbol in expr.free_symbols:
            if inter_vars[symbol] is not None:
                expr = expr.subs(symbol, inter_vars[symbol])

        expr = expr.n(2)
        print(f"代入中间变量后的表达式: {expr}")
        return expr
    else:
        print("未找到 return")


def find_best_eq(results_root: str):
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

                        with open("./llm_explain.config", 'r', encoding='utf-8') as f:
                            llm_config = json.load(f)
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

        """
                基于敏感度分析剪枝
                移除对输出影响小于阈值的项
                """
        dependent = re.search(r'Dependent:\s*(\w+)', func).group(1)
        independent = re.search(r'Independents:\s+(.*)', func).group(1)

        # 加上','以便变量字符串被转换为元组tuple
        independent = independent + ','

        # 创建智能剪枝器
        pruner = SensitivityPruner(
            # X_sample=X_sample,
            # expr=expr,
            symbols=sp.symbols(independent),
            threshold=0.1,
            sample_range=(1, 14)
            # mse_threshold=0.1  # 默认允许10% MSE增加
        )

        # 字符串替换 np.log -> log, np.exp -> exp
        # expr_str = expr_str.replace('np.asarray','').replace('np.log', 'log').replace('np.exp', 'exp').replace('np.sin', 'sin').replace('np.cos', 'cos').replace('np.sqrt','sqrt').replace('np.maximum','maximum')

        # # 转换为 SymPy 表达式
        # expr = sp.parse_expr(expr_str)
        # expr = expr.n(2)

        expr = expr_substitution(func, params)
        print(f"剪纸前的表达式式为 {dependent} =")
        sp.pprint(expr)
        # 保存为PNG图片
        sp.preview(expr, output='png', filename=f'{results_root}/expr.png', viewer='file')

        # 使用智能剪枝
        pruned_expr = pruner.prune(expr, verbose=True)


        pruned_expr=pruned_expr.n(2)

        # pruned_expr = sp.simplify(pruned_expr)

        print(f"剪纸后的表达式式为 {dependent} =")
        sp.pprint(pruned_expr)
        # 保存为PNG图片
        sp.preview(pruned_expr, output='png', filename=f'{results_root}/prunedExpr.png', viewer='file')

        print(".......")

        src1 = Source(dotprint(expr))
        src1.render(f'{results_root}/original_expr_tree', view=True)  # 保存为PDF并显示

        src2 = Source(dotprint(pruned_expr))
        src2.render(f'{results_root}/pruned_expr_tree', view=True)  # 保存为PDF并显示


if __name__ == "__main__":
    find_best_eq("..\\experiments\\MRFCompress-Ellipsoid_20260813-133229")