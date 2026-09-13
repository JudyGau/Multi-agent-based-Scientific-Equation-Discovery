"""物理解释：让 LLM 对最优公式做逐项力学解释，并落盘 ``explain.txt``。

角色归属
--------
收尾分析（analysis）阶段的**可读性产物**：把"数学上最优"翻译成"物理上讲得通"。

协作
----
* 输入：``experiences.json`` 里该样本的 Good 条目（含模型的思考过程与含参公式）；
* LLM：通过 ReAct 循环（``explain_re_act``）调用，模型可自行发起 MCP 检索工具；
* 增强：RAG 知识库注入相关文献摘要（库为空或检索失败则静默跳过）；
* 产物：``<results_root>/explain.txt``。

失败策略：任一环节（无经验文件 / 无匹配条目 / 提示词构造失败 / LLM 初始化失败 /
保存失败）都只告警并返回，绝不抛出——收尾流程后面还有剪枝与可视化要做。
"""
from __future__ import annotations

import json
import os
import re

from drsr_420.core.console import LineStreamPrinter, print_block
from drsr_420.core import prompt_config as pc
import drsr_420.llm as llm
from drsr_420.knowledge.tool_runner import mcp_call_tool


def explain_re_act(client: llm.LLMClient, content: str) -> str | None:
    """ReAct 循环：流式对话，模型调工具就执行并回传，直到它给出最终答复。"""
    if client is None:
        return None
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
                # 局部变量名刻意不叫 content：外层 content 是提示词，历史上被这里的
                # 同名赋值覆盖过（ReAct 第二轮再读提示词就会拿到最后一段正文）。
                chunk_content = chunk.get('content') or ''
                text = reasoning + chunk_content
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
        return None


def build_explain_content(func: str, exp: dict) -> str | None:
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


def explain_best_sample(results_root: str, func: str, sample_order: str) -> None:
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

    content = build_explain_content(func, matched)
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
