"""工具调用 Agent：与 LLM 多轮对话，自动执行模型发起的工具调用并把结果回传给模型。

角色：ToolCallerAgent 是"工具调用者"，为 SamplerAgent 提供 MCP 工具循环能力。

协作：
- 上游：SamplerAgent（通过 complete() 发起多轮工具调用）；
- 下游：tool_runner.mcp_call_tool（实际执行 MCP 工具）。
"""
from __future__ import annotations

import json
import threading

from drsr_420.console import LineStreamPrinter, print_block
from drsr_420 import prompt_config as pc


class ToolCallerAgent:
    """与 LLM 多轮对话：若模型返回 tool_calls 则执行工具并回传，直到模型给出最终答复。

    Args:
        llm_client: LLMClient 实例，提供 chat(messages) -> dict，
                    返回字段含 content / reasoning_content / tool_calls。
        tool_executor: 可调用对象 (name, args) -> str，默认使用 tool_runner.mcp_call_tool。
        max_tool_rounds: 单个样本内工具调用轮次上限，防止模型无限检索文献。
    """

    def __init__(self, llm_client, tool_executor=None, max_tool_rounds: int = 4):
        self._llm_client = llm_client
        if tool_executor is None:
            from drsr_420.tool_runner import mcp_call_tool
            tool_executor = mcp_call_tool
        self._tool_executor = tool_executor
        self._max_tool_rounds = max_tool_rounds

    def complete(self, content: str, repeat: int = 1):
        """对同一 content 连续采样 repeat 次；每次内部自动处理工具调用循环。

        Returns:
            repeat > 1 时返回 (responses, think_responses) 两个 list；
            repeat == 1 时返回 (response_str, think_str)。
        """
        responses = []
        think_responses = []
        for _ in range(max(1, repeat)):
            try:
                messages = [
                    {"role": "system", "content": pc.sampling_system_prompt},
                    {"role": "user", "content": content},
                ]
                tool_rounds = 0
                while True:
                    # 流式迭代：reasoning 与 content 按到达顺序实时打印增量，[思考]/[正文] 视觉分隔
                    stream = LineStreamPrinter()
                    shown = 0
                    think_label_printed = False
                    content_label_printed = False

                    def _on_delta(chunk):
                        nonlocal shown, think_label_printed, content_label_printed
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

                    resp = self._llm_client.chat(messages, on_delta=_on_delta)
                    stream.flush()
                    print("\n====================================================\n")
                    tool_calls = resp.get('tool_calls', []) or []
                    messages.append({
                        "role": "assistant",
                        "content": resp.get('content', ''),
                        "tool_calls": tool_calls,
                    })
                    # 如果调了 tool，执行后回传
                    if tool_calls:
                        tool_rounds += 1
                        print(f"调用了工具（第 {tool_rounds}/{self._max_tool_rounds} 轮）：", tool_calls)
                        for tc in tool_calls:
                            fn_name = tc.get('function', {}).get('name', '')
                            try:
                                args = json.loads(tc.get('function', {}).get('arguments', '{}'))
                            except (json.JSONDecodeError, TypeError):
                                args = {}
                            result = self._tool_executor(fn_name, args)
                            print_block("======工具调用结果======")
                            print_block(result)
                            print_block("======================")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.get('id', ''),
                                "content": result,
                            })
                        # 达到轮次上限：强制结束，把当前响应作为结果（避免无限检索拖死采样）
                        if tool_rounds >= self._max_tool_rounds:
                            print(f"[ToolCaller] 达到工具轮次上限（{self._max_tool_rounds}），强制返回当前响应")
                            final_text = resp.get('content', '') or resp.get('reasoning_content', '')
                            responses.append(final_text)
                            think_responses.append(resp.get('reasoning_content', ''))
                            break
                    # 如果未调用，则跳出循环
                    else:
                        # 兜底：content 为空时回退到 reasoning，避免模型只思考不输出正文时骨架丢失
                        responses.append(resp.get('content', '') or resp.get('reasoning_content', ''))
                        think_responses.append(resp.get('reasoning_content', ''))
                        break
            except Exception as e:
                print(f"API请求发生错误: {str(e)}")
                responses.append("")
                think_responses.append("")
        if repeat <= 1:
            return responses[0], think_responses[0]
        return responses, think_responses


# 兼容别名：旧模块名 drsr_420.tool_caller.ToolCaller 指向本类
ToolCaller = ToolCallerAgent
