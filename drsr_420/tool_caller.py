"""MCP 工具循环：与 LLM 多轮对话，自动执行模型发起的工具调用并把结果回传给模型。

从 sampler.py 中拆分出的独立组件，可注入 mock client 与 tool_executor 单独单测。
"""
from __future__ import annotations

import json


class ToolCaller:
    """与 LLM 多轮对话：若模型返回 tool_calls 则执行工具并回传，直到模型给出最终答复。

    Args:
        llm_client: LLMClient 实例，提供 chat(messages) -> dict，
                    返回字段含 content / reasoning_content / tool_calls。
        tool_executor: 可调用对象 (name, args) -> str，默认使用 tool_runner.mcp_call_tool。
    """

    def __init__(self, llm_client, tool_executor=None):
        self._llm_client = llm_client
        if tool_executor is None:
            from drsr_420.tool_runner import mcp_call_tool
            tool_executor = mcp_call_tool
        self._tool_executor = tool_executor

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
                messages = [{"role": "user", "content": content}]
                resp = self._llm_client.chat([{"role": "user", "content": content}])
                while True:
                    tool_calls = resp.get('tool_calls', []) or []
                    messages.append({
                        "role": "assistant",
                        "content": resp.get('content', ''),
                        "tool_calls": tool_calls,
                    })
                    # 如果调了 tool，执行后回传
                    if tool_calls:
                        print("调用了工具：", tool_calls)
                        for tc in tool_calls:
                            fn_name = tc.get('function', {}).get('name', '')
                            try:
                                args = json.loads(tc.get('function', {}).get('arguments', '{}'))
                            except (json.JSONDecodeError, TypeError):
                                args = {}
                            result = self._tool_executor(fn_name, args)
                            print("======工具调用结果======")
                            print(result)
                            print("======================")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.get('id', ''),
                                "content": result,
                            })
                        resp = self._llm_client.chat(messages)
                    # 如果未调用，则跳出循环
                    else:
                        responses.append(resp.get('content', ''))
                        think_responses.append(resp.get('reasoning_content', ''))
                        break
            except Exception as e:
                print(f"API请求发生错误: {str(e)}")
                responses.append("")
                think_responses.append("")
        if repeat <= 1:
            return responses[0], think_responses[0]
        return responses, think_responses
