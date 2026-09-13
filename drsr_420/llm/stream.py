"""SSE 流式响应的增量拼装：把逐块 delta 还原成完整 content / tool_calls。

为什么单独成文件
================
这里是**纯函数**（无实例状态、无网络），与 ``client.py`` 的"怎么发请求、怎么记账"
属于两种变化原因：本模块的改动驱动力是**各家 SSE 的 delta 形状差异**（思考内容有的叫
``reasoning_content``、``tool_calls`` 的 name 与 arguments 还是分片下发的），而传输层
关心的是重试、超时与统计。分开之后 client.py 也不必再逼近文件长度预算。
"""
from __future__ import annotations

from typing import Dict, List


def accumulate_stream_delta(acc_content: List[str], acc_reasoning: List[str],
                            acc_tool_calls: Dict[int, dict], delta: dict) -> None:
    """把单个 SSE chunk 的 delta 累积到缓冲区。

    Args:
        acc_content: 正文增量片段（按到达顺序 append）。
        acc_reasoning: 思考过程增量片段。
        acc_tool_calls: ``{index: {'id', 'name', 'arguments': [...]}}``，按分片累积。
        delta: 单个 chunk 里的 ``choices[0].delta``；缺字段一律视为没有。
    """
    c = delta.get('content')
    if c:
        acc_content.append(c)
    r = delta.get('reasoning_content')
    if r:
        acc_reasoning.append(r)
    for tc in delta.get('tool_calls') or []:
        idx = tc.get('index', 0)
        slot = acc_tool_calls.setdefault(idx, {'id': '', 'name': '', 'arguments': []})
        if tc.get('id'):
            slot['id'] = tc['id']
        fn = tc.get('function') or {}
        if fn.get('name'):
            slot['name'] = fn['name']
        if fn.get('arguments'):
            slot['arguments'].append(fn['arguments'])


def assemble_tool_calls(acc_tool_calls: Dict[int, dict]) -> list:
    """把累积的 tool_calls 增量拼成 OpenAI 兼容的完整 tool_calls 列表。"""
    calls = []
    for idx in sorted(acc_tool_calls):
        slot = acc_tool_calls[idx]
        calls.append({
            "id": slot['id'],
            "type": "function",
            "function": {
                "name": slot['name'],
                "arguments": ''.join(slot['arguments']),
            },
        })
    return calls
