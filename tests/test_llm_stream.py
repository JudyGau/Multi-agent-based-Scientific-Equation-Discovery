"""LLM 客户端流式输出单元测试。

覆盖 drsr_420.llm.client.LLMClient 的 SSE 流式路径：
- chat_stream 增量累积（content / reasoning_content / tool_calls）
- chat() 默认流式且返回结构与非流式一致
- stream=False 显式回退非流式
- 网关忽略 stream 参数直接返回完整 JSON 时的单块兜底
- 网关把错误包在 HTTP 200 里（``code``/``msg``/``success`` 信封）时的异常信息

全部通过 mock 定义处的 ``_post_with_retry`` 完成，不发起真实网络请求。
（打桩必须落在定义该名字的 ``drsr_420.llm.client`` 上：``LLMClient.chat`` 在
``client`` 模块的全局命名空间里查找它，打在门面 ``drsr_420.llm`` 上等于没打。）
"""
import unittest
from unittest import mock

import requests

from drsr_420 import llm


class _FakeResponse:
    """模拟 requests.Response：只暴露流式路径用到的接口。"""

    def __init__(self, lines=None, json_data=None, content_type='text/event-stream'):
        self._lines = [l if isinstance(l, bytes) else l.encode('utf-8') for l in (lines or [])]
        self._json = json_data
        self.headers = {'Content-Type': content_type}
        self.status_code = 200

    def raise_for_status(self):
        pass

    def iter_lines(self):
        yield from self._lines

    def json(self):
        return self._json


def _sse(*data_items):
    """把 JSON 片段拼成 SSE data 行（末尾带 [DONE]）。"""
    lines = []
    for d in data_items:
        lines.append('data: ' + d)
    lines.append('data: [DONE]')
    return lines


class ChatStreamTest(unittest.TestCase):

    def setUp(self):
        self.client = llm.LLMClient(api_key='test-key', model='test/model',
                                    base_url='http://test-host/v1')

    @mock.patch('drsr_420.llm.client._post_with_retry')
    def test_stream_accumulates_content(self, mock_post):
        chunks = [
            '{"choices":[{"delta":{"content":"Hello "}}]}',
            '{"choices":[{"delta":{"content":"world"}}]}',
            '{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,'
            '"total_tokens":15,"completion_tokens_details":{"reasoning_tokens":0}}}',
        ]
        mock_post.return_value = _FakeResponse(lines=_sse(*chunks))

        out = list(self.client.chat_stream([{'role': 'user', 'content': 'hi'}]))
        # 增量块按到达顺序累积
        self.assertEqual(out[0]['content'], 'Hello ')
        self.assertEqual(out[1]['content'], 'Hello world')
        final = out[-1]
        self.assertTrue(final['final'])
        self.assertEqual(final['content'], 'Hello world')
        self.assertEqual(final['tokens']['prompt'], 10)
        self.assertEqual(final['tokens']['content'], 5)
        self.assertEqual(final['tokens']['total'], 15)
        # payload 必须携带 stream=True（位置参数），且以关键字 stream=True 请求
        args = mock_post.call_args[0]
        self.assertTrue(args[2]['stream'])
        self.assertIs(mock_post.call_args.kwargs.get('stream'), True)

    @mock.patch('drsr_420.llm.client._post_with_retry')
    def test_stream_accumulates_reasoning(self, mock_post):
        chunks = [
            '{"choices":[{"delta":{"reasoning_content":"Think"}}]}',
            '{"choices":[{"delta":{"reasoning_content":" step."}}]}',
            '{"choices":[{"delta":{"content":"Answer"}}]}',
            '{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,'
            '"total_tokens":2,"completion_tokens_details":{"reasoning_tokens":1}}}',
        ]
        mock_post.return_value = _FakeResponse(lines=_sse(*chunks))

        out = list(self.client.chat_stream([]))
        self.assertEqual(out[-1]['reasoning_content'], 'Think step.')
        self.assertEqual(out[-1]['content'], 'Answer')
        self.assertEqual(out[-1]['tokens']['reasoning'], 1)

    @mock.patch('drsr_420.llm.client._post_with_retry')
    def test_stream_accumulates_tool_calls(self, mock_post):
        # 工具调用 arguments 跨多个 chunk 增量返回
        chunks = [
            '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
            '"function":{"name":"search_paper","arguments":""}}]}}]}',
            '{"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"{\\"query\\":\\"MR\\"}"}}]}}]}',
            '{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,'
            '"total_tokens":2,"completion_tokens_details":{"reasoning_tokens":0}}}',
        ]
        mock_post.return_value = _FakeResponse(lines=_sse(*chunks))

        out = list(self.client.chat_stream([]))
        calls = out[-1]['tool_calls']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['id'], 'call_1')
        self.assertEqual(calls[0]['function']['name'], 'search_paper')
        self.assertEqual(calls[0]['function']['arguments'], '{"query":"MR"}')

    @mock.patch('drsr_420.llm.client._post_with_retry')
    def test_chat_default_streams_and_returns_full_dict(self, mock_post):
        chunks = [
            '{"choices":[{"delta":{"reasoning_content":"r"}}]}',
            '{"choices":[{"delta":{"content":"hello"}}]}',
            '{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,'
            '"total_tokens":5,"completion_tokens_details":{"reasoning_tokens":1}}}',
        ]
        mock_post.return_value = _FakeResponse(lines=_sse(*chunks))

        resp = self.client.chat([{'role': 'user', 'content': 'hi'}])
        self.assertEqual(resp['content'], 'hello')
        self.assertEqual(resp['reasoning_content'], 'r')
        self.assertEqual(resp['tool_calls'], [])
        self.assertEqual(resp['tokens']['total'], 5)

    @mock.patch('drsr_420.llm.client._post_with_retry')
    def test_chat_on_delta_receives_accumulated_chunks(self, mock_post):
        chunks = [
            '{"choices":[{"delta":{"reasoning_content":"r1"}}]}',
            '{"choices":[{"delta":{"content":"hi"}}]}',
            '{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,'
            '"total_tokens":2,"completion_tokens_details":{"reasoning_tokens":1}}}',
        ]
        mock_post.return_value = _FakeResponse(lines=_sse(*chunks))

        seen = []
        resp = self.client.chat([{'role': 'user', 'content': 'hi'}],
                                on_delta=lambda c: seen.append(c))
        # 回调收到非 final 增量块（累积值），且不影响返回契约
        # 注意：usage 块也会触发一次回调（delta 为空、累积值不变）
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[0]['reasoning_content'], 'r1')
        self.assertEqual(seen[1]['content'], 'hi')
        self.assertFalse(any(c.get('final') for c in seen))
        self.assertEqual(resp['content'], 'hi')
        self.assertEqual(resp['reasoning_content'], 'r1')

    @mock.patch('drsr_420.llm.client._post_with_retry')
    def test_chat_on_delta_exception_ignored(self, mock_post):
        chunks = [
            '{"choices":[{"delta":{"content":"hi"}}]}',
            '{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,'
            '"total_tokens":2,"completion_tokens_details":{"reasoning_tokens":0}}}',
        ]
        mock_post.return_value = _FakeResponse(lines=_sse(*chunks))

        def _bad_delta(_chunk):
            raise RuntimeError('boom')

        resp = self.client.chat([{'role': 'user', 'content': 'hi'}], on_delta=_bad_delta)
        self.assertEqual(resp['content'], 'hi')

    @mock.patch('drsr_420.llm.client._post_with_retry')
    def test_chat_non_stream_fallback(self, mock_post):
        # config 显式设置 stream=False 时回退非流式路径
        self.client.kwargs['stream'] = False
        json_data = {
            'choices': [{'message': {'content': 'nope', 'tool_calls': []}}],
            'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3},
        }
        mock_post.return_value = _FakeResponse(lines=[], json_data=json_data,
                                               content_type='application/json')

        resp = self.client.chat([{'role': 'user', 'content': 'hi'}])
        self.assertEqual(resp['content'], 'nope')
        # 非流式路径：只传 3 个位置参数，stream 走默认 False；payload 内显式 stream=False
        args = mock_post.call_args[0]
        self.assertEqual(len(args), 3)
        self.assertIs(args[2]['stream'], False)

    @mock.patch('drsr_420.llm.client._post_with_retry')
    def test_stream_falls_back_to_full_json(self, mock_post):
        # 个别网关忽略 stream 参数、直接返回完整 JSON：按单块处理
        json_data = {
            'choices': [{'message': {'content': 'full', 'reasoning_content': 'think',
                                     'tool_calls': []}}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        }
        mock_post.return_value = _FakeResponse(lines=[], json_data=json_data,
                                               content_type='application/json')

        out = list(self.client.chat_stream([]))
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0]['final'])
        self.assertEqual(out[0]['content'], 'full')


class GatewayErrorEnvelopeTest(unittest.TestCase):
    """网关把错误包在 **HTTP 200** 里时，异常信息必须带上它那句原话。

    真实案例（本轮实测撞到）：档案里的端点路径写成了 ``/api/ants/v1``，智谱网关返回

        HTTP 200 + {"code":500,"msg":"404 NOT_FOUND","success":false}

    旧实现只在 print 里留下原始 JSON，异常永远是 ``API response missing choices``——只看
    异常的人会去怀疑模型名或密钥，而真正该改的是路径。所以网关的 ``msg`` 必须进异常。
    """

    def _client(self):
        return llm.LLMClient(api_key='k', model='m', base_url='http://h/v1')

    def _chat_with(self, payload):
        resp = _FakeResponse(json_data=payload, content_type='application/json')
        with mock.patch('drsr_420.llm.client._post_with_retry', return_value=resp), \
             mock.patch('builtins.print'):
            return self._client().chat([{'role': 'user', 'content': 'hi'}])

    def test_gateway_code_msg_envelope_reaches_the_exception(self):
        with self.assertRaises(requests.exceptions.HTTPError) as ctx:
            self._chat_with({"code": 500, "msg": "404 NOT_FOUND", "success": False})
        message = str(ctx.exception)
        self.assertIn("404 NOT_FOUND", message)
        self.assertIn("code=500", message)

    def test_error_key_envelope_reaches_the_exception(self):
        with self.assertRaises(requests.exceptions.HTTPError) as ctx:
            self._chat_with({"error": {"message": "invalid api key",
                                       "type": "auth_error", "code": "401"}})
        self.assertIn("invalid api key", str(ctx.exception))

    def test_message_key_envelope_reaches_the_exception(self):
        with self.assertRaises(requests.exceptions.HTTPError) as ctx:
            self._chat_with({"message": "model not found"})
        self.assertIn("model not found", str(ctx.exception))

    def test_unrecognised_body_keeps_the_plain_message(self):
        """认不出的响应体不该编造原因——保持原样，别在异常里塞空括号。"""
        with self.assertRaises(requests.exceptions.HTTPError) as ctx:
            self._chat_with({"foo": 1})
        self.assertEqual(str(ctx.exception), "API response missing choices")

    def test_helper_extracts_nothing_from_non_dict(self):
        from drsr_420.llm.client import gateway_error_detail

        for value in (None, "text", [], 3):
            with self.subTest(value=value):
                self.assertEqual(gateway_error_detail(value), "")


if __name__ == '__main__':
    unittest.main()
