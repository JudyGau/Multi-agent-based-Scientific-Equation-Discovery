"""LLM 客户端流式输出单元测试。

覆盖 llm.LLMClient 的 SSE 流式路径：
- chat_stream 增量累积（content / reasoning_content / tool_calls）
- chat() 默认流式且返回结构与非流式一致
- stream=False 显式回退非流式
- 网关忽略 stream 参数直接返回完整 JSON 时的单块兜底

全部通过 mock _post_with_retry 完成，不发起真实网络请求。
"""
import unittest
from unittest import mock

import llm


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

    @mock.patch('llm._post_with_retry')
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

    @mock.patch('llm._post_with_retry')
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

    @mock.patch('llm._post_with_retry')
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

    @mock.patch('llm._post_with_retry')
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

    @mock.patch('llm._post_with_retry')
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

    @mock.patch('llm._post_with_retry')
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

    @mock.patch('llm._post_with_retry')
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

    @mock.patch('llm._post_with_retry')
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


if __name__ == '__main__':
    unittest.main()
