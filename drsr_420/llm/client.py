"""LLM 客户端：请求发送、指数退避重试、流式解析、token 记账与参数适配。

提供商子类见 ``providers.py``；实例构造与配置归一化见 ``factory.py``；
**请求体方言适配**（哪个提供商把 `思考强度` 拼成什么字段）见 ``adapt.py``。
"""
import copy
import json
import os
import threading
import time
from typing import Dict, List, Tuple

import requests

from drsr_420.core.llm_stats import _accumulate_global_stats
from drsr_420.llm.adapt import adapt_payload
from drsr_420.llm.tools_schema import tools


# 大模型请求重试：网络异常与 429/5xx 指数退避重试
LLM_REQUEST_MAX_RETRIES = 4
LLM_REQUEST_BACKOFF_BASE = 2.0


def _post_with_retry(url, headers, payload,
                     max_retries=LLM_REQUEST_MAX_RETRIES,
                     backoff_base=LLM_REQUEST_BACKOFF_BASE,
                     timeout=(10, 3600),
                     stream=False):
    """带指数退避的 POST 请求：网络异常与 429/5xx 自动重试。"""
    attempt = 0
    while True:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout, stream=stream)
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = None
                try:
                    retry_after = float(resp.headers.get('Retry-After', ''))
                except (TypeError, ValueError):
                    retry_after = None
                wait = retry_after if (retry_after and retry_after > 0) else backoff_base * (2 ** attempt)
                attempt += 1
                if attempt > max_retries:
                    resp.raise_for_status()
                    return resp
                print(f"[LLM] HTTP {resp.status_code}，{wait:.1f}s 后重试（{attempt}/{max_retries}）")
                if stream:
                    # stream=True 时响应体尚未消费：不关闭就 continue 会每次重试
                    # 泄漏一个连接池中的连接直到进程结束
                    try:
                        resp.close()
                    except Exception:
                        pass
                time.sleep(wait)
                continue
            return resp
        except requests.exceptions.RequestException as e:
            attempt += 1
            if attempt > max_retries:
                raise
            wait = backoff_base * (2 ** (attempt - 1))
            print(f"[LLM] 请求异常: {e}，{wait:.1f}s 后重试（{attempt}/{max_retries}）")
            time.sleep(wait)



class LLMClient:
    """OpenAI Chat Completions 兼容的通用 LLM 客户端。

    - ``kwargs``：生成参数（temperature/top_p/max_tokens 等），
      由构造方（ClientFactory）从配置统一注入；调用方按用途浅拷贝后覆盖。
    - 每次 ``chat`` 只透传白名单内的生成参数，并按提供商做差异适配（见 _adapt_payload）。
    """

    # 构建 payload 时允许透传的 OpenAI 兼容生成参数（其余字段一律忽略，避免提供商拒绝未知参数）
    ALLOWED_GEN_KEYS = {
        'max_tokens', 'max_completion_tokens', 'temperature', 'top_p', 'top_k', 'n',
        'stream', 'presence_penalty', 'frequency_penalty', 'stop', 'logprobs',
        'options', 'extra_body', 'thinking', 'reasoning_effort',
    }

    def __init__(self, api_key: str, model: str, base_url: str, provider: str | None = None):
        """
        初始化 LLM 客户端。

        :param api_key: API 密钥
        :param model: 模型名称
        :param base_url: API 的基础 URL
        :param provider: 提供商标识（如 'glm'）；缺省时由 base_url 推断
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.provider = (provider or '').lower()
        # 请求体方言（见 adapt.py）：由 ClientFactory 按档案的 ``dialect`` 字段设置。
        # 留空表示"按 provider 名推断"——直接构造客户端的调用方（含单测）行为不变。
        self.dialect = ''
        # 生成参数（temperature/top_p/max_tokens 等）
        self.kwargs = {
            'temperature': 0.5,
            'top_p': 0.5,
            'n': 1,
            'stream': True,
        }
        # token 统计：实例级独立字典（原为类变量，会被多实例共享污染，故改为实例属性）
        self.tokens = {'prompt': 0, 'content': 0, 'reasoning': 0, 'total': 0}
        # 实例级别的累计统计与耗时（无需显式 reset；通常每个实验构造一个 client）
        self._call_index = 0
        self._cum_tokens = {
            'prompt': 0,
            'thinking': 0,
            'content': 0,
            'total': 0,
        }
        self._cum_time_seconds: float = 0.0
        # 任务级私有参数（如思考强度），由 ClientFactory 从配置的 'tasks' 字段注入；
        # clone_for_task() 按任务克隆客户端时据此注入对应参数
        self.task_params: Dict[str, dict] = {}

    def clone_for_task(self, task_name: str):
        """按任务克隆客户端并注入该任务声明的私有参数（如思考强度）。

        任务参数来自配置文件的 ``tasks`` 字段（如 ``tasks: {"sampling": {"reasoning_effort": "low"}}``），
        由 ClientFactory 解析后存入 ``task_params``。克隆后的实例 kwargs 相互独立，
        避免不同任务（采样/经验/残差/分析/解释）互相覆盖生成参数。
        """
        new_client = copy.copy(self)
        # 字典/列表值必须深拷贝：``extra_body``、``thinking``、``stop`` 是嵌套结构，
        # 浅拷贝会让"克隆"与原件共享同一个内层 dict——改一个克隆就改掉全部克隆，
        # 正是本方法要根除的那类串味（历史上三个用途的参数互相覆盖过）。
        new_client.kwargs = {
            k: (copy.deepcopy(v) if isinstance(v, (dict, list)) else v)
            for k, v in self.kwargs.items()
        }
        for k, v in (self.task_params.get(task_name) or {}).items():
            if v is not None:
                new_client.kwargs[k] = v
        # 重置独立实例的累计统计，避免计数重复累加
        new_client._call_index = 0
        new_client.tokens = {'prompt': 0, 'content': 0, 'reasoning': 0, 'total': 0}
        new_client._cum_tokens = {
            'prompt': 0, 'thinking': 0, 'content': 0, 'total': 0,
        }
        new_client._cum_time_seconds = 0.0
        return new_client

    def _provider_name(self) -> str:
        if self.provider:
            return self.provider
        try:
            url = (self.base_url or '').lower()
            if 'deepseek' in url:
                return 'deepseek'
            if 'siliconflow' in url or 'siliconflow.cn' in url:
                return 'siliconflow'
            if 'deepinfra' in url:
                return 'deepinfra'
            if 'bltcy' in url or 'blt' in url:
                return 'blt'
            if 'ollama' in url or 'localhost' in url:
                return 'ollama'
            if 'cstcloud' in url or 'uni-api.cstcloud.cn' in url:
                return 'cstcloud'
            if 'bigmodel' in url or 'zhipu' in url or 'glm' in url:
                return 'glm'
        except Exception:
            pass
        return 'llm'

    def _build_payload(self, messages: List[Dict[str, str]]) -> dict:
        """构造请求体：固定字段 + 白名单生成参数 + 提供商差异适配。"""
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        # 仅透传白名单内的生成参数，避免提供商拒绝未知字段
        for k, v in self.kwargs.items():
            if k in self.ALLOWED_GEN_KEYS:
                payload[k] = v
        # 输出 token 上限保护（部分模型上限较低，统一 clamp 到 65536）
        for key in ('max_tokens', 'max_completion_tokens'):
            if isinstance(payload.get(key), int) and payload[key] > 65536:
                payload[key] = 65536
        # extra_body 的语义（沿用 openai SDK）是"把这些键**并入**请求体"，而不是一个
        # 发给服务的字段。它必须最后并入——那是调用方对具体端点能力的显式声明
        # （如自建 vLLM 的 chat_template_kwargs），方言规则不应把它删掉。
        extra = payload.pop('extra_body', None)
        self._adapt_payload(payload)
        if isinstance(extra, dict):
            payload.update(extra)
        return payload

    def _adapt_payload(self, payload: dict) -> None:
        """按方言修正请求体（方言规则集中在 :mod:`drsr_420.llm.adapt`）。

        思考强度（reasoning_effort/thinking）是跨提供商的语义参数，由角色解析按任务
        注入 kwargs，此处翻译成该方言合法的请求字段。方言来自档案的 ``dialect`` 字段
        （由 ClientFactory 设置）；未设置时按 provider 名推断——因此直接构造客户端的
        调用方（含单测）行为不变。
        """
        adapt_payload(payload, self.dialect or self._provider_name())

    def chat(self, messages: List[Dict[str, str]], on_delta=None) -> dict:
        """与 LLM 对话（默认流式）。

        - 默认以 SSE 流式方式请求（stream=True），逐块累积后返回完整结果 dict，
          返回结构与原先非流式调用完全一致，调用方无需改动；
        - ``on_delta``：可选回调，每收到一个流式增量块时调用 ``on_delta(chunk)``，
          chunk 为截至当前的累积值 ``{'content', 'reasoning_content', 'tool_calls'}``，
          便于调用方实时显示输出（回调异常被忽略，不影响主流程）；
        - 若 config 显式设置 stream=False，则回退到非流式路径（_chat_non_stream），
          此时 on_delta 不会触发；
        - 需要逐段输出（如控制台实时打印）的调用方可直接迭代 chat_stream()。
        """
        if self.kwargs.get('stream', True) in (False, 0, 'false', 'False'):
            return self._chat_non_stream(messages)
        full_result = None
        for chunk in self.chat_stream(messages):
            if on_delta is not None and not chunk.get('final'):
                try:
                    on_delta(chunk)
                except Exception:
                    pass
            if chunk.get('final'):
                full_result = {k: v for k, v in chunk.items() if k != 'final'}
        if full_result is None:
            raise RuntimeError("流式调用未返回任何结果")
        return full_result

    def chat_stream(self, messages: List[Dict[str, str]]):
        """SSE 流式调用生成器。

        Yields:
            增量块：{'content', 'reasoning_content', 'tool_calls'}，为截至当前已累积的结果；
            最后一块额外携带 'final': True 与 'tokens'，结构与 chat() 的返回一致。
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        request_url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = self._build_payload(messages)
        payload['stream'] = True

        start_time = time.time()
        try:
            response = _post_with_retry(request_url, headers, payload, stream=True)
            response.raise_for_status()

            acc_content: List[str] = []
            acc_reasoning: List[str] = []
            acc_tool_calls: Dict[int, dict] = {}
            usage: dict = {}

            if 'text/event-stream' not in (response.headers.get('Content-Type') or ''):
                # 个别网关忽略 stream 参数、直接返回完整 JSON：按单块处理
                full = self._finalize_response_data(response.json(), start_time)
                yield {**full, 'final': True}
                return

            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8', errors='replace').strip()
                if not line.startswith('data:'):
                    continue
                data = line[len('data:'):].strip()
                if data == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk, dict) and chunk.get('usage'):
                    usage = chunk['usage']
                if isinstance(chunk, dict) and chunk.get('choices'):
                    delta = chunk['choices'][0].get('delta') or {}
                else:
                    delta = {}
                self._accumulate_stream_delta(acc_content, acc_reasoning, acc_tool_calls, delta)
                yield {
                    'content': ''.join(acc_content),
                    'reasoning_content': ''.join(acc_reasoning),
                    'tool_calls': self._assemble_tool_calls(acc_tool_calls),
                }

            full = self._finalize_response(
                ''.join(acc_content), ''.join(acc_reasoning),
                self._assemble_tool_calls(acc_tool_calls), usage, start_time)
            yield {**full, 'final': True}

        except requests.exceptions.RequestException as e:
            print(f"通过 requests 调用 API 时出错: {e}")
            if e.response is not None:
                try:
                    print("错误详情(JSON):", e.response.json())
                except ValueError:
                    try:
                        print("错误详情(TEXT):", e.response.text[:500])
                    except Exception:
                        pass
            raise

        except Exception as e:
            print(e)
            raise

    def _chat_non_stream(self, messages: List[Dict[str, str]]) -> dict:
        """非流式路径：config 显式设置 stream=False 时使用（行为与原实现一致）。"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        request_url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = self._build_payload(messages)

        start_time = time.time()
        try:
            response = _post_with_retry(request_url, headers, payload)
            response.raise_for_status()
            try:
                response_data = response.json()
            except ValueError:
                print("API 响应无法解析为 JSON，原始文本预览:", response.text[:500])
                raise
            return self._finalize_response_data(response_data, start_time)

        except requests.exceptions.RequestException as e:
            print(f"通过 requests 调用 API 时出错: {e}")
            if e.response is not None:
                try:
                    print("错误详情(JSON):", e.response.json())
                except ValueError:
                    try:
                        print("错误详情(TEXT):", e.response.text[:500])
                    except Exception:
                        pass
            raise

        except Exception as e:
            print(e)
            raise

    def _finalize_response_data(self, response_data: dict, start_time: float) -> dict:
        """从完整 JSON 响应构造返回 dict（非流式与流式兜底共用）。"""
        # OpenAI 兼容接口错误格式：{"error": {...}}
        if isinstance(response_data, dict) and 'error' in response_data:
            err = response_data.get('error') or {}
            print("API 返回错误:", {
                'type': err.get('type'),
                'code': err.get('code'),
                'message': err.get('message') or err,
            })
            raise requests.exceptions.HTTPError(f"API error: {err}")

        # 保护性判断：缺少 choices 时打印提示
        if 'choices' not in response_data or not response_data['choices']:
            print("API 响应不包含 choices 字段或为空：", str(response_data)[:500])
            raise requests.exceptions.HTTPError("API response missing choices")

        message = response_data['choices'][0].get('message', {})
        content = message.get('content', '') or ''
        reasoning_content = message.get('reasoning_content', '') or ''
        tool_calls = message.get('tool_calls', [])
        usage = response_data.get('usage', {})
        return self._finalize_response(content, reasoning_content, tool_calls, usage, start_time)

    def _finalize_response(self, content: str, reasoning_content: str,
                           tool_calls: list, usage: dict, start_time: float) -> dict:
        """统计 token/耗时并构造统一返回 dict。"""
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        reasoning_tokens = 0
        if 'completion_tokens_details' in usage:
            reasoning_tokens = usage['completion_tokens_details'].get('reasoning_tokens', 0)
        # completion - reasoning 对"completion_tokens 不含 reasoning"的提供商会
        # 算出负数；统一 clamp 一次，三处累计（实例/全局/打印）共用该值。
        content_tokens = max(0, int(completion_tokens) - int(reasoning_tokens))

        self.tokens['prompt'] += prompt_tokens
        self.tokens['content'] += content_tokens
        self.tokens['reasoning'] += reasoning_tokens
        self.tokens['total'] += total_tokens

        # 更新单次实验全局统计（带锁，见 _accumulate_global_stats）
        try:
            _accumulate_global_stats(
                int(prompt_tokens), int(reasoning_tokens), content_tokens,
                int(total_tokens), time.time() - start_time)
        except Exception:
            pass

        # 实例级累计与打印
        try:
            elapsed = time.time() - start_time
            self._cum_time_seconds += float(elapsed)

            self._call_index += 1
            self._cum_tokens['prompt'] += int(prompt_tokens)
            self._cum_tokens['thinking'] += int(reasoning_tokens)
            self._cum_tokens['content'] += content_tokens
            self._cum_tokens['total'] += int(total_tokens)

            provider = self._provider_name()
            header = f"[{provider}][{self.model}] 第{self._call_index}次"
            line_cur = (
                f"本次 tokens：prompt={int(prompt_tokens)}, thinking={int(reasoning_tokens)}, "
                f"content={int(completion_tokens - reasoning_tokens)}, total={int(total_tokens)}"
            )
            line_cum = (
                f"累计 tokens：prompt={self._cum_tokens['prompt']}, thinking={self._cum_tokens['thinking']}, "
                f"content={self._cum_tokens['content']}, total={self._cum_tokens['total']}"
            )
            line_time = (
                f"本次用时：{elapsed:.2f}s，"
                f"累计用时：{self._cum_time_seconds:.2f}s"
            )
            print(header + "\n" + line_cur + "\n" + line_cum + "\n" + line_time)
        except Exception:
            pass

        return {
            "content": content,
            "reasoning_content": reasoning_content,
            "tokens": {
                "prompt": prompt_tokens,
                "content": content_tokens,
                "reasoning": reasoning_tokens,
                "total": total_tokens
            },
            "tool_calls": tool_calls,
        }

    @staticmethod
    def _accumulate_stream_delta(acc_content: List[str], acc_reasoning: List[str],
                                 acc_tool_calls: Dict[int, dict], delta: dict) -> None:
        """把单个 SSE chunk 的 delta 累积到缓冲区。"""
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

    @staticmethod
    def _assemble_tool_calls(acc_tool_calls: Dict[int, dict]) -> list:
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
