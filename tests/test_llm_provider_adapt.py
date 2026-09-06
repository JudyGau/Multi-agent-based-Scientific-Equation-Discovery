"""LLM 客户端 provider 适配层单元测试。

覆盖新增的 per-provider 私有参数适配逻辑：
- ``LLMClient._adapt_payload``：glm / deepseek / ollama / 其他提供商对
  ``thinking`` / ``reasoning_effort`` / ``extra_body`` / ``max_completion_tokens``
  的翻译与静默忽略；
- ``LLMClient.clone_for_task``：按任务注入 ``task_params`` 声明的私有参数，
  克隆实例 kwargs 相互独立、统计计数重置；
- ``ClientFactory.from_config``：从配置 ``tasks`` 字段解析并注入 ``task_params``。

全部为纯逻辑验证，不发起真实网络请求。
"""
import unittest

import llm


def _mk_client(provider='glm', **kwargs):
    """构造指定 provider 的客户端，并注入生成参数 kwargs。"""
    client = llm.LLMClient(
        api_key='test-key',
        model=f'{provider}/test-model',
        base_url='http://test-host/v1',
        provider=provider,
    )
    client.kwargs.update(kwargs)
    return client


class GlmAdapterTest(unittest.TestCase):
    """智谱：thinking 开启时 reasoning_effort 生效；max_completion_tokens -> max_tokens。"""

    def test_glm_keeps_effort_and_enables_thinking(self):
        c = _mk_client('glm', reasoning_effort='low')
        payload = c._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertEqual(payload['reasoning_effort'], 'low')
        self.assertEqual(payload['thinking'], {'type': 'enabled'})

    def test_glm_high_effort(self):
        c = _mk_client('glm', reasoning_effort='high')
        payload = c._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertEqual(payload['reasoning_effort'], 'high')
        self.assertEqual(payload['thinking'], {'type': 'enabled'})

    def test_glm_without_effort_no_thinking(self):
        c = _mk_client('glm')
        payload = c._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertNotIn('reasoning_effort', payload)
        self.assertNotIn('thinking', payload)

    def test_glm_converts_max_completion_tokens(self):
        c = _mk_client('glm', max_completion_tokens=4096)
        payload = c._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertEqual(payload['max_tokens'], 4096)
        self.assertNotIn('max_completion_tokens', payload)

    def test_glm_removes_extra_body(self):
        c = _mk_client('glm', extra_body={'temperature': 0.1})
        payload = c._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertNotIn('extra_body', payload)


class DeepSeekAdapterTest(unittest.TestCase):
    """DeepSeek：reasoning_effort 直通；thinking 不支持移除。"""

    def test_deepseek_passes_effort_through(self):
        c = _mk_client('deepseek', reasoning_effort='high')
        payload = c._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertEqual(payload['reasoning_effort'], 'high')
        self.assertNotIn('thinking', payload)

    def test_deepseek_without_effort_removes_it(self):
        c = _mk_client('deepseek')
        payload = c._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertNotIn('reasoning_effort', payload)
        self.assertNotIn('thinking', payload)

    def test_deepseek_keeps_max_completion_tokens(self):
        c = _mk_client('deepseek', max_completion_tokens=4096)
        payload = c._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertEqual(payload['max_completion_tokens'], 4096)
        self.assertNotIn('max_tokens', payload)


class OllamaAdapterTest(unittest.TestCase):
    """Ollama：reasoning_effort -> think 布尔；两者互斥清理。"""

    def test_ollama_effort_becomes_think_bool(self):
        c = _mk_client('ollama', reasoning_effort='low')
        payload = c._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertIs(payload['think'], True)
        self.assertNotIn('reasoning_effort', payload)
        self.assertNotIn('thinking', payload)

    def test_ollama_without_effort_no_think(self):
        c = _mk_client('ollama')
        payload = c._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertNotIn('think', payload)
        self.assertNotIn('reasoning_effort', payload)
        self.assertNotIn('thinking', payload)


class GenericAdapterTest(unittest.TestCase):
    """其他提供商（siliconflow/cstcloud 等）：私有参数静默忽略，保持 OpenAI 兼容。"""

    def test_generic_ignores_private_params(self):
        c = _mk_client(
            'siliconflow',
            thinking={'type': 'enabled'},
            reasoning_effort='low',
            extra_body={'x': 1},
        )
        payload = c._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertNotIn('thinking', payload)
        self.assertNotIn('reasoning_effort', payload)
        self.assertNotIn('extra_body', payload)

    def test_generic_keeps_standard_params(self):
        c = _mk_client('cstcloud', temperature=0.3, top_p=0.9)
        payload = c._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertEqual(payload['temperature'], 0.3)
        self.assertEqual(payload['top_p'], 0.9)


class CloneForTaskTest(unittest.TestCase):
    """clone_for_task：任务参数注入、kwargs 独立、统计重置。"""

    def setUp(self):
        self.client = _mk_client('glm')
        self.client.task_params = {
            'sampling': {'reasoning_effort': 'low'},
            'analysis': {'reasoning_effort': 'high'},
        }

    def test_injects_task_params(self):
        sampling = self.client.clone_for_task('sampling')
        payload = sampling._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertEqual(payload['reasoning_effort'], 'low')

        analysis = self.client.clone_for_task('analysis')
        payload = analysis._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertEqual(payload['reasoning_effort'], 'high')

    def test_original_untouched(self):
        sampling = self.client.clone_for_task('sampling')
        sampling.kwargs['reasoning_effort'] = 'high'
        self.assertNotIn('reasoning_effort', self.client.kwargs)

    def test_clone_kwargs_independent_dict(self):
        clone = self.client.clone_for_task('sampling')
        self.assertIsNot(clone.kwargs, self.client.kwargs)
        clone.kwargs['temperature'] = 0.0
        self.assertNotEqual(self.client.kwargs.get('temperature'), 0.0)

    def test_clone_resets_stats(self):
        self.client._call_index = 5
        self.client.tokens['total'] = 100
        self.client._cum_tokens['total'] = 100
        self.client._cum_time_seconds = 9.9
        clone = self.client.clone_for_task('sampling')
        self.assertEqual(clone._call_index, 0)
        self.assertEqual(clone.tokens['total'], 0)
        self.assertEqual(clone._cum_tokens['total'], 0)
        self.assertEqual(clone._cum_time_seconds, 0.0)

    def test_unknown_task_injects_nothing(self):
        clone = self.client.clone_for_task('nonexistent')
        self.assertEqual(clone.kwargs, self.client.kwargs)

    def test_empty_task_params_injects_nothing(self):
        client = _mk_client('glm')
        client.task_params = {}
        clone = client.clone_for_task('sampling')
        self.assertEqual(clone.kwargs, client.kwargs)


class ClientFactoryTaskParamsTest(unittest.TestCase):
    """ClientFactory：从配置 tasks 字段解析 task_params。"""

    def test_from_config_injects_task_params(self):
        cfg = llm.load_llm_config('llm.config')
        client = llm.ClientFactory.from_config(cfg)
        self.assertIn('sampling', client.task_params)
        self.assertEqual(client.task_params['sampling'], {'reasoning_effort': 'low'})
        self.assertEqual(client.task_params['analysis'], {'reasoning_effort': 'high'})
        self.assertEqual(client.task_params['residual'], {'reasoning_effort': 'high'})

    def test_config_without_tasks_keeps_empty(self):
        cfg = {'host': 'https://test-host/v1', 'api_key': 'k', 'model': 'glm/test'}
        client = llm.ClientFactory.from_config(cfg)
        self.assertEqual(client.task_params, {})


if __name__ == '__main__':
    unittest.main()
