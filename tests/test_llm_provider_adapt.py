"""LLM 客户端 provider 适配层单元测试。

覆盖新增的 per-provider 私有参数适配逻辑：
- ``LLMClient._adapt_payload``：glm / deepseek / ollama / 其他提供商对
  ``thinking`` / ``reasoning_effort`` / ``extra_body`` / ``max_completion_tokens``
  的翻译与静默忽略（方言规则本身在 ``drsr_420/llm/adapt.py``）；
- ``extra_body`` 的**并入语义**：它是调用方对端点私有能力的显式声明，最后并入且
  不被方言规则删掉（自定义提供商接 vLLM 私有字段的唯一出口）；
- ``LLMClient.clone_for_task``：按任务注入 ``task_params`` 声明的私有参数，
  克隆实例 kwargs 相互独立（含嵌套结构）、统计计数重置；
- ``ClientFactory.from_config``：从配置 ``tasks`` 字段解析并注入 ``task_params``。

全部为纯逻辑验证，不发起真实网络请求。
"""
import unittest

from drsr_420 import llm


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


class ExtraBodyTest(unittest.TestCase):
    """``extra_body``：把键**并入**请求体（沿用 openai SDK 语义），且最后发言。

    这条曾经是**死配置**：工厂不注入它、``_adapt_payload`` 又把它 pop 掉，
    ``config/ollama_*.config.example`` 里那份 ``enable_thinking`` 从来没上过线。
    而它正是自定义提供商接端点私有字段（如 vLLM 的 ``chat_template_kwargs``）
    的唯一出口，所以语义必须钉死。"""

    def test_keys_are_merged_into_the_body(self):
        c = _mk_client('glm', extra_body={'chat_template_kwargs': {'thinking': True}})
        payload = c._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertEqual(payload['chat_template_kwargs'], {'thinking': True})
        self.assertNotIn('extra_body', payload)      # 包装字段本身不发出去

    def test_extra_body_has_the_last_word_over_dialect_cleanup(self):
        """openai 方言默认删 reasoning_effort，但写在 extra_body 里的是**显式声明**，
        不能被方言规则顺手删掉——那等于这个出口对最需要它的端点失效。"""
        c = _mk_client('siliconflow', extra_body={'reasoning_effort': 'high'})
        payload = c._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertEqual(payload['reasoning_effort'], 'high')

    def test_private_params_still_cleaned_when_not_in_extra_body(self):
        """对照组：同样一个 reasoning_effort，从 kwargs 进就被方言清理。"""
        c = _mk_client('siliconflow', reasoning_effort='high')
        payload = c._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertNotIn('reasoning_effort', payload)

    def test_non_dict_extra_body_is_ignored(self):
        c = _mk_client('glm', extra_body='not-a-mapping')
        payload = c._build_payload([{'role': 'user', 'content': 'hi'}])
        self.assertNotIn('extra_body', payload)
        self.assertNotIn('not-a-mapping', payload.values())


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

    def test_clone_deep_copies_nested_kwargs(self):
        """嵌套结构（extra_body/thinking）也必须隔离：浅拷贝会让"改一个克隆"
        改掉全部克隆——正是本方法要根除的那类串味。"""
        self.client.kwargs['extra_body'] = {'chat_template_kwargs': {'thinking': False}}
        clone = self.client.clone_for_task('sampling')
        clone.kwargs['extra_body']['chat_template_kwargs']['thinking'] = True
        self.assertFalse(
            self.client.kwargs['extra_body']['chat_template_kwargs']['thinking'])

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
    """ClientFactory：从配置 ``tasks`` 字段（旧格式）解析 task_params。

    角色化的参数现由 :mod:`drsr_420.llm.roles` 经 ``task_params=`` 传入（优先级更高），
    这里守的是**向后兼容**：老档案文件里的 ``tasks`` 字段仍然生效。
    """

    def test_from_config_injects_task_params(self):
        # 不读用户的真实档案（config/ 下的 *.config 含密钥且不入库，
        # 新克隆的仓库里根本不存在）——那会让本测试只在特定机器上通过。
        # 用内联构造的配置，令测试与本机凭据解耦。
        cfg = {
            'base_url': 'https://open.bigmodel.cn/api/paas/v4',
            'api_key': 'test-key-not-secret',
            'model': 'glm/glm-5.3-flash',
            'tasks': {
                'sampling': {'reasoning_effort': 'low'},
                'analysis': {'reasoning_effort': 'high'},
                'residual': {'reasoning_effort': 'high'},
            },
        }
        client = llm.ClientFactory.from_config(cfg)
        self.assertIn('sampling', client.task_params)
        self.assertEqual(client.task_params['sampling'], {'reasoning_effort': 'low'})
        self.assertEqual(client.task_params['analysis'], {'reasoning_effort': 'high'})
        self.assertEqual(client.task_params['residual'], {'reasoning_effort': 'high'})

    def test_config_without_tasks_keeps_empty(self):
        cfg = {'base_url': 'https://test-host/v1', 'api_key': 'k', 'model': 'glm/test'}
        client = llm.ClientFactory.from_config(cfg)
        self.assertEqual(client.task_params, {})


if __name__ == '__main__':
    unittest.main()
