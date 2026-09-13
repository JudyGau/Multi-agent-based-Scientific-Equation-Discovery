"""自定义提供商护栏：不在内置表里的 OpenAI 兼容端点必须"零代码改动"接得进来。

为什么需要这组测试
==================
``ClientFactory._PROVIDER_SPECS`` 原先是一张封闭表，provider 段不在表里就直接抛
"不支持的提供商"。于是"给某个角色换一个自建/校内/第三方网关"这件事**只能改代码**，
而角色注册表（``config/agents.config.json``）里恰好有一行 ``roles.summary.config``
是专门用来做这件事的——表达力缺口就卡在这一层：

* 端点属于 Q1（连接谁 / 用哪把钥匙），按设计本就该在档案文件里；
* 请求体差异（reasoning_effort 怎么拼）属于"发给谁时字段长什么样"，
  由档案的 ``dialect`` 字段声明，默认 ``openai``：不认识的一律不发明。

所以这里守四条：

1. 未知 provider 段 + ``base_url`` ⇒ 照常构造（走 :class:`OpenAICompatClient`）；
2. 未知 provider 段 + 无 ``base_url`` ⇒ 报错必须**教人怎么接进来**，而不是只列内置表；
3. 密钥解析要有确定规则：``api_key`` > ``api_key_env`` > 按 provider 段派生的
   环境变量（``ustc`` -> ``USTC_API_KEY``），且报错信息里必须是**具体变量名**；
4. 端点键名与写法统一：键名是 ``base_url``（``host`` 已下线，出现即报错并给出改名
   提示），取值是**完整 URL**（``http(s)://…``，不再替用户补 scheme）——静默容忍
   两种写法会让请求打到别处（内置提供商有默认端点），或报出与写法无关的底层错误；
5. 内置提供商的行为逐位不变（这轮只加了一条新路径，没动老路径）。

全部断言不依赖本机凭据，也不发起网络请求。
"""
from __future__ import annotations

import json
import os
import pathlib
import unittest
from unittest import mock

from drsr_420.llm import factory as factory_mod
from drsr_420.llm.adapt import DIALECTS
from drsr_420.llm.client import LLMClient
from drsr_420.llm.factory import ClientFactory
from drsr_420.llm.providers import OpenAICompatClient, ZhipuClient

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / "config"


def _cfg(**overrides) -> dict:
    """一份最小的自定义提供商档案（provider 段 'ustc' 不在内置表里）。"""
    base = {
        "model": "ustc/deepseek-v4-flash",
        "base_url": "https://api.llm.ustc.edu.cn/v1",
        "api_key": "test-key-not-secret",
    }
    base.update(overrides)
    return base


class CustomProviderTest(unittest.TestCase):
    """provider 段是代码没见过的名字时，档案自带 base_url 即可。"""

    def test_unknown_provider_with_base_url_is_accepted(self):
        client = ClientFactory.from_config(_cfg())
        self.assertIsInstance(client, OpenAICompatClient)
        # provider 段原样保留：日志与快照里要能看出"连的到底是哪家"
        self.assertEqual(client.provider, "ustc")
        self.assertEqual(client._provider_name(), "ustc")
        self.assertEqual(client.model, "deepseek-v4-flash")
        self.assertEqual(client.base_url, "https://api.llm.ustc.edu.cn/v1")

    def test_custom_provider_defaults_to_plain_openai_dialect(self):
        """默认方言 openai：角色参数里的 reasoning_effort 不发出去（避免 400）。"""
        client = ClientFactory.from_config(_cfg())
        self.assertEqual(client.dialect, "openai")
        client.kwargs["reasoning_effort"] = "high"
        payload = client._build_payload([{"role": "user", "content": "hi"}])
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("thinking", payload)

    def test_deprecated_host_key_is_rejected_with_a_rename_hint(self):
        """``host`` 已下线：静默忽略它会让请求打到内置默认端点，而不是配置里那个。"""
        with self.assertRaises(ValueError) as ctx:
            ClientFactory.from_config(
                {"model": "ustc/x", "host": "api.llm.ustc.edu.cn/v1", "api_key": "k"})
        message = str(ctx.exception)
        self.assertIn("host", message)
        self.assertIn("base_url", message)          # 报错要自带改名方法
        self.assertIn("api.llm.ustc.edu.cn/v1", message)

    def test_no_base_url_error_teaches_how_to_add_one(self):
        """报错必须可操作：服务器里爬不到"原来还能自定义提供商"。"""
        with self.assertRaises(ValueError) as ctx:
            ClientFactory.from_config(
                {"model": "ustc/deepseek-v4-flash", "api_key": "k"})
        message = str(ctx.exception)
        self.assertIn("ustc", message)
        self.assertIn("base_url", message)
        self.assertIn("dialect", message)
        for builtin in ClientFactory._PROVIDER_SPECS:
            with self.subTest(builtin=builtin):
                self.assertIn(builtin, message,
                              "内置提供商列表应由 _PROVIDER_SPECS 生成，不该手写后漂移")

    def test_alias_normalisation_still_wins(self):
        """别名（zhipu -> glm）必须照旧归一到内置表，不能误判成自定义提供商。"""
        client = ClientFactory.from_config(
            {"model": "zhipu/glm-5.3-flash", "api_key": "k"})
        self.assertNotIsInstance(client, OpenAICompatClient)
        self.assertEqual(client.provider, "glm")
        self.assertEqual(client.dialect, "glm")


class CustomProviderApiKeyTest(unittest.TestCase):
    """密钥解析规则：api_key 字段 > api_key_env > 派生环境变量。"""

    def test_required_key_error_names_a_concrete_env_var(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                ClientFactory.from_config(
                    {"model": "ustc/x", "base_url": "https://h/v1"})
        message = str(ctx.exception)
        self.assertIn("USTC_API_KEY", message)     # 'ustc' 派生，而不是写死
        self.assertIn("环境变量", message)
        self.assertNotIn("None", message)          # 曾经可能拼出 "环境变量 None"

    def test_env_var_is_used_when_field_is_empty(self):
        with mock.patch.dict(os.environ, {"USTC_API_KEY": "from-env"}, clear=True):
            client = ClientFactory.from_config(
                {"model": "ustc/x", "base_url": "https://h/v1", "api_key": ""})
        self.assertEqual(client.api_key, "from-env")

    def test_api_key_env_field_overrides_derived_name(self):
        with mock.patch.dict(os.environ, {"MY_GATEWAY_TOKEN": "custom-env"},
                             clear=True):
            client = ClientFactory.from_config(
                {"model": "ustc/x", "base_url": "https://h/v1",
                 "api_key_env": "MY_GATEWAY_TOKEN"})
        self.assertEqual(client.api_key, "custom-env")

    def test_api_key_env_field_is_named_in_the_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                ClientFactory.from_config(
                    {"model": "ustc/x", "base_url": "https://h/v1",
                     "api_key_env": "MY_GATEWAY_TOKEN"})
        self.assertIn("MY_GATEWAY_TOKEN", str(ctx.exception))

    def test_api_key_required_false_allows_keyless_local_endpoint(self):
        """本地免鉴权服务（vLLM 默认）不该被迫编一个假 key。"""
        client = ClientFactory.from_config(
            {"model": "local/qwen3", "base_url": "http://localhost:8000/v1",
             "api_key_required": False})
        self.assertEqual(client.api_key, "")
        self.assertEqual(client.base_url, "http://localhost:8000/v1")


class CustomProviderDialectTest(unittest.TestCase):
    """dialect 字段让自定义提供商复用某个已知家族的请求体分支。"""

    def test_declared_dialect_reuses_a_known_branch(self):
        client = ClientFactory.from_config(_cfg(dialect="deepseek"))
        self.assertEqual(client.dialect, "deepseek")
        client.kwargs["reasoning_effort"] = "high"
        payload = client._build_payload([{"role": "user", "content": "hi"}])
        self.assertEqual(payload["reasoning_effort"], "high")

    def test_glm_dialect_translates_max_completion_tokens(self):
        client = ClientFactory.from_config(_cfg(dialect="glm", max_completion_tokens=4096))
        payload = client._build_payload([{"role": "user", "content": "hi"}])
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertNotIn("max_completion_tokens", payload)

    def test_misspelled_dialect_is_rejected(self):
        """拼错必须报错：静默降级成 openai 会变成"配了却没生效"。"""
        with self.assertRaises(ValueError) as ctx:
            ClientFactory.from_config(_cfg(dialect="deepsek"))
        message = str(ctx.exception)
        self.assertIn("deepsek", message)
        for dialect in DIALECTS:
            with self.subTest(dialect=dialect):
                self.assertIn(dialect, message)

    def test_builtin_providers_keep_their_implicit_dialect(self):
        """内置提供商的方言仍按规范名推断，未声明的档案行为逐位不变。"""
        cases = {
            "glm/x": "glm", "deepseek/x": "deepseek", "ollama/x": "ollama",
            "siliconflow/x": "openai", "cstcloud/x": "openai",
        }
        for model, expected in cases.items():
            with self.subTest(model=model):
                cfg = {"model": model, "api_key": "k"}
                if expected == "ollama":
                    cfg.pop("api_key")
                self.assertEqual(
                    ClientFactory.from_config(cfg).dialect, expected)


class BaseUrlNamingTest(unittest.TestCase):
    """端点键名与写法都要统一：键名是 ``base_url``（``host`` 已下线），
    取值是**完整 URL**（带 ``http(s)://`` 与路径），不接受裸主机域名。

    统一的是**同一个字段的两种拼写**（不是两个字段），所以"两个都写"也算残留配置，
    一样报错——否则迁移会停在半路，而 `host` 从此没人再看一眼。
    """

    def test_complete_url_is_accepted_as_is(self):
        client = ClientFactory.from_config(
            {"model": "ustc/x", "base_url": "https://api.llm.ustc.edu.cn/v1",
             "api_key": "k"})
        self.assertEqual(client.base_url, "https://api.llm.ustc.edu.cn/v1")

    def test_bare_hostname_is_rejected(self):
        """曾经会替用户补 https://，于是"裸主机域名"与完整 URL 两种写法长期并存。"""
        with self.assertRaises(ValueError) as ctx:
            ClientFactory.from_config(
                {"model": "deepseek/x", "base_url": "api.deepseek.com", "api_key": "k"})
        message = str(ctx.exception)
        self.assertIn("api.deepseek.com", message)
        self.assertIn("https://api.deepseek.com/v1", message)   # 报错给出该写成什么样

    def test_whitespace_is_stripped(self):
        client = ClientFactory.from_config(
            {"model": "glm/x", "base_url": "  https://open.bigmodel.cn/api/paas/v4  ",
             "api_key": "k"})
        self.assertEqual(client.base_url, "https://open.bigmodel.cn/api/paas/v4")

    def test_existing_http_scheme_is_preserved(self):
        client = ClientFactory.from_config(
            {"model": "local/x", "base_url": "http://localhost:8000/v1",
             "api_key_required": False})
        self.assertEqual(client.base_url, "http://localhost:8000/v1")

    def test_empty_base_url_falls_back_to_the_builtin_endpoint(self):
        """空串是"用内置默认端点"的老写法，不能被当成非法 URL 拦掉。"""
        client = ClientFactory.from_config({"model": "glm/x", "api_key": "k",
                                            "base_url": ""})
        self.assertEqual(client.base_url, "https://open.bigmodel.cn/api/paas/v4")

    def test_client_constructor_is_the_backstop_for_every_channel(self):
        """配置之外还有环境变量兜底通道（如 ``ZHIPU_API_BASE``），同样不许裸主机。"""
        with mock.patch.dict(os.environ, {"ZHIPU_API_BASE": "open.bigmodel.cn"}):
            with self.assertRaises(ValueError):
                ZhipuClient(api_key="k", model="glm-5.3-flash", base_url=None)

    def test_direct_client_construction_rejects_bare_hostname(self):
        with self.assertRaises(ValueError):
            LLMClient(api_key="k", model="m", base_url="api.example.com")

    def test_host_is_rejected_even_when_base_url_is_also_present(self):
        with self.assertRaises(ValueError) as ctx:
            ClientFactory.from_config(
                {"model": "glm/x", "base_url": "https://a/v1", "host": "https://b/v1",
                 "api_key": "k"})
        self.assertIn("base_url", str(ctx.exception))

    def test_normalize_does_not_mutate_the_input_dict(self):
        original = {"model": "glm/x", "base_url": "https://api.deepseek.com/v1"}
        snapshot = dict(original)
        factory_mod.normalize_llm_config(original)
        self.assertEqual(original, snapshot)


class ShippedCustomProviderTest(unittest.TestCase):
    """随仓库入库的注册表把 ``summary`` 绑到自定义提供商——这是该功能的真实消费者。

    刻意**不写死厂商名**：从注册表读出 summary 实际用的档案，再断言它的 provider 段
    不在内置表里。这样"把 summary 换回内置提供商"会立刻被这条测试拦下——
    那是设计变更，应当显式改这里，而不是悄悄发生。
    """

    def _summary_profile_config(self) -> dict:
        from drsr_420.llm import roles as roles_mod

        profile = roles_mod.RoleRegistry.load().entry("summary").profile
        self.assertIsNotNone(profile, "summary 应当单独绑定档案（见注册表）")
        template = _CONFIG_DIR / f"{profile}.config.example"
        self.assertTrue(template.is_file(),
                        f"summary 绑定的档案 {profile} 缺少可入库的模板 {template.name}，"
                        f"新克隆的仓库无从建立它")
        return json.loads(template.read_text(encoding="utf-8"))

    def test_summary_role_uses_a_custom_provider(self):
        config = self._summary_profile_config()
        provider, _model = factory_mod.parse_provider_model(config["model"])
        self.assertNotIn(provider, ClientFactory._PROVIDER_SPECS,
                         "summary 这个角色刻意用自定义提供商（校内网关）演示零代码接入；"
                         "若确实要换回内置提供商，请连同本测试一起修改")
        self.assertIn("base_url", config,
                      "自定义提供商的端点在档案里（Q1），不能靠代码表兜底")

    def test_summary_template_builds_a_working_client(self):
        config = dict(self._summary_profile_config())
        config["api_key"] = "placeholder-not-a-secret"
        client = ClientFactory.from_config(config)
        self.assertIsInstance(client, OpenAICompatClient)
        self.assertTrue(client.base_url.startswith("http"))
        self.assertEqual(client._provider_name(),
                         factory_mod.parse_provider_model(config["model"])[0])

    def test_explain_role_binds_a_shipped_profile_and_builds(self):
        """explain 走内置提供商（官方端点），与 summary 的自定义路线各留一条。"""
        from drsr_420.llm import roles as roles_mod

        profile = roles_mod.RoleRegistry.load().entry("explain").profile
        self.assertIsNotNone(profile, "explain 应当单独绑定档案（见注册表）")
        template = _CONFIG_DIR / f"{profile}.config.example"
        self.assertTrue(template.is_file(), f"缺少模板 {template.name}")
        config = json.loads(template.read_text(encoding="utf-8"))
        provider, _ = factory_mod.parse_provider_model(config["model"])
        self.assertIn(provider, ClientFactory._PROVIDER_SPECS)
        config["api_key"] = "placeholder-not-a-secret"
        self.assertIsNotNone(ClientFactory.from_config(config))

    def test_explain_and_summary_do_not_share_a_profile(self):
        """两个角色单独绑定档案的意义就在于此——否则等于没绑。"""
        from drsr_420.llm import roles as roles_mod

        registry = roles_mod.RoleRegistry.load()
        self.assertNotEqual(registry.entry("explain").profile,
                            registry.entry("summary").profile)


if __name__ == "__main__":
    unittest.main()
