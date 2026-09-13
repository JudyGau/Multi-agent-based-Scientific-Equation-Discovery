"""配置角色化护栏：锁死「哪个 Agent 用哪套配置」的声明方式与解析行为。

为什么需要这组测试
==================
重构前这套信息**在代码里没有任何一处声明**，只能靠 grep 反推；由此产生过两个
真实故障：``analysis/explain.py`` 硬编码了一个不存在的档案文件名，异常被
``try/except`` 吞掉后长期静默写出空的 ``explain.txt``；而 ``llm_explain.config`` /
``llm_summary.config`` 这两个"想给特定角色换模型"的档案，因为旧结构**在原理上**
无法表达该意图，一直无人读取。

所以这里守三层：

1. **不得再出现硬编码档案名**（扫描库代码里的字符串字面量）——直接堵住上述故障的
   成因，而不是只修那一个文件；
2. **解析优先级必须是文档写的那一套**（6 级，逐级构造场景验证）；
3. **模板完整性**：每个 ``*.config`` 都要有随仓库分发的 ``*.config.example``，
   否则新克隆的仓库拿不到配置起点；且每个模板都要**真能构造出客户端**——
   注册表一旦把某个角色绑到某份档案，那份档案的模板就必须是可用的，
   否则"换模型"这件事在新克隆里会以运行时报错的形式才暴露。

全部断言都刻意做到**不依赖本机是否已建好真实档案**（那属于部署状态，不是仓库状态），
因此新克隆的仓库里这组测试同样全绿。
"""
from __future__ import annotations

import ast
import contextlib
import copy
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from drsr_420.llm import factory as factory_mod
from drsr_420.llm import role_clients as role_clients_mod
from drsr_420.llm import role_diagnostics as diag_mod
from drsr_420.llm import roles as roles_mod
from drsr_420.llm.role_clients import RoleClients
from drsr_420.llm.role_diagnostics import check_roles, describe_roles

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PKG_DIR = _REPO_ROOT / "drsr_420"
_CONFIG_DIR = _REPO_ROOT / "config"

#: 一个字符串字面量若"整体就是一个档案文件名"，即视为硬编码。
#: 注意 ``".config"``（纯后缀常量）、``"agents.config.json"``（注册表，可入库）
#: 与文档字符串里的散文都不匹配——因此这条护栏不需要任何豁免名单。
_PROFILE_FILENAME_RE = re.compile(r"^[\w.\-]+\.config$")


def _string_literals(path: pathlib.Path) -> list[tuple[int, str]]:
    """收集文件中所有字符串字面量（含文档字符串）及其行号。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.append((node.lineno, node.value))
    return found


class NoHardcodedProfileFilenameTest(unittest.TestCase):
    """库代码里不得出现写死的 ``*.config`` 文件名。

    「哪个角色用哪个档案」必须经 :mod:`drsr_420.llm.roles` 解析——这样
    ``--role-config`` / 环境变量 / 注册表才管得住它。写死一个文件名就等于绕过了
    整套机制（explain 的空文件故障正是这么来的）。

    这是绊线而非沙箱：``"rag" + ".config"`` 这类拼接能绕过，但那种写法在 review
    里一眼可见。
    """

    def test_no_module_hardcodes_a_profile_filename(self):
        offenders: list[str] = []
        for path in sorted(_PKG_DIR.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for lineno, value in _string_literals(path):
                if _PROFILE_FILENAME_RE.match(value.strip()):
                    offenders.append(
                        f"{path.relative_to(_REPO_ROOT)}:{lineno} -> {value!r}")
        self.assertEqual(
            offenders, [],
            "库代码里出现了硬编码的 LLM 档案文件名——请改走 "
            "drsr_420.llm.roles（角色 → 档案 的解析）：\n" + "\n".join(offenders))

    def test_explain_module_no_longer_loads_a_fixed_profile(self):
        """回归：explain 曾硬编码一个**不存在**的档案，导致 explain.txt 恒为空。"""
        source = (_PKG_DIR / "analysis" / "explain.py").read_text(encoding="utf-8")
        self.assertNotIn("load_llm_config(", source,
                         "analysis/explain.py 不应自己加载固定档案，应经角色解析取客户端")


# ── 替身 ────────────────────────────────────────────────────────────

class _FakeClient:
    """最小客户端替身：``clone_for_task`` 返回 kwargs 独立的副本。"""

    def __init__(self, model: str = "fake/model", provider: str = "fake", **kwargs):
        self.model = model
        self.provider = provider
        self.kwargs = dict(kwargs)
        self.task_params: dict[str, dict] = {}

    def _provider_name(self) -> str:
        return self.provider

    def clone_for_task(self, task):
        new = copy.copy(self)
        new.kwargs = dict(self.kwargs)
        for key, value in (self.task_params.get(task) or {}).items():
            new.kwargs[key] = value
        return new


def _write_json(path: pathlib.Path, payload: dict) -> pathlib.Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class ShippedRegistryTest(unittest.TestCase):
    """随仓库入库的注册表必须自洽。"""

    def test_registry_is_tracked_and_has_no_secrets(self):
        path = roles_mod.registry_path()
        self.assertTrue(path.is_file(), f"缺少角色注册表 {path}")
        text = path.read_text(encoding="utf-8")
        for token in ("api_key", "sk-", "Bearer "):
            with self.subTest(token=token):
                self.assertNotIn(token, text,
                                 "注册表必须可入库：不得包含任何密钥字段或明文密钥")

    def test_shipped_registry_covers_every_role(self):
        registry = roles_mod.RoleRegistry.load()
        self.assertIsNotNone(registry.path, "应当能加载到随仓库分发的注册表")
        for role in roles_mod.TASKS:
            with self.subTest(role=role):
                self.assertIn(role, registry.roles,
                              f"注册表未声明角色 {role}（参数会退回内置默认，容易与文档不一致）")

    def test_default_profile_has_a_shipped_template(self):
        """零配置起步：默认档案必须有可复制的模板，否则新克隆无从下手。"""
        registry = roles_mod.RoleRegistry.load()
        template = _CONFIG_DIR / f"{registry.default}.config.example"
        self.assertTrue(template.is_file(),
                        f"默认档案 {registry.default} 缺少随仓库分发的模板 {template.name}")

    def test_every_existing_profile_has_a_template(self):
        """本机已有的每个 ``*.config`` 都要有同名模板，否则别人克隆后拿不到该档案。

        （反向不成立：新克隆只有模板、没有档案，那是正常状态。）
        """
        if not _CONFIG_DIR.is_dir():
            self.skipTest("配置目录不存在")
        profiles = {p.name for p in _CONFIG_DIR.glob("*.config")}
        templates = {p.name[: -len(".example")]
                     for p in _CONFIG_DIR.glob("*.config.example")}
        self.assertEqual(profiles - templates, set(),
                         "这些档案没有可入库的模板（新克隆拿不到起点）")

    def test_registry_bound_profiles_have_shipped_templates(self):
        """注册表引用的每份档案都必须有模板，``--check`` 给出的 ``cp`` 才对得上。

        这条随"角色单独绑定档案"一起变得重要：绑定越具体，越容易指向一份
        别人克隆后根本建不出来的档案。
        """
        registry = roles_mod.RoleRegistry.load()
        bound = {registry.default} | {e.profile for e in registry.roles.values()
                                      if e.profile}
        for profile in sorted(bound):
            with self.subTest(profile=profile):
                template = _CONFIG_DIR / f"{profile}.config.example"
                self.assertTrue(template.is_file(),
                                f"注册表引用了 {profile}，但缺少模板 {template.name}")

    def test_every_shipped_template_builds_a_client(self):
        """模板不能只是"存在"——填上占位密钥后必须真的能构造出客户端。

        这是唯一能同时守住三类漂移的检查：``model`` 拼写、自定义提供商漏写
        ``base_url``、``dialect`` 拼错。跳过不含 ``model`` 的模板（``rag.config``
        是知识库配置，不是 LLM 档案——用结构而非文件名区分）。
        """
        templates = sorted(_CONFIG_DIR.glob("*.config.example"))
        self.assertTrue(templates, "应当有随仓库分发的模板")
        checked = 0
        for template in templates:
            config = json.loads(template.read_text(encoding="utf-8"))
            if "model" not in config:
                continue
            with self.subTest(template=template.name):
                config = dict(config, api_key="placeholder-not-a-secret")
                client = factory_mod.ClientFactory.from_config(config)
                self.assertIsNotNone(client)
            checked += 1
        self.assertGreaterEqual(checked, 2, "至少要检查到默认档案与自定义提供商档案")

    def test_shipped_endpoints_are_complete_urls(self):
        """随仓库分发的端点必须是**完整 URL**（带 scheme 且带路径），不接受裸主机域名。

        运行时只强制"带 scheme"（``llm.client.require_absolute_url``）——根路径挂载的
        自建网关是合法的；但**我们分发出去的**档案统一成 ``https://<主机>/v1`` 这种形状，
        否则下一个人照抄模板又把主机域名带了回来（``api.deepseek.com`` 就是这么来的）。
        """
        pattern = re.compile(r"^https?://[^/\s]+/.+")
        checked = 0
        for path in sorted(_CONFIG_DIR.glob("*.config.example")):
            data = json.loads(path.read_text(encoding="utf-8"))
            endpoint = data.get("base_url") or data.get("api_base_url")
            if not endpoint:
                continue
            with self.subTest(config=path.name):
                self.assertRegex(endpoint, pattern,
                                 f"{path.name} 的端点 {endpoint!r} 不是完整 URL")
            checked += 1
        self.assertGreaterEqual(checked, 3, "至少要检查到 LLM 与 RAG 两侧的端点")

    def test_shipped_configs_use_the_unified_base_url_key(self):
        """端点键名统一：随仓库分发的档案里不得再出现 ``host`` / ``api_host``。

        只看**键名**、不做子串匹配——``base_url`` 自己就含 "url"，而 RAG 侧的
        ``api_base_url`` 正是同一次统一的对应写法。代码侧由 factory / rag_kb 的
        报错（含改名提示）兜底：旧键不是"还能用"，而是"写了就报错"。
        """
        renamed = {"host": "base_url", "api_host": "api_base_url"}
        for path in sorted(_CONFIG_DIR.glob("*.config.example")):
            data = json.loads(path.read_text(encoding="utf-8"))
            for old, new in renamed.items():
                with self.subTest(config=path.name, key=old):
                    self.assertNotIn(old, data,
                                     f"{path.name} 仍用旧键名 {old!r}，应改为 {new!r}")


class RoleRegistryLoadingTest(unittest.TestCase):
    """注册表的加载与校验。"""

    def test_missing_file_falls_back_to_empty_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = roles_mod.RoleRegistry.load(pathlib.Path(tmp) / "nope.json")
        self.assertIsNone(registry.path)
        self.assertEqual(registry.default, factory_mod.DEFAULT_PROFILE)
        self.assertEqual(registry.roles, {})

    def test_unknown_role_is_rejected(self):
        """拼错角色名必须报错——静默忽略一个角色会让"配置了却没生效"极难排查。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_json(pathlib.Path(tmp) / "agents.config.json",
                               {"roles": {"smapling": {}}})
            with self.assertRaises(ValueError) as ctx:
                roles_mod.RoleRegistry.load(path)
        self.assertIn("smapling", str(ctx.exception))
        self.assertIn("sampling", str(ctx.exception))

    def test_unsupported_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_json(pathlib.Path(tmp) / "agents.config.json",
                               {"version": 99, "roles": {}})
            with self.assertRaises(ValueError):
                roles_mod.RoleRegistry.load(path)

    def test_entry_defaults_to_empty_declaration(self):
        registry = roles_mod.RoleRegistry.empty()
        entry = registry.entry("sampling")
        self.assertIsNone(entry.profile)
        self.assertEqual(entry.params, {})


class RoleResolutionPrecedenceTest(unittest.TestCase):
    """6 级优先级逐级验证（文档写的就是这里测的）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.registry = roles_mod.RoleRegistry.load(_write_json(
            pathlib.Path(self._tmp.name) / "agents.config.json",
            {"default": "registry_default",
             "roles": {"sampling": {"config": "registry_sampling"}}}))

    def _resolve(self, **kwargs):
        kwargs.setdefault("registry", self.registry)
        kwargs.setdefault("environ", {})
        return roles_mod.resolve_roles(**kwargs)

    def test_registry_default_when_nothing_is_declared(self):
        result = self._resolve()["analysis"]
        self.assertEqual(result.profile, "registry_default")
        self.assertEqual(result.source, "registry:default")

    def test_registry_role_binding_beats_cli_default(self):
        """关键规则：角色绑定高于 --llm_config。

        否则 IDE 运行配置里那句 ``--llm_config`` 会把注册表的角色绑定永久屏蔽
        ——这正是旧结构下"给 explain 换模型"无法生效的根因。
        """
        result = self._resolve(cli_default="from_cli")["sampling"]
        self.assertEqual(result.profile, "registry_sampling")
        self.assertEqual(result.source, "registry:roles.sampling.config")

    def test_cli_default_beats_registry_default(self):
        result = self._resolve(cli_default="from_cli")["analysis"]
        self.assertEqual(result.profile, "from_cli")
        self.assertEqual(result.source, "cli:--llm_config")

    def test_env_beats_registry_role_binding(self):
        result = self._resolve(
            environ={"DRSR_ROLE_CONFIG_SAMPLING": "from_env"})["sampling"]
        self.assertEqual(result.profile, "from_env")
        self.assertEqual(result.source, "env:DRSR_ROLE_CONFIG_SAMPLING")

    def test_role_config_beats_everything(self):
        result = self._resolve(
            cli_default="from_cli",
            cli_overrides={"sampling": "from_role_cli"},
            environ={"DRSR_ROLE_CONFIG_SAMPLING": "from_env"},
        )["sampling"]
        self.assertEqual(result.profile, "from_role_cli")
        self.assertEqual(result.source, "cli:--role-config")

    def test_wildcard_role_config_forces_every_role(self):
        resolutions = self._resolve(cli_overrides="*=forced")
        for role, resolution in resolutions.items():
            with self.subTest(role=role):
                self.assertEqual(resolution.profile, "forced")

    def test_unknown_role_in_cli_override_is_rejected(self):
        with self.assertRaises(ValueError):
            self._resolve(cli_overrides={"explian": "x"})

    def test_malformed_cli_override_is_rejected(self):
        with self.assertRaises(ValueError):
            self._resolve(cli_overrides=["no-equals-sign"])

    def test_every_role_is_resolved(self):
        self.assertEqual(tuple(self._resolve()), roles_mod.TASKS)


class RoleParamResolutionTest(unittest.TestCase):
    """参数合并：内置默认 < 档案 tasks[role]（旧格式） < 注册表 params。"""

    def test_builtin_defaults_match_historical_behaviour(self):
        """默认值刻意与重构前逐位相同：思考强度来自档案 ``tasks``，经验/残差的
        temperature 等原先是写死在 ``coordinator_agent`` 实参里的。"""
        registry = roles_mod.RoleRegistry.empty()
        self.assertEqual(roles_mod.resolve_params("sampling", registry=registry),
                         {"reasoning_effort": "low"})
        experience = roles_mod.resolve_params("experience", registry=registry)
        self.assertEqual(experience["temperature"], 0.0)
        self.assertEqual(experience["top_p"], 1.0)
        residual = roles_mod.resolve_params("residual", registry=registry)
        self.assertEqual(residual["temperature"], 0.4)
        self.assertEqual(residual["frequency_penalty"], 0.1)

    def test_legacy_tasks_field_still_honoured(self):
        """模板里已不再有 ``tasks``，但用户已有的档案文件里可能还有。"""
        registry = roles_mod.RoleRegistry.empty()
        legacy = {"tasks": {"sampling": {"reasoning_effort": "medium"}}}
        params = roles_mod.resolve_params("sampling", registry=registry,
                                         profile_config=legacy)
        self.assertEqual(params["reasoning_effort"], "medium")

    def test_registry_params_beat_legacy_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = roles_mod.RoleRegistry.load(_write_json(
                pathlib.Path(tmp) / "agents.config.json",
                {"roles": {"sampling": {"params": {"reasoning_effort": "high"}}}}))
        legacy = {"tasks": {"sampling": {"reasoning_effort": "medium"}}}
        params = roles_mod.resolve_params("sampling", registry=registry,
                                         profile_config=legacy)
        self.assertEqual(params["reasoning_effort"], "high")

    def test_null_values_are_not_injected(self):
        """显式写 null 表示"不注入该参数"，不应把它塞进 kwargs。"""
        with tempfile.TemporaryDirectory() as tmp:
            registry = roles_mod.RoleRegistry.load(_write_json(
                pathlib.Path(tmp) / "agents.config.json",
                {"roles": {"sampling": {"params": {"temperature": None}}}}))
        params = roles_mod.resolve_params("sampling", registry=registry)
        self.assertNotIn("temperature", params)


class ConfigLocationTest(unittest.TestCase):
    """档案定位：找不到时给出的路径必须是对的（报错里的路径写错会把人带偏）。"""

    def test_missing_relative_path_does_not_double_the_config_dir(self):
        resolved = factory_mod.locate_config("config/definitely_missing.config")
        self.assertEqual(resolved.parent, factory_mod.config_dir())
        self.assertEqual(resolved.name, "definitely_missing.config")

    def test_missing_bare_name_lands_in_the_config_dir(self):
        resolved = factory_mod.locate_config("no_such_profile_at_all")
        self.assertEqual(resolved.parent, factory_mod.config_dir())
        self.assertEqual(resolved.name, "no_such_profile_at_all.config")


class RoleClientsTest(unittest.TestCase):
    """客户端池：独立克隆、参数注入、单客户端兜底。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.registry = roles_mod.RoleRegistry.load(_write_json(
            pathlib.Path(self._tmp.name) / "agents.config.json",
            {"default": "shared_profile",
             "roles": {"residual": {"params": {"temperature": 0.4}}}}))
        self.clients = RoleClients.from_registry(
            registry=self.registry, environ={}, base_client=_FakeClient())

    def test_get_returns_independent_clones(self):
        """kwargs 隔离——历史上三个用途互相覆盖，最后全变成 temperature=0.4。"""
        first = self.clients.get("sampling")
        second = self.clients.get("sampling")
        self.assertIsNot(first, second)
        first.kwargs["temperature"] = 999.0
        self.assertNotEqual(second.kwargs.get("temperature"), 999.0)

    def test_role_params_are_injected(self):
        self.assertEqual(self.clients.get("residual").kwargs["temperature"], 0.4)
        self.assertEqual(self.clients.get("sampling").kwargs["reasoning_effort"], "low")

    def test_unknown_role_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.clients.get("typo")

    def test_single_none_yields_no_clients(self):
        """``llm_client=None``（测试与"不接模型"的旧调用方）不得去读档案建连接。"""
        empty = RoleClients.single(None)
        for role in roles_mod.TASKS:
            with self.subTest(role=role):
                self.assertIsNone(empty.get(role))

    def test_single_client_is_shared_by_all_roles(self):
        single = RoleClients.single(_FakeClient())
        self.assertIsNotNone(single.get("explain"))
        self.assertIsNotNone(single.get("summary"))

    def test_one_base_client_per_distinct_profile(self):
        """6 个角色共用一份档案时只建一个底层客户端（不建 6 套连接）。

        构造是**按需**的：``from_registry`` 只解析，第一次 ``get()`` 才建。
        """
        with mock.patch.object(role_clients_mod, "load_llm_config",
                               return_value={"model": "fake/model"}), \
             mock.patch.object(role_clients_mod.ClientFactory, "from_config",
                               return_value=_FakeClient()) as build:
            clients = RoleClients.from_registry(registry=self.registry, environ={})
            self.assertEqual(build.call_count, 0,
                             "from_registry 不该建连接——那会让用不到的角色拖死启动")
            clients.get("sampling")
            clients.get("analysis")
        self.assertEqual(build.call_count, 1)
        self.assertEqual(len(clients._base_by_profile), 1)

    def test_describe_and_resolved_profiles_cover_every_role(self):
        described = self.clients.describe()
        self.assertEqual(set(described), set(roles_mod.TASKS))
        for role, info in described.items():
            with self.subTest(role=role):
                self.assertIn("config", info)
                self.assertIn("source", info)
                self.assertNotIn("api_key", info)
        self.assertEqual(set(self.clients.resolved_profiles()), set(roles_mod.TASKS))


class LazyClientConstructionTest(unittest.TestCase):
    """按需构造：一份坏档案不该拖死用不到它的角色。

    动机是真实场景——``summary`` 绑定了一份密钥没填好的档案，结果一次完全不读文献的
    实验连启动都起不来（``cli.llm_setup`` 原本一启动就构造所有档案并 SystemExit）。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = pathlib.Path(self._tmp.name)
        # 好档案：能真的构造出客户端（glm 有内置端点，构造不发网络请求）
        self.good = _write_json(self.tmp / "good.config",
                                {"model": "glm/glm-x", "api_key": "k"})
        # 坏档案：**确定性**失败，不依赖环境变量（未知 provider 且没给 base_url）。
        # 不用"密钥没填"当坏例子——那会被 ZHIPU_API_KEY 之类的环境变量救活，
        # 测试就跑在与真实部署不同的分支上。
        self.bad = _write_json(self.tmp / "bad.config", {"model": "typo-provider/x"})
        self.registry = roles_mod.RoleRegistry.load(_write_json(
            self.tmp / "agents.config.json",
            {"default": str(self.good),
             "roles": {"summary": {"config": str(self.bad)}}}))

    def _clients(self):
        return RoleClients.from_registry(registry=self.registry, environ={})

    def test_resolution_does_not_read_or_build_anything(self):
        with mock.patch.object(role_clients_mod, "load_llm_config") as loader, \
             mock.patch.object(role_clients_mod.ClientFactory, "from_config") as build:
            self._clients()
        loader.assert_not_called()
        build.assert_not_called()

    def test_only_the_needed_profile_is_built(self):
        """核心行为：坏档案只影响它自己的角色，别的角色照常可用。"""
        clients = self._clients()
        with mock.patch.object(role_clients_mod.ClientFactory, "from_config",
                               return_value=_FakeClient()) as build:
            self.assertIsNotNone(clients.get("sampling"))
        self.assertEqual(build.call_count, 1)
        self.assertEqual([pathlib.Path(k).name for k in clients._base_by_profile],
                         ["good.config"])

    def test_broken_profile_raises_on_use_with_role_and_path_in_the_message(self):
        clients = self._clients()
        with self.assertRaises(RuntimeError) as ctx:
            clients.get("summary")
        message = str(ctx.exception)
        self.assertIn("summary", message)                  # 哪个角色
        self.assertIn("bad.config", message)               # 哪份档案
        self.assertIn("不支持的提供商", message)             # 原始原因（ClientFactory 的可操作提示）
        self.assertIn("--check", message)                  # 下一步怎么查

    def test_healthy_role_still_works_after_a_broken_one_failed(self):
        clients = self._clients()
        with self.assertRaises(RuntimeError):
            clients.get("summary")
        self.assertIsNotNone(clients.get("sampling"))

    def test_successful_construction_is_cached(self):
        clients = self._clients()
        first = clients.get("sampling")
        second = clients.get("sampling")
        self.assertEqual(len(clients._base_by_profile), 1, "同一档案只建一次")
        self.assertIsNot(first, second, "但每次 get 都必须是独立克隆")

    def test_build_role_client_only_touches_its_own_profile(self):
        """收尾分析（explain）这类"只用一份档案"的调用方不该被别的档案牵连。"""
        from drsr_420.llm.role_clients import build_role_client

        with mock.patch.object(role_clients_mod.RoleRegistry, "load",
                               return_value=self.registry), \
             mock.patch.object(role_clients_mod.ClientFactory, "from_config",
                               return_value=_FakeClient()) as build:
            self.assertIsNotNone(build_role_client("sampling"))
        self.assertEqual(build.call_count, 1)
        self.assertEqual(build.call_args[0][0].get("model"), "glm/glm-x")

    def test_cli_startup_warns_but_does_not_exit_on_a_broken_profile(self):
        """CLI 启动只把坏档案**告警**出来，不 SystemExit——否则用不到它也得起不来。"""
        from drsr_420.cli.llm_setup import build_role_clients

        out = io.StringIO()
        with mock.patch.object(roles_mod.RoleRegistry, "load",
                               return_value=self.registry), \
             mock.patch.dict(os.environ, {}, clear=True), \
             contextlib.redirect_stdout(out):
            clients = build_role_clients(None, None)      # 不抛 SystemExit
        text = out.getvalue()
        self.assertIn("[WARN]", text)
        self.assertIn("不支持的提供商", text)
        self.assertIn("bad.config", text)
        self.assertEqual(set(clients.resolved_profiles()), set(roles_mod.TASKS))


class RoleDiagnosticsTest(unittest.TestCase):
    """渲染与自检（``python -m drsr_420.llm.roles`` 的内容来源）。"""

    def test_describe_lists_every_role_and_the_precedence(self):
        text = describe_roles()
        for role in roles_mod.TASKS:
            with self.subTest(role=role):
                self.assertIn(role, text)
        self.assertIn("--role-config", text)

    def test_describe_output_is_gbk_encodable(self):
        """诊断输出会打到 Windows 控制台（默认代码页 GBK）——不可编码会直接抛异常。"""
        describe_roles().encode("gbk")

    def test_check_never_raises_and_every_problem_is_actionable(self):
        """自检本身不能抛异常；每条问题都要指出"下一步做什么"。

        刻意不断言"零问题"：本机是否已建好真实档案属于部署状态，新克隆的仓库里
        ``check`` 就该报告缺档案——关键是报告得可操作。

        也刻意不把"可操作"等同于"含 ``cp``"：问题分两类、修复动作不同——档案不存在
        用 ``cp <模板> <档案>``；档案在但密钥不可达则要"填字段 / 设环境变量"，给后者
        塞一句 cp 反而是误导（文件明明就在）。
        """
        problems = check_roles()
        self.assertIsInstance(problems, list)
        for problem in problems:
            with self.subTest(problem=problem):
                self.assertTrue(
                    "cp " in problem or "环境变量" in problem,
                    f"问题描述没给出可执行的下一步：{problem}")

    def test_check_reports_missing_key_with_the_env_var_name(self):
        """自定义提供商缺密钥：必须点出**具体**环境变量名，否则用户没法照做。"""
        with tempfile.TemporaryDirectory() as tmp:
            profile = _write_json(pathlib.Path(tmp) / "probe.config", {
                "model": "probevendor/probe-model",
                "base_url": "https://probe.invalid/v1",
                "api_key": "",
            })
            registry = roles_mod.RoleRegistry.load(_write_json(
                pathlib.Path(tmp) / "agents.config.json",
                {"default": str(profile)}))
            problems = check_roles(registry=registry, environ={})
        joined = "\n".join(problems)
        self.assertIn("PROBEVENDOR_API_KEY", joined)
        self.assertIn("环境变量", joined)

    def test_check_reports_missing_profile_with_copy_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = roles_mod.RoleRegistry.load(_write_json(
                pathlib.Path(tmp) / "agents.config.json",
                {"default": "definitely_missing_profile"}))
            problems = check_roles(registry=registry, environ={})
        self.assertTrue(any("档案文件不存在" in p for p in problems), problems)
        self.assertTrue(any("cp " in p for p in problems),
                        "缺失档案时应给出可复制的修复命令")

    def test_describe_widens_columns_for_long_profile_names(self):
        """列宽必须按内容算：``deepseek_deepseek-v4-flash``（26 字符）正好顶满旧的
        26 宽列，来源列被挤成 ``...flashregistry:roles...`` 连字。"""
        long_name = "deepseek_deepseek-v4-flash"
        with tempfile.TemporaryDirectory() as tmp:
            registry = roles_mod.RoleRegistry.load(_write_json(
                pathlib.Path(tmp) / "agents.config.json", {"default": long_name}))
            text = describe_roles(registry=registry, environ={})
        self.assertIn(long_name + "  ", text)
        self.assertNotIn(long_name + "registry", text)


class AgentSpecRoleTest(unittest.TestCase):
    """每个 Agent 声明的 llm_task 必须落在角色表里。"""

    def test_agent_llm_tasks_are_known_roles(self):
        from drsr_420.agents import agent_specs

        for key, spec in agent_specs().items():
            if spec.llm_task is None:
                continue
            with self.subTest(agent=key):
                self.assertIn(spec.llm_task, roles_mod.TASKS,
                              f"{key} 声明了未注册的角色 {spec.llm_task!r}")

    def test_every_role_is_used_by_something(self):
        """反向：角色表里不该有"没人用"的角色（孤儿配置就是这个问题）。"""
        from drsr_420.agents import agent_specs

        declared = {spec.llm_task for spec in agent_specs().values() if spec.llm_task}
        # explain / summary 不是 Agent，分别由 analysis.explain 与 MCP 工具 read_paper 使用
        non_agent_roles = {"explain", "summary"}
        self.assertEqual(set(roles_mod.TASKS) - declared - non_agent_roles, set())


class ExplainRoleWiringTest(unittest.TestCase):
    """回归：物理解释必须用 ``explain`` 角色解析出的客户端。

    旧实现自建客户端并硬编码档案名，既绕过了 ``--llm_config`` / ``--role-config``，
    也在档案不存在时静默产出空的 ``explain.txt``。
    """

    def _run(self, role_clients):
        from drsr_420.analysis import explain as explain_mod

        captured = {}

        def fake_re_act(client, content):
            captured["client"] = client
            return "EXPLAINED"

        with tempfile.TemporaryDirectory() as tmp:
            results_root = pathlib.Path(tmp)
            _write_json(results_root / "experiences.json", {
                "Good": [{"sample_order": "7", "function": "def f():\n    return 1"}],
            })
            with mock.patch.object(explain_mod, "explain_re_act", fake_re_act), \
                 mock.patch.object(explain_mod, "build_explain_content",
                                   lambda func, exp: "PROMPT"), \
                 mock.patch("builtins.print"):
                explain_mod.explain_best_sample(
                    str(results_root), "func", "7", role_clients=role_clients)
            written = (results_root / "explain.txt").read_text(encoding="utf-8")

        return captured, written

    def test_uses_the_explain_role_client(self):
        sentinel = _FakeClient()
        captured, written = self._run(RoleClients.single(sentinel))
        self.assertIsNotNone(captured.get("client"),
                             "explain 应当从 role_clients 取到 explain 角色的客户端")
        self.assertEqual(captured["client"].model, sentinel.model)
        self.assertEqual(written, "EXPLAINED")

    def test_without_role_clients_it_resolves_the_explain_role_itself(self):
        """直接调用（不经 CLI）时也要按注册表解析 ``explain`` 角色取客户端。

        这样才能摆脱旧实现"自建客户端 + 硬编码档案名"的老路。
        """
        from drsr_420.analysis import explain as explain_mod

        sentinel = _FakeClient(model="resolved/model")
        with mock.patch.object(explain_mod.llm, "build_role_client",
                               return_value=sentinel) as resolver:
            captured, _written = self._run(None)
        resolver.assert_called_once_with("explain")
        self.assertIs(captured["client"].model, sentinel.model)

    def test_client_init_failure_is_reported_not_swallowed_silently(self):
        """档案解析失败时必须留下 WARN 与排查指引（旧实现只留一行 WARN 后写空文件）。"""
        from drsr_420.analysis import explain as explain_mod

        with tempfile.TemporaryDirectory() as tmp:
            results_root = pathlib.Path(tmp)
            _write_json(results_root / "experiences.json", {
                "Good": [{"sample_order": "7", "function": "def f():\n    return 1"}],
            })
            with mock.patch.object(explain_mod, "build_explain_content",
                                   lambda func, exp: "PROMPT"), \
                 mock.patch.object(explain_mod.llm, "build_role_client",
                                   side_effect=RuntimeError("档案不存在")), \
                 mock.patch("builtins.print") as printer:
                explain_mod.explain_best_sample(str(results_root), "func", "7")
        printed = "\n".join(str(call.args[0]) for call in printer.call_args_list if call.args)
        self.assertIn("Failed to init LLM client", printed)
        self.assertIn("drsr_420.llm.roles --check", printed)


class SubprocessRolePropagationTest(unittest.TestCase):
    """MCP 子进程拿不到父进程的对象，只能靠环境变量继承选定的档案。"""

    def test_server_env_forwards_role_config_vars(self):
        from drsr_420.knowledge import tool_runner

        with mock.patch.dict(os.environ, {
            "DRSR_ROLE_CONFIG_SUMMARY": "/tmp/summary.config",
            "DRSR_SOMETHING_ELSE": "x",
        }, clear=False):
            env = tool_runner._server_env()
        self.assertEqual(env.get("DRSR_ROLE_CONFIG_SUMMARY"), "/tmp/summary.config")
        self.assertEqual(env.get("DRSR_SOMETHING_ELSE"), "x")

    def test_read_paper_follows_the_summary_role_env_var(self):
        """``read_paper`` 按 summary 角色解析档案，而不是写死文件名。"""
        from drsr_420.knowledge.tools import read_paper

        with tempfile.TemporaryDirectory() as tmp:
            profile = _write_json(pathlib.Path(tmp) / "probe.config",
                                  {"model": "fake/probe-model", "api_key": "k"})
            with mock.patch.dict(os.environ,
                                 {"DRSR_ROLE_CONFIG_SUMMARY": str(profile)},
                                 clear=False):
                config = read_paper._load_llm_config()
        self.assertEqual(config.get("model"), "fake/probe-model")

    def test_cli_exports_resolved_profiles_to_env(self):
        """CLI 必须把解析结果回写环境变量，子进程才看得到同一份选择。"""
        from drsr_420.cli.llm_setup import build_role_clients

        with tempfile.TemporaryDirectory() as tmp:
            registry = roles_mod.RoleRegistry.load(_write_json(
                pathlib.Path(tmp) / "agents.config.json",
                {"default": "cli_probe"}))
            with mock.patch.object(role_clients_mod, "load_llm_config",
                                   return_value={"model": "fake/model"}), \
                 mock.patch.object(role_clients_mod.ClientFactory, "from_config",
                                   return_value=_FakeClient()), \
                 mock.patch.object(role_clients_mod.RoleRegistry, "load",
                                   return_value=registry), \
                 mock.patch("builtins.print"), \
                 mock.patch.dict(os.environ, {}, clear=True):
                build_role_clients(None, None)
                exported = {k: v for k, v in os.environ.items()
                            if k.startswith(roles_mod.ENV_ROLE_PREFIX)}
        self.assertEqual(set(exported), {
            f"{roles_mod.ENV_ROLE_PREFIX}{role.upper()}" for role in roles_mod.TASKS})


class RolesCliSmokeTest(unittest.TestCase):
    """``python -m drsr_420.llm.roles`` 本身要干净可跑（无 runpy 警告）。"""

    def test_module_runs_without_warnings(self):
        # 显式 encoding：子进程按 PYTHONIOENCODING 输出 UTF-8，而 text=True 会用
        # 父进程的区域设置（本机为 GBK）解码，直接 UnicodeDecodeError。
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        proc = subprocess.run(
            [sys.executable, "-m", "drsr_420.llm.roles"],
            cwd=_REPO_ROOT, env=env, capture_output=True,
            encoding="utf-8", timeout=120,
        )
        self.assertEqual(proc.returncode, 0, (proc.stderr or "")[-1500:])
        self.assertNotIn("RuntimeWarning", proc.stderr or "")
        for role in roles_mod.TASKS:
            self.assertIn(role, proc.stdout or "")


class PingRolesTest(unittest.TestCase):
    """``--ping``：每份被引用的档案发一次**真实**请求。

    这里用假 transport 验证逻辑（不联网）：去重、按首个角色取参数、状态码与网关信封
    的呈现、输出上限、以及"坏档案不联网也能报出来"。
    """

    class _Resp:
        def __init__(self, status_code=200, payload=None, text=None):
            self.status_code = status_code
            self._payload = payload
            self.text = text if text is not None else json.dumps(payload or {},
                                                                 ensure_ascii=False)

        def json(self):
            if self._payload is None:
                raise ValueError("not json")
            return self._payload

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = pathlib.Path(self._tmp.name)

    def _write_registry(self, *, with_bad=True, max_tokens=None):
        good = _write_json(self.tmp / "good.config",
                           {"model": "glm/glm-x", "api_key": "k",
                            **({"max_tokens": max_tokens} if max_tokens else {})})
        other = _write_json(self.tmp / "other.config",
                            {"model": "deepseek/dx", "api_key": "k"})
        roles = {"explain": {"config": str(other)}}
        if with_bad:
            bad = _write_json(self.tmp / "bad.config", {"model": "typo-provider/x"})
            roles["summary"] = {"config": str(bad)}
        path = _write_json(self.tmp / "agents.config.json",
                           {"default": str(good), "roles": roles})
        return roles_mod.RoleRegistry.load(path)

    def _ping(self, registry, resp):
        with mock.patch.object(diag_mod, "_post_with_retry", return_value=resp) as post:
            outcomes = diag_mod.ping_roles(registry=registry, environ={})
        return outcomes, post

    def _ok_payload(self):
        return {"choices": [{"message": {"content": "pong"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    def test_ping_dedupes_by_profile_and_skips_unbuildable_ones(self):
        """4 个角色共用 default，但只发一次请求；坏档案连网都不联。"""
        registry = self._write_registry()
        outcomes, post = self._ping(registry, self._Resp(payload=self._ok_payload()))
        self.assertEqual(len(outcomes), 3, "3 份档案：default / explain / summary")
        self.assertEqual(post.call_count, 2,
                         "可用的两份各一次；坏档案在构造阶段就失败了，不该发请求")

    def test_ping_marks_results_and_problems(self):
        registry = self._write_registry()
        outcomes, _post = self._ping(registry, self._Resp(payload=self._ok_payload()))
        by_profile = {o.profile: o for o in outcomes}
        self.assertTrue(by_profile["good"].ok)
        self.assertTrue(by_profile["other"].ok)
        self.assertFalse(by_profile["bad"].ok)
        self.assertIn("不支持的提供商", by_profile["bad"].detail)

        problems = diag_mod.ping_problems(outcomes)
        self.assertEqual(len(problems), 1)
        self.assertIn("bad", problems[0])
        self.assertIn("summary", problems[0])

    def test_ping_reports_http_status_and_body(self):
        registry = self._write_registry(with_bad=False)
        outcomes, _post = self._ping(
            registry, self._Resp(status_code=401, payload=None, text="bad key"))
        self.assertTrue(all(not o.ok for o in outcomes))
        self.assertIn("HTTP 401", outcomes[0].detail)
        self.assertIn("bad key", outcomes[0].detail)
        self.assertIn("http", outcomes[0].detail)      # 端点也带上，便于直接改档案

    def test_ping_reports_gateway_envelope_inside_http_200(self):
        """网关把 404 包在 200 里 —— 正是本轮实测撞到的那种。"""
        registry = self._write_registry(with_bad=False)
        envelope = {"code": 500, "msg": "404 NOT_FOUND", "success": False}
        outcomes, _post = self._ping(registry, self._Resp(payload=envelope))
        self.assertTrue(all(not o.ok for o in outcomes))
        self.assertIn("404 NOT_FOUND", outcomes[0].detail)
        self.assertIn("没有 choices", outcomes[0].detail)

    def test_ping_caps_tokens_and_disables_stream(self):
        registry = self._write_registry(with_bad=False, max_tokens=65536)
        _outcomes, post = self._ping(registry, self._Resp(payload=self._ok_payload()))
        # 第一次调用就是 default 档案（首个角色 sampling）；call_args 是*最后一次*调用
        payload = post.call_args_list[0][0][2]
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["max_tokens"], diag_mod.PING_MAX_TOKENS)

    def test_ping_uses_the_first_role_of_the_profile(self):
        """default 档案的首个角色是 sampling（TASKS 顺序）→ 带上它的 thinking 方言。"""
        registry = self._write_registry(with_bad=False)
        _outcomes, post = self._ping(registry, self._Resp(payload=self._ok_payload()))
        payload = post.call_args_list[0][0][2]
        self.assertEqual(payload.get("thinking"), {"type": "enabled"})

    def test_format_ping_is_gbk_encodable_even_with_hostile_error_text(self):
        """诊断输出要能在 Windows 控制台（GBK）打出来，服务端回什么都得兜住。"""
        registry = self._write_registry(with_bad=False)
        outcomes, _post = self._ping(
            registry, self._Resp(status_code=500, payload=None, text="boom 🧨 é"))
        text = diag_mod.format_ping(outcomes)
        text.encode("gbk")                       # 不抛 UnicodeEncodeError
        self.assertIn("boom", text)

    def test_format_ping_lists_profiles_and_roles(self):
        registry = self._write_registry()
        outcomes, _post = self._ping(registry, self._Resp(payload=self._ok_payload()))
        text = diag_mod.format_ping(outcomes)
        self.assertIn("good", text)
        self.assertIn("summary", text)           # 角色列
        self.assertIn("OK", text)
        self.assertIn("FAIL", text)


class PingCliTest(unittest.TestCase):
    """``python -m drsr_420.llm.roles --ping`` 的退出码与输出。"""

    class _Resp:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload
            self.text = json.dumps(payload, ensure_ascii=False)

        def json(self):
            return self._payload

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = pathlib.Path(self._tmp.name)
        good = _write_json(self.tmp / "good.config",
                           {"model": "glm/glm-x", "api_key": "k"})
        bad = _write_json(self.tmp / "bad.config", {"model": "typo-provider/x"})
        self.registry_path = _write_json(
            self.tmp / "agents.config.json",
            {"default": str(good), "roles": {"summary": {"config": str(bad)}}})

    def _run(self, argv):
        out = io.StringIO()
        with mock.patch.object(diag_mod, "_post_with_retry",
                               return_value=self._Resp({"choices": [{"message": {"content": "pong"}}]})), \
             contextlib.redirect_stdout(out):
            code = roles_mod.main(argv)
        return code, out.getvalue()

    def test_ping_returns_1_and_names_the_broken_profile(self):
        code, text = self._run(["--ping", "--registry", str(self.registry_path)])
        self.assertEqual(code, 1)
        self.assertIn("bad", text)
        self.assertIn("连通性自检失败", text)

    def test_ping_returns_0_when_only_healthy_profiles_exist(self):
        good = _write_json(self.tmp / "good.config",
                           {"model": "glm/glm-x", "api_key": "k"})
        path = _write_json(self.tmp / "ok.json", {"default": str(good)})
        code, text = self._run(["--ping", "--registry", str(path)])
        self.assertEqual(code, 0)
        self.assertIn("OK", text)
        self.assertIn("每份档案都真实响应了", text)


if __name__ == "__main__":
    unittest.main()
