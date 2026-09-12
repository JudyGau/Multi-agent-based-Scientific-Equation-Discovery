"""架构护栏测试：守护"结构契约"，而非功能行为。

本文件随 `docs/REFACTOR_PLAN.md` 的分阶段重构逐步收紧。阶段 0 只锁定
"当前成立、且重构过程中绝不能破坏"的几件事：

1. 旧导入路径（兼容层）与规范模块必须是**同一个对象**（`is` 比较）——
   防止 shim 被复制粘贴成副本后在两处分叉演进；
2. 包内每个模块都能单独导入（防循环导入随重构引入）；
3. `drsr_420/agents/__init__.py` 在模块顶层不得 import 任何 Agent 子模块——
   这是既有的防循环导入设计，后续改用 PEP 562 惰性导出后必须继续保持；
4. Agent 层不得反向依赖编排层（`pipeline` / `find_best_eq` / `cli`）。

阶段 3 会在本文件追加"分层目录 + 依赖方向"的完整断言。

基线（阶段 0 记录，阶段 5 对照）：
    测试数 217 ｜ drsr_420+tests 源码 9,375 行 ｜ drsr_420/ 顶层 .py 21 个
    （15 实现 + 6 兼容 shim）｜ 子包 2 个 ｜ 最长文件 llm.py 773 行 ｜ Agent 7 个
"""
from __future__ import annotations

import ast
import importlib
import os
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AGENTS_DIR = os.path.join(_REPO_ROOT, "drsr_420", "agents")

# 规范模块 → 该模块内的公开类
_CANONICAL_AGENTS = {
    "drsr_420.agents.coordinator_agent": ["CoordinatorAgent"],
    "drsr_420.agents.sampler_agent": ["SamplerAgent", "LLM"],
    "drsr_420.agents.tool_caller_agent": ["ToolCallerAgent"],
    "drsr_420.agents.evaluator_agent": ["EvaluatorAgent", "Sandbox", "LocalSandbox"],
    "drsr_420.agents.experience_summarizer_agent": ["ExperienceSummarizerAgent"],
    "drsr_420.agents.residual_analyzer_agent": ["ResidualAnalyzerAgent"],
    "drsr_420.agents.data_analyzer_agent": ["DataAnalyzerAgent"],
}

# 旧路径模块 → {旧名字: (规范模块, 规范名字)}
_LEGACY_SHIMS = {
    "drsr_420.sampler": {
        "Sampler": ("drsr_420.agents.sampler_agent", "SamplerAgent"),
        "SamplingOrchestrator": ("drsr_420.agents.coordinator_agent", "CoordinatorAgent"),
        "LLM": ("drsr_420.agents.sampler_agent", "LLM"),
    },
    "drsr_420.evaluator": {
        "Evaluator": ("drsr_420.agents.evaluator_agent", "EvaluatorAgent"),
        "Sandbox": ("drsr_420.agents.evaluator_agent", "Sandbox"),
        "LocalSandbox": ("drsr_420.agents.evaluator_agent", "LocalSandbox"),
    },
    "drsr_420.tool_caller": {
        "ToolCaller": ("drsr_420.agents.tool_caller_agent", "ToolCallerAgent"),
    },
    "drsr_420.experience_summarizer": {
        "ExperienceSummarizer": (
            "drsr_420.agents.experience_summarizer_agent",
            "ExperienceSummarizerAgent",
        ),
    },
    "drsr_420.residual_analyzer": {
        "ResidualAnalyzer": (
            "drsr_420.agents.residual_analyzer_agent",
            "ResidualAnalyzerAgent",
        ),
    },
    "drsr_420.data_analyse_real": {
        "DataAnalyzer": ("drsr_420.agents.data_analyzer_agent", "DataAnalyzerAgent"),
    },
}


def _is_type_checking_guard(test: ast.expr) -> bool:
    """识别 `if TYPE_CHECKING:` / `if typing.TYPE_CHECKING:`——运行时不执行的守卫。"""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _module_level_imports(path: str) -> list[str]:
    """收集模块顶层会**真正执行**的 import 目标（函数体内的 import 视为惰性，不计入）。

    类体（ClassDef）在导入时执行，因此计入；函数体与 ``if TYPE_CHECKING:`` 块不计入。
    """
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)

    found: list[str] = []

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # 函数体内的 import 是惰性的
            if isinstance(child, ast.If) and _is_type_checking_guard(child.test):
                continue  # if TYPE_CHECKING: 块运行时不执行
            if isinstance(child, ast.Import):
                found.extend(alias.name for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module:
                found.append(child.module)
                found.extend(f"{child.module}.{alias.name}" for alias in child.names)
            walk(child)

    walk(tree)
    return found


class LegacyShimIdentityTest(unittest.TestCase):
    """旧路径必须是同一对象的别名，不能是复制出来的副本。"""

    def test_legacy_shims_alias_canonical_classes(self):
        for legacy_module, names in _LEGACY_SHIMS.items():
            legacy = importlib.import_module(legacy_module)
            for legacy_name, (canonical_module, canonical_name) in names.items():
                canonical = importlib.import_module(canonical_module)
                with self.subTest(shim=legacy_module, name=legacy_name):
                    self.assertTrue(
                        hasattr(legacy, legacy_name),
                        f"{legacy_module}.{legacy_name} 缺失（兼容层被破坏）",
                    )
                    self.assertIs(
                        getattr(legacy, legacy_name),
                        getattr(canonical, canonical_name),
                        f"{legacy_module}.{legacy_name} 与 "
                        f"{canonical_module}.{canonical_name} 不是同一对象",
                    )


class ModuleImportabilityTest(unittest.TestCase):
    """每个 Agent 模块都必须能单独导入（防循环导入）。"""

    def test_every_agent_module_imports(self):
        for module_path in _CANONICAL_AGENTS:
            with self.subTest(module=module_path):
                module = importlib.import_module(module_path)
                for name in _CANONICAL_AGENTS[module_path]:
                    self.assertTrue(hasattr(module, name), f"{module_path} 缺少 {name}")

    def test_agents_package_has_no_eager_agent_imports(self):
        """`agents/__init__.py` 顶层不得 import Agent 子模块（既有的防循环设计）。"""
        init_path = os.path.join(_AGENTS_DIR, "__init__.py")
        self.assertTrue(os.path.exists(init_path), "drsr_420/agents/__init__.py 缺失")
        eager = [
            name for name in _module_level_imports(init_path)
            if name.startswith("drsr_420.agents.")
        ]
        self.assertEqual(
            eager, [],
            f"agents/__init__.py 顶层 import 了 Agent 子模块（可能引入循环导入）: {eager}",
        )


class LayerDirectionTest(unittest.TestCase):
    """Agent 角色层不得反向依赖编排层。"""

    FORBIDDEN_TARGETS = (
        "drsr_420.pipeline",
        "drsr_420.find_best_eq",
        "drsr_420.cli",
        "main",
    )

    def test_agents_do_not_import_orchestration_layers(self):
        for filename in sorted(os.listdir(_AGENTS_DIR)):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(_AGENTS_DIR, filename)
            imports = _module_level_imports(path)
            for forbidden in self.FORBIDDEN_TARGETS:
                for imported in imports:
                    with self.subTest(file=filename, imported=imported):
                        self.assertFalse(
                            imported == forbidden or imported.startswith(forbidden + "."),
                            f"agents/{filename} 反向依赖编排层 {imported}",
                        )


def _agent_classes_by_key() -> dict[str, type]:
    """{SPEC.key: Agent 类}——只收集声明了 SPEC 的类（LLM 抽象基类不算 Agent）。"""
    classes: dict[str, type] = {}
    for module_path in _CANONICAL_AGENTS:
        module = importlib.import_module(module_path)
        for name in _CANONICAL_AGENTS[module_path]:
            cls = getattr(module, name)
            spec = getattr(cls, "SPEC", None)
            if spec is not None:
                classes[spec.key] = cls
    return classes


class AgentContractTest(unittest.TestCase):
    """阶段 1：7 个 Agent 必须都继承 BaseAgent，并声明自洽的角色卡。"""

    def test_registry_matches_agent_order(self):
        from drsr_420.agents import AGENT_ORDER, agent_specs

        specs = agent_specs()
        self.assertEqual(len(specs), 7, "系统应当恰好有 7 个 Agent 角色")
        self.assertEqual(list(specs), list(AGENT_ORDER))

    def test_every_agent_inherits_base_and_declares_spec(self):
        from drsr_420.agents.base import AgentSpec, BaseAgent

        classes = _agent_classes_by_key()
        self.assertEqual(len(classes), 7, "应当恰好收集到 7 个 Agent 类")
        for key, cls in classes.items():
            with self.subTest(agent=key):
                self.assertTrue(
                    issubclass(cls, BaseAgent),
                    f"{cls.__name__} 未继承 BaseAgent——多 Agent 契约不成立",
                )
                self.assertIsInstance(cls.SPEC, AgentSpec)

    def test_declared_entrypoints_are_callable(self):
        for key, cls in _agent_classes_by_key().items():
            for entry in cls.SPEC.entrypoints:
                with self.subTest(agent=key, entrypoint=entry):
                    self.assertTrue(callable(getattr(cls, entry, None)),
                                    f"{key}.SPEC 声明的入口 {entry} 不存在")

    def test_canonical_entrypoint_uses_us_spelling(self):
        """规范入口统一用 analyze（英式 analyse 只作为兼容别名存在）。"""
        for key, cls in _agent_classes_by_key().items():
            with self.subTest(agent=key):
                self.assertFalse(
                    cls.SPEC.canonical_entrypoint.startswith("analyse"),
                    f"{key} 的规范入口不应使用英式拼写 analyse",
                )

    def test_contract_self_check_passes(self):
        from drsr_420.agents import check_contracts

        self.assertEqual(check_contracts(), [])

    def test_architecture_renders_every_agent(self):
        from drsr_420.agents import agent_specs, describe_architecture

        rendered = describe_architecture()
        for key, spec in agent_specs().items():
            with self.subTest(agent=key):
                self.assertIn(f"[{key}]", rendered)
                self.assertIn(spec.role, rendered)


class LazyImportTest(unittest.TestCase):
    """`import drsr_420.agents` 不得顺带拉起任何 Agent 子模块。"""

    def test_package_import_stays_lazy(self):
        import subprocess
        import sys

        code = (
            "import sys, drsr_420.agents; "
            "loaded = sorted(m for m in sys.modules "
            "if m.startswith('drsr_420.agents.') and m != 'drsr_420.agents.base'); "
            "print(','.join(loaded))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=_REPO_ROOT,
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(
            proc.stdout.strip(), "",
            "import drsr_420.agents 触发了 eager 导入（可能引入循环导入与启动开销）",
        )


if __name__ == "__main__":
    unittest.main()
