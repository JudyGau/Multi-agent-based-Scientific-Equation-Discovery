"""架构护栏测试：守护"结构契约"，而非功能行为。

本文件随 `docs/REFACTOR_PLAN.md` 的分阶段重构逐步收紧。

阶段 0：包内模块可单独导入、`agents/__init__.py` 顶层不 eager import、
Agent 层不反向依赖编排层。

阶段 1：7 个 Agent 都继承 `BaseAgent` 并声明自洽的 `SPEC`（角色卡）。

阶段 3：8 层目录结构、**依赖方向**硬约束、`__file__` 路径锚点。

阶段 4：评估机制归 `evaluation` 层，角色文件只留编排。

阶段 6：兼容层清退——旧路径必须导入失败、顶层只留 `__init__.py`、
库代码与测试都不得再 import 旧路径。

基线（阶段 0 记录）：测试数 217 ｜ drsr_420+tests 源码 9,375 行 ｜
drsr_420/ 顶层 .py 21 个（15 实现 + 6 兼容 shim）｜ 子包 2 个 ｜
最长文件 llm.py 773 行 ｜ Agent 7 个
"""
from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
import unittest
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG_DIR = os.path.join(_REPO_ROOT, "drsr_420")
_AGENTS_DIR = os.path.join(_PKG_DIR, "agents")

# ── 阶段 3：分层结构 ────────────────────────────────────────────────
#: 层的展示顺序即"自底向上"，越靠后越上层。
_LAYERS = ("core", "llm", "evaluation", "knowledge", "agents", "analysis",
           "runtime", "cli")

#: 每一层允许 import 的层（含自身）；其余一律视为越界/倒置。
_ALLOWED_LAYER_DEPS = {
    "core": set(),                                    # 最底层
    "llm": {"core"},                                  # token 统计下沉到 core.llm_stats
    "evaluation": {"core"},                           # 机制层：借 core 的 AST 与程序拼装
    "knowledge": {"llm"},                             # RAG 与 MCP 工具只依赖 LLM 接入
    "agents": {"core", "llm", "evaluation", "knowledge"},
    "analysis": {"core", "llm", "knowledge"},
    "runtime": {"core", "agents", "analysis", "knowledge"},
    "cli": {"core", "llm", "evaluation", "agents", "runtime"},
}

#: 顶层不再有任何实现模块，也不再有实现例外白名单（阶段 6 清退后只剩 __init__.py）。
#: 新代码一律进分层子包；确需新增一层请同时更新 _LAYERS 与 _ALLOWED_LAYER_DEPS。

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
# 阶段 6 已删除全部兼容层，此处只保留"旧路径不得复活"的清单（见 LegacyPathRemovalTest）。
_REMOVED_LEGACY_PATHS: dict[str, str] = {
    # core
    "drsr_420.buffer": "drsr_420.core.buffer",
    "drsr_420.code_manipulation": "drsr_420.core.code_manipulation",
    "drsr_420.config": "drsr_420.core.config",
    "drsr_420.console": "drsr_420.core.console",
    "drsr_420.profile": "drsr_420.core.profile",
    "drsr_420.prompt_config": "drsr_420.core.prompt_config",
    # evaluation
    "drsr_420.evaluate_on_problems": "drsr_420.evaluation.problems",
    "drsr_420.evaluator_accelerate": "drsr_420.evaluation.accelerate",
    # knowledge
    "drsr_420.rag_kb": "drsr_420.knowledge.rag_kb",
    "drsr_420.rag_build": "drsr_420.knowledge.rag_build",
    "drsr_420.tool_runner": "drsr_420.knowledge.tool_runner",
    "drsr_420.tools": "drsr_420.knowledge.tools",
    "drsr_420.tools.mcp_server": "drsr_420.knowledge.tools.mcp_server",
    "drsr_420.tools.search_paper": "drsr_420.knowledge.tools.search_paper",
    "drsr_420.tools.read_paper": "drsr_420.knowledge.tools.read_paper",
    "drsr_420.tools.tools_description": "drsr_420.llm.tools_schema",
    # analysis
    "drsr_420.find_best_eq": "drsr_420.analysis.find_best_eq",
    "drsr_420.sensitivity_prune": "drsr_420.analysis.sensitivity_prune",
    # runtime
    "drsr_420.pipeline": "drsr_420.runtime.pipeline",
    # agents
    "drsr_420.sampler": "drsr_420.agents.sampler_agent",
    "drsr_420.evaluator": "drsr_420.agents.evaluator_agent",
    "drsr_420.tool_caller": "drsr_420.agents.tool_caller_agent",
    "drsr_420.experience_summarizer": "drsr_420.agents.experience_summarizer_agent",
    "drsr_420.residual_analyzer": "drsr_420.agents.residual_analyzer_agent",
    "drsr_420.data_analyse_real": "drsr_420.agents.data_analyzer_agent",
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


class LegacyPathRemovalTest(unittest.TestCase):
    """阶段 6：兼容层已按 docs/ARCHITECTURE.md §8 清退，旧路径不得复活。

    三层保证：旧路径导入必须失败、规范路径必须仍然可用、源码里（含测试）不得
    再出现旧路径的 import——最后一条是关键：只要有人重新 import 旧路径而 shim
    已删除，运行期就会 ImportError，这里让它在提交前就失败。
    """

    def test_removed_paths_are_not_importable(self):
        for legacy, canonical in _REMOVED_LEGACY_PATHS.items():
            with self.subTest(legacy=legacy):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(legacy)
                importlib.import_module(canonical)   # 功能仍在，只是搬了家

    def test_top_level_holds_only_package_init(self):
        leftovers = sorted(
            path.name for path in Path(_PKG_DIR).glob("*.py")
            if path.name != "__init__.py")
        self.assertEqual(
            leftovers, [],
            "drsr_420/ 顶层只应剩 __init__.py：实现进分层子包，历史路径不再保留转发层")

    def test_no_module_imports_removed_paths(self):
        offenders: list[str] = []
        targets = (list(Path(_PKG_DIR).rglob("*.py"))
                   + list(Path(_REPO_ROOT, "tests").rglob("*.py")))
        for path in targets:
            if "__pycache__" in path.parts:
                continue
            for module_name in _all_imports(str(path)):
                for legacy in _REMOVED_LEGACY_PATHS:
                    if module_name == legacy or module_name.startswith(legacy + "."):
                        offenders.append(
                            f"{path.relative_to(_REPO_ROOT)} -> {module_name}")
        self.assertEqual(
            sorted(set(offenders)), [],
            "这些 import 指向已删除的旧路径（打包/运行时会 ImportError）:\n"
            + "\n".join(sorted(set(offenders))))


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
        "drsr_420.runtime",
        "drsr_420.analysis",
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


# ── 阶段 3：分层结构（兼容层已于阶段 6 清退，见 LegacyPathRemovalTest）──
#: 独立运行时需要 `__file__` 兜底插入仓库根的脚本（搬迁后深度必须同步修正）
_STANDALONE_SCRIPTS = (
    "drsr_420/knowledge/tools/read_paper.py",   # 最深层：parents[3]
)


def _all_imports(path: str) -> list[str]:
    """收集文件里**运行时会执行**的 import 目标。

    计入函数体内的 import（延迟导入也会形成真实依赖）；跳过 ``if TYPE_CHECKING:``
    块——它在运行时不执行，只是类型注解引用（例如 core/config 注解 agents 与
    evaluation 的类型），不构成依赖倒置。
    """
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)

    found: list[str] = []

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and _is_type_checking_guard(child.test):
                continue
            if isinstance(child, ast.Import):
                found.extend(alias.name for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module:
                found.append(child.module)
                found.extend(f"{child.module}.{alias.name}" for alias in child.names)
            walk(child)

    walk(tree)
    return found


def _layer_of(module_name: str) -> str | None:
    """模块名所属的层（不属于任何层则返回 None）。"""
    parts = module_name.split(".")
    if len(parts) >= 2 and parts[0] == "drsr_420" and parts[1] in _LAYERS:
        return parts[1]
    return None


class LayerLayoutTest(unittest.TestCase):
    """drsr_420/ 顶层只剩 `__init__.py`；实现都在分层子包里。"""

    def test_all_layers_exist_with_init(self):
        for layer in _LAYERS:
            init = Path(_PKG_DIR) / layer / "__init__.py"
            with self.subTest(layer=layer):
                self.assertTrue(init.exists(), f"缺少 {layer}/__init__.py")

    def test_package_has_version(self):
        import drsr_420
        self.assertTrue(getattr(drsr_420, "__version__", None),
                        "drsr_420 应声明 __version__")

    def test_no_subpackage_stays_empty(self):
        """每个分层子包都应有实现（空的层目录是"分层被掏空"的信号）。"""
        for layer in _LAYERS:
            files = [p.name for p in (Path(_PKG_DIR) / layer).glob("*.py")
                     if p.name != "__init__.py"]
            with self.subTest(layer=layer):
                self.assertTrue(files, f"{layer}/ 下没有任何实现模块")


class LayerDependencyTest(unittest.TestCase):
    """依赖方向硬约束：只允许自底向上，禁止越界与倒置。"""

    def test_dependencies_follow_layer_rules(self):
        violations: list[str] = []
        for layer in _LAYERS:
            allowed = _ALLOWED_LAYER_DEPS[layer] | {layer}
            for path in sorted((Path(_PKG_DIR) / layer).rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                for module_name in _all_imports(str(path)):
                    target = _layer_of(module_name)
                    if target is None or target in allowed:
                        continue
                    violations.append(
                        f"{layer}/{path.name} -> {target}  ({module_name})")
        self.assertEqual(
            violations, [],
            "存在越界/倒置的层间依赖（分层规则见 _ALLOWED_LAYER_DEPS）:\n"
            + "\n".join(sorted(set(violations))))

    def test_core_is_the_bottom_layer(self):
        """core 不得依赖任何其它层（否则最底层就不成其为底层）。"""
        allowed = _ALLOWED_LAYER_DEPS["core"] | {"core"}
        for path in sorted((Path(_PKG_DIR) / "core").rglob("*.py")):
            for module_name in _all_imports(str(path)):
                target = _layer_of(module_name)
                if target is not None:
                    with self.subTest(file=path.name, imported=module_name):
                        self.assertIn(target, allowed)

    def test_no_layer_imports_cli(self):
        """cli 是入口层：任何库代码都不应反向依赖它。"""
        offenders: list[str] = []
        for layer in _LAYERS:
            if layer == "cli":
                continue
            for path in sorted((Path(_PKG_DIR) / layer).rglob("*.py")):
                for module_name in _all_imports(str(path)):
                    if _layer_of(module_name) == "cli":
                        offenders.append(f"{layer}/{path.name}")
        self.assertEqual(offenders, [], f"库代码反向依赖 cli: {offenders}")

    def test_llm_layer_does_not_depend_on_agents(self):
        for path in sorted((Path(_PKG_DIR) / "llm").rglob("*.py")):
            for module_name in _all_imports(str(path)):
                with self.subTest(file=path.name, imported=module_name):
                    self.assertNotEqual(_layer_of(module_name), "agents")


class PathAnchorTest(unittest.TestCase):
    """`.parent` 级数必须随目录深度同步修正（搬迁最易踩的静默坑）。"""

    def test_repo_root_anchors_point_to_repo_root(self):
        from drsr_420 import llm
        from drsr_420.knowledge import rag_kb, tool_runner

        expected = Path(_REPO_ROOT).resolve()
        for label, anchor in (("llm._REPO_ROOT", llm._REPO_ROOT),
                              ("rag_kb._REPO_ROOT", rag_kb._REPO_ROOT),
                              ("tool_runner._ROOT", tool_runner._ROOT)):
            with self.subTest(anchor=label):
                self.assertEqual(Path(anchor).resolve(), expected,
                                 f"{label} 指错了目录——相对配置/子进程 cwd 会静默跑偏")

    def test_standalone_script_bootstrap_resolves_repo_root(self):
        """把脚本文件当脚本加载时（__package__ 为空）`__file__` 兜底必须让包可导入。"""
        import tempfile

        code = (
            "import importlib.util, sys;"
            "name = 'probe_{i}';"
            "spec = importlib.util.spec_from_file_location(name, r'{script}');"
            "module = importlib.util.module_from_spec(spec);"
            # 真实导入会先把模块注册进 sys.modules 再执行（兼容层的模块类替换依赖这一点）
            "sys.modules[name] = module;"
            "spec.loader.exec_module(module);"
            "print('IMPORTED-OK')"
        )
        with tempfile.TemporaryDirectory() as tmp:
            for i, rel in enumerate(_STANDALONE_SCRIPTS):
                script = os.path.join(_REPO_ROOT, rel)
                with self.subTest(script=rel):
                    proc = subprocess.run(
                        [sys.executable, "-c",
                         code.format(i=i, script=script).replace("\\", "\\\\")],
                        cwd=tmp, capture_output=True, text=True,
                    )
                    self.assertEqual(proc.returncode, 0,
                                     f"{rel} 独立加载失败（__file__ 兜底级数不对？）:\n"
                                     f"{proc.stderr[-1200:]}")
                    self.assertIn("IMPORTED-OK", proc.stdout)


class EvaluationSubsystemTest(unittest.TestCase):
    """阶段 4：评估执行机制归 evaluation 层，角色文件只留编排。"""

    def test_agent_module_holds_no_execution_mechanism(self):
        src = (Path(_PKG_DIR) / "agents" / "evaluator_agent.py").read_text(encoding="utf-8")
        for forbidden in ("multiprocessing", "Pipe(", "Process(", "exec(program"):
            with self.subTest(token=forbidden):
                self.assertNotIn(
                    forbidden, src,
                    f"agents/evaluator_agent.py 不应再含执行机制痕迹 {forbidden!r}"
                    "（应放在 evaluation/sandbox.py）")

    def test_mechanism_lives_in_evaluation_layer(self):
        from drsr_420.agents import evaluator_agent
        from drsr_420.evaluation import sandbox

        self.assertIs(evaluator_agent.LocalSandbox, sandbox.LocalSandbox)
        self.assertIs(evaluator_agent.Sandbox, sandbox.Sandbox)

    def test_mechanism_symbols_remain_importable_from_agent_module(self):
        """机制符号仍可从角色模块导入（同对象），方便只关心评估流程的调用方。"""
        from drsr_420.agents import evaluator_agent as ea
        from drsr_420.evaluation import sandbox

        for name in ("LocalSandbox", "Sandbox", "_run_evaluation_task",
                     "_sample_residuals", "_sample_to_program", "_calls_ancestor"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(ea, name), f"evaluator_agent 缺少 {name}")
                self.assertIs(getattr(ea, name), getattr(sandbox, name))


if __name__ == "__main__":
    unittest.main()
