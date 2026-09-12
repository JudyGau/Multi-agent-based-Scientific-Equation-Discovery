"""架构护栏测试：守护"结构契约"，而非功能行为。

本文件随 `docs/REFACTOR_PLAN.md` 的分阶段重构逐步收紧。

阶段 0：旧路径与规范模块必须是**同一对象**（防 shim 分叉）、包内模块可单独导入、
`agents/__init__.py` 顶层不 eager import、Agent 层不反向依赖编排层。

阶段 1：7 个 Agent 都继承 `BaseAgent` 并声明自洽的 `SPEC`（角色卡）。

阶段 3：8 层目录结构、**依赖方向**硬约束、`__file__` 路径锚点、
顶层文件只能是兼容层。

基线（阶段 0 记录，阶段 5 对照）：
    测试数 217 ｜ drsr_420+tests 源码 9,375 行 ｜ drsr_420/ 顶层 .py 21 个
    （15 实现 + 6 兼容 shim）｜ 子包 2 个 ｜ 最长文件 llm.py 773 行 ｜ Agent 7 个
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

#: 顶层仍属"实现"而非兼容层的文件（阶段 5 计划删除 parallel_bfgs.py）。
_TOP_LEVEL_IMPLEMENTATIONS: set[str] = {"parallel_bfgs.py"}

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


# ── 阶段 3：兼容层对照表（旧模块 → 规范模块 → 抽检名字）─────────────────
_LAYER_SHIMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "drsr_420.buffer": ("drsr_420.core.buffer", ("ExperienceBuffer", "Prompt")),
    "drsr_420.code_manipulation": (
        "drsr_420.core.code_manipulation", ("Program", "text_to_program")),
    "drsr_420.config": ("drsr_420.core.config", ("Config", "ClassConfig")),
    "drsr_420.console": ("drsr_420.core.console", ("print_block",)),
    "drsr_420.profile": ("drsr_420.core.profile", ("Profiler",)),
    "drsr_420.prompt_config": ("drsr_420.core.prompt_config", ("PromptContext",)),
    "drsr_420.rag_kb": ("drsr_420.knowledge.rag_kb", ("get_kb", "chunk_text")),
    "drsr_420.rag_build": ("drsr_420.knowledge.rag_build", ("main",)),
    "drsr_420.tool_runner": ("drsr_420.knowledge.tool_runner", ("mcp_call_tool",)),
    "drsr_420.tools.mcp_server": (
        "drsr_420.knowledge.tools.mcp_server", ("mcp", "main", "search_paper")),
    "drsr_420.tools.search_paper": (
        "drsr_420.knowledge.tools.search_paper", ("search_paper",)),
    "drsr_420.tools.read_paper": (
        "drsr_420.knowledge.tools.read_paper", ("read_paper",)),
    "drsr_420.tools.tools_description": ("drsr_420.llm.tools_schema", ("tools",)),
    "drsr_420.find_best_eq": ("drsr_420.analysis.find_best_eq", ("find_best_eq",)),
    "drsr_420.sensitivity_prune": (
        "drsr_420.analysis.sensitivity_prune", ("sensitivity_prune",)),
    "drsr_420.pipeline": ("drsr_420.runtime.pipeline", ("main",)),
    "drsr_420.evaluate_on_problems": (
        "drsr_420.evaluation.problems", ("evaluate", "MAX_NPARAMS")),
    "drsr_420.evaluator_accelerate": (
        "drsr_420.evaluation.accelerate", ("try_add_numba_decorator",)),
    "llm": ("drsr_420.llm", ("LLMClient", "ClientFactory", "tools")),
    "main": ("drsr_420.cli.main", ("main",)),
}

#: 独立运行时需要 `__file__` 兜底插入仓库根的脚本（搬迁后深度必须同步修正）
_STANDALONE_SCRIPTS = (
    "drsr_420/knowledge/tools/read_paper.py",   # 新位置：deepest，parents[3]
    "drsr_420/tools/read_paper.py",             # 兼容层：parents[2]
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


class LayerShimIdentityTest(unittest.TestCase):
    """19 个分层兼容层（+ 根 main/llm）必须都是同一对象的转发，而非副本。"""

    def test_layer_shims_alias_canonical_objects(self):
        for legacy_module, (canonical_module, names) in _LAYER_SHIMS.items():
            legacy = importlib.import_module(legacy_module)
            canonical = importlib.import_module(canonical_module)
            for name in names:
                with self.subTest(shim=legacy_module, name=name):
                    self.assertTrue(hasattr(legacy, name),
                                    f"{legacy_module}.{name} 缺失（兼容层被破坏）")
                    self.assertIs(getattr(legacy, name), getattr(canonical, name),
                                  f"{legacy_module}.{name} 与 {canonical_module}.{name} 不是同一对象")


class LayerLayoutTest(unittest.TestCase):
    """drsr_420/ 顶层只应剩兼容层；实现都在分层子包里。"""

    def test_all_layers_exist_with_init(self):
        for layer in _LAYERS:
            init = Path(_PKG_DIR) / layer / "__init__.py"
            with self.subTest(layer=layer):
                self.assertTrue(init.exists(), f"缺少 {layer}/__init__.py")

    def test_package_has_version(self):
        import drsr_420
        self.assertTrue(getattr(drsr_420, "__version__", None),
                        "drsr_420 应声明 __version__")

    def test_top_level_modules_are_compat_shims(self):
        offenders = []
        for path in sorted(Path(_PKG_DIR).glob("*.py")):
            if path.name == "__init__.py" or path.name in _TOP_LEVEL_IMPLEMENTATIONS:
                continue
            text = path.read_text(encoding="utf-8")
            is_shim = ("兼容层" in text) or ("__getattr__" in text and "_impl" in text)
            if not is_shim:
                offenders.append(path.name)
        self.assertEqual(
            offenders, [],
            "实现文件不应留在 drsr_420/ 顶层（应放进分层子包；确需保留请加入 "
            f"_TOP_LEVEL_IMPLEMENTATIONS 并说明原因）: {offenders}")

    def test_top_level_implementation_whitelist_is_documented(self):
        """白名单必须保持"有据可查"：每个例外都要有理由（阶段 5 清空）。"""
        self.assertLessEqual(
            _TOP_LEVEL_IMPLEMENTATIONS, {"parallel_bfgs.py"},
            "顶层实现例外只允许在阶段 5 删除前存在（parallel_bfgs.py 已确认零引用）")


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
        """历史导入路径必须继续可用（测试与外部脚本直接用这些名字）。"""
        import drsr_420.agents.evaluator_agent as ea
        import drsr_420.evaluator as legacy

        for name in ("LocalSandbox", "Sandbox", "_run_evaluation_task",
                     "_sample_residuals", "_sample_to_program", "_calls_ancestor"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(ea, name), f"evaluator_agent 缺少 {name}")
                self.assertIs(getattr(legacy, name), getattr(ea, name))


if __name__ == "__main__":
    unittest.main()
