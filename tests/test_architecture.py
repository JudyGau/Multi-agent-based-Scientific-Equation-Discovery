"""架构护栏测试：守护"结构契约"，而非功能行为。

本文件随 `docs/REFACTOR_PLAN.md` 的分阶段重构逐步收紧。

阶段 0：包内模块可单独导入、`agents/__init__.py` 顶层不 eager import、
Agent 层不反向依赖编排层。

阶段 1：7 个 Agent 都继承 `BaseAgent` 并声明自洽的 `SPEC`（角色卡）。

阶段 3：8 层目录结构、**依赖方向**硬约束、`__file__` 路径锚点。

阶段 4：评估机制归 `evaluation` 层，角色文件只留编排。

阶段 6：兼容层清退——旧路径必须导入失败、顶层只留 `__init__.py`、
库代码与测试都不得再 import 旧路径。

阶段 7：仓库根的 `main.py` / `llm.py` 也清退——入口统一为
`python -m drsr_420.cli.main`（IDE 运行配置与 `.sh` 同步改为模块方式），
`llm` 的公开 API 由 `drsr_420.llm` 直接提供；仓库根不再贡献任何可导入模块。

阶段 10：包结构按"领域"收敛——新增 `equations`（公式领域模型）与 `evidence`
（可验证证据）；原 `evaluation` 拆为 `execution`（拟合/沙箱机制）+ `evidence`；
原 `analysis` 拆为 `reporting`（收尾报告）+ `equations`（表达式工具箱）。
目的是让代码结构与论文的两条主线（证据层 / 报告）一眼对应。

基线（阶段 0 记录）：测试数 217 ｜ drsr_420+tests 源码 9,375 行 ｜
drsr_420/ 顶层 .py 21 个（15 实现 + 6 兼容 shim）｜ 子包 2 个 ｜
最长文件 llm.py 773 行 ｜ Agent 7 个
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG_DIR = os.path.join(_REPO_ROOT, "drsr_420")
_AGENTS_DIR = os.path.join(_PKG_DIR, "agents")

# ── 阶段 3：分层结构（阶段 10：新增 equations / evidence，拆分 evaluation、analysis）──
#: 层的展示顺序即"自底向上"，越靠后越上层。
_LAYERS = ("core", "equations", "llm", "execution", "evidence", "knowledge",
           "reporting", "agents", "runtime", "cli")

#: 每一层允许 import 的层（含自身）；其余一律视为越界/倒置。
_ALLOWED_LAYER_DEPS = {
    "core": set(),                                          # 最底层（领域无关基础设施）
    "equations": {"core"},                                  # 公式领域模型（解析/求值/样本词汇/病理内核）
    "llm": {"core"},                                        # token 统计下沉到 core.llm_stats
    "execution": {"core", "equations"},                     # 评估执行机制：拟合打分 / 沙箱 / 加速
    "evidence": {"core", "equations", "execution"},         # 可验证证据：事实表 / 架构地形 / 未试邻域
    "knowledge": {"llm"},                                   # RAG 与 MCP 工具只依赖 LLM 接入
    "reporting": {"core", "equations", "llm", "knowledge"}, # 收尾分析与报告装配
    "agents": {"core", "equations", "llm", "execution", "evidence", "knowledge"},
    "runtime": {"core", "agents", "reporting", "knowledge"},
    "cli": {"core", "llm", "execution", "agents", "runtime"},
}

#: 顶层不再有任何实现模块，也不再有实现例外白名单（阶段 6 清退后只剩 __init__.py）。
#: 新代码一律进分层子包；确需新增一层请同时更新 _LAYERS 与 _ALLOWED_LAYER_DEPS。

# 规范模块 → 该模块内的公开类
_CANONICAL_AGENTS = {
    "drsr_420.agents.coordinator_agent": ["CoordinatorAgent"],
    "drsr_420.agents.sampler_agent": ["SamplerAgent", "SamplingBackend", "LLM"],
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
    # core → equations（阶段 10：公式领域模型独立成层）
    "drsr_420.core.range_check": "drsr_420.equations.pathology",
    "drsr_420.core.sample_header": "drsr_420.equations.header",
    "drsr_420.core.sample_records": "drsr_420.equations.records",
    # evaluation → execution / evidence / equations（阶段 10：拆"执行机制"与"可验证证据"）
    "drsr_420.evaluate_on_problems": "drsr_420.execution.problems",
    "drsr_420.evaluator_accelerate": "drsr_420.execution.accelerate",
    "drsr_420.evaluation": "drsr_420.execution",
    "drsr_420.evaluation.problems": "drsr_420.execution.problems",
    "drsr_420.evaluation.sandbox": "drsr_420.execution.sandbox",
    "drsr_420.evaluation.accelerate": "drsr_420.execution.accelerate",
    "drsr_420.evaluation.data_facts": "drsr_420.evidence.facts",
    "drsr_420.evaluation.architecture_facts": "drsr_420.evidence.terrain",
    "drsr_420.evaluation.skeleton_gen": "drsr_420.evidence.neighborhood",
    "drsr_420.evaluation.equation_text": "drsr_420.equations.text_algebra",
    # knowledge
    "drsr_420.rag_kb": "drsr_420.knowledge.rag_kb",
    "drsr_420.rag_build": "drsr_420.knowledge.rag_build",
    "drsr_420.tool_runner": "drsr_420.knowledge.tool_runner",
    "drsr_420.tools": "drsr_420.knowledge.tools",
    "drsr_420.tools.mcp_server": "drsr_420.knowledge.tools.mcp_server",
    "drsr_420.tools.search_paper": "drsr_420.knowledge.tools.search_paper",
    "drsr_420.tools.read_paper": "drsr_420.knowledge.tools.read_paper",
    "drsr_420.tools.tools_description": "drsr_420.llm.tools_schema",
    # analysis → reporting / equations（阶段 10：表达式工具箱归 equations，收尾报告归 reporting）
    "drsr_420.find_best_eq": "drsr_420.reporting.find_best_eq",
    "drsr_420.sensitivity_prune": "drsr_420.reporting.pruning.sensitivity",
    "drsr_420.analysis": "drsr_420.reporting",
    "drsr_420.analysis.expr_parse": "drsr_420.equations.parse",
    "drsr_420.analysis.expr_numeric": "drsr_420.equations.numeric",
    "drsr_420.analysis.expr_evaluation": "drsr_420.equations.evaluator",
    "drsr_420.analysis.expr_curves": "drsr_420.reporting.curves",
    "drsr_420.analysis.expr_viz": "drsr_420.reporting.viz",
    "drsr_420.analysis.sensitivity_prune": "drsr_420.reporting.pruning.sensitivity",
    "drsr_420.analysis.prune_eval": "drsr_420.reporting.pruning.verdict",
    "drsr_420.analysis.prune_stats": "drsr_420.reporting.pruning.stats",
    "drsr_420.analysis.prune_demo": "drsr_420.reporting.pruning.demo",
    "drsr_420.analysis.holdout": "drsr_420.reporting.generalization.holdout",
    "drsr_420.analysis.loo": "drsr_420.reporting.generalization.loo",
    "drsr_420.analysis.report_sections": "drsr_420.reporting.report_sections",
    "drsr_420.analysis.references": "drsr_420.reporting.references",
    "drsr_420.analysis.md_sections": "drsr_420.reporting.md_sections",
    "drsr_420.analysis.data_io": "drsr_420.reporting.data_io",
    "drsr_420.analysis.progress_curve": "drsr_420.reporting.progress_curve",
    "drsr_420.analysis.explain": "drsr_420.reporting.explain",
    "drsr_420.analysis.find_best_eq": "drsr_420.reporting.find_best_eq",
    # analysis 层内重命名：prune_report 名实不符（当时兼当全层数据工具库），
    # 数据工具拆到 data_io / expr_numeric 后改名为 prune_eval。旧路径不得复活。
    "drsr_420.analysis.prune_report": "drsr_420.reporting.pruning.verdict",
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


#: 仓库根目录曾有过的两个顶层模块：命令行入口 `main.py` 与 LLM 客户端 `llm.py`。
#: 阶段 7 把两者也清退了（此前只是被降级成转发层）。
_REMOVED_ROOT_MODULES = ("main", "llm")


class RootEntrypointRemovalTest(unittest.TestCase):
    """阶段 7：仓库根不再有可导入的模块，入口统一走 `-m drsr_420.cli.main`。

    与 `LegacyPathRemovalTest` 的分工：那个守"包内旧路径"，这个守"包外的顶层模块"。
    根目录的 `main.py`（入口）与 `llm.py`（LLM 客户端）都已清退，根目录不再是
    `sys.path` 的一部分，因此包自包含、也不会被 PyPI 上的同名 `llm` 包劫持。
    """

    def test_root_holds_no_python_module(self):
        leftovers = sorted(path.name for path in Path(_REPO_ROOT).glob("*.py"))
        self.assertEqual(
            leftovers, [],
            "仓库根不应再有 .py 模块：命令行入口用 `python -m drsr_420.cli.main`，"
            "库代码一律进 drsr_420/ 的分层子包")

    def test_removed_root_modules_resolve_outside_repo_root(self):
        """`import main` / `import llm` 不得解析回仓库根。

        只看"解析结果是否落在仓库根"而不是"是否 ModuleNotFoundError"：环境里若恰好
        装了同名的 PyPI 包（`llm` 就是这种情况），断言前者依然成立。
        """
        root = Path(_REPO_ROOT).resolve()
        for name in _REMOVED_ROOT_MODULES:
            with self.subTest(module=name):
                spec = importlib.util.find_spec(name)
                if spec is None or spec.origin is None:
                    continue          # 压根解析不到，正是期望
                origin = Path(spec.origin).resolve()
                self.assertNotEqual(
                    origin.parent, root,
                    f"`{name}` 仍从仓库根解析（{origin}）——顶层模块已清退")

    def test_no_source_imports_root_modules(self):
        offenders: list[str] = []
        targets = (list(Path(_PKG_DIR).rglob("*.py"))
                   + list(Path(_REPO_ROOT, "tests").rglob("*.py")))
        for path in targets:
            if "__pycache__" in path.parts:
                continue
            for module_name in _all_imports(str(path)):
                if module_name.split(".")[0] in _REMOVED_ROOT_MODULES:
                    offenders.append(
                        f"{path.relative_to(_REPO_ROOT)} -> {module_name}")
        self.assertEqual(
            sorted(set(offenders)), [],
            "这些 import 指向已删除的仓库根模块（运行时会 ImportError 或解析到别的库）:\n"
            + "\n".join(sorted(set(offenders))))


class RenamedSymbolAliasTest(unittest.TestCase):
    """重命名后的旧名必须是**同一对象**（不是副本），且定义处是规范名。

    与 :class:`LegacyPathRemovalTest` 的分工：那个管"模块路径"（必须彻底消失），
    这个管"类名"（保留别名但要锁死同一性与规范名）。区分的理由是使用方式不同——
    模块路径靠 import 语句引用，改名后旧写法必然 ImportError（应当立刻炸）；
    而 ``llm_class`` / ``Type[...]`` 这类**类型注解**与外部 isinstance 检查引用类名，
    别名留着成本为零，但必须防"复制粘贴式分叉"（副本会让 ``isinstance`` 悄悄失效）。
    """

    def test_sampling_backend_is_canonical_and_alias_is_identical(self):
        from drsr_420.agents import sampler_agent

        self.assertIs(sampler_agent.LLM, sampler_agent.SamplingBackend,
                      "LLM 必须是 SamplingBackend 的同一对象（别名），不是副本")
        self.assertEqual(sampler_agent.SamplingBackend.__name__, "SamplingBackend",
                         "规范名（定义处）应为 SamplingBackend")
        self.assertTrue(issubclass(sampler_agent.SamplerAgent,
                                   sampler_agent.SamplingBackend))

    def test_no_llm_abstract_base_is_defined_anywhere(self):
        """全局只应有一个采样后端抽象，且叫 SamplingBackend。

        这条堵的是"改名不彻底"：若别处又冒出一个 ``class LLM``，同名歧义就回来了
        （``LLMClient`` 与 ``LLM`` 曾让"LLM 指哪一个"每次都要重新推断）。
        """
        offenders: list[str] = []
        for path in sorted(Path(_PKG_DIR).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "LLM":
                    offenders.append(f"{path.relative_to(_PKG_DIR)}:{node.lineno}")
        self.assertEqual(offenders, [],
                         f"不允许再定义名为 LLM 的类（应叫 SamplingBackend）: {offenders}")


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
        "main",          # 仓库根入口已删除（阶段 7）；保留为墓碑，防旧写法复活
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


class IntraLayerPrivateImportTest(unittest.TestCase):
    """**层内**不得跨模块引用私有名（下划线开头）。

    ``LayerDependencyTest`` 管的是层与层之间的方向（``agents`` 不能反向依赖
    ``runtime``）；本类管**同一层内部**的接口纪律：``analysis.holdout`` 曾经
    ``from ...prune_eval import _warn_once``（当时那个模块叫 ``prune_report``）——
    下划线是"别碰我"的信号，一旦跨模块引用，等于**声明了一个不存在于任何文档、
    ``__all__`` 或类型里的接口**。这类引用会随模块拆分/改名静默失效，也让"这个私有名
    到底能不能改"变得无从判断。

    实测的成因（本护栏要堵的正是它）：把不同职责的函数塞进同一个模块后，其他模块
    只能靠私有名取用；正确做法是把它们提升为公开名或移到共享模块（本次重构已把
    ``_warn_once``/``_REPO_ROOT`` 等提为 ``warn_once``/``REPO_ROOT``）。

    豁免只允许两条白名单（附理由），新增豁免必须在同一处写明为什么它是契约而非巧合。
    """

    #: ``(文件相对路径, 被导入的模块, 私有名)`` → 豁免理由。
    ALLOWED = {
        ("reporting/holdout.py", "drsr_420.reporting.generalization.loo", "_loo_fit"):
            "holdout 只是把 LOO 的名字转发回旧路径（对象同一），_loo_fit 是 loo 的内部实现",
    }

    def test_no_cross_module_private_imports_within_a_layer(self):
        offenders: list[str] = []
        for layer in _LAYERS:
            for path in sorted((Path(_PKG_DIR) / layer).rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
                rel = f"{layer}/{path.name}"
                for node in ast.walk(tree):
                    if not (isinstance(node, ast.ImportFrom) and node.module):
                        continue
                    if _layer_of(node.module) != layer:
                        continue          # 跨层由 LayerDependencyTest 管
                    if node.module == f"drsr_420.{layer}.{path.stem}":
                        continue          # 自己 import 自己不算
                    for alias in node.names:
                        if not (alias.name.startswith("_")
                                and not alias.name.startswith("__")):
                            continue
                        if (rel, node.module, alias.name) in self.ALLOWED:
                            continue
                        offenders.append(
                            f"{rel}:{node.lineno} -> {node.module}.{alias.name}")
        self.assertEqual(
            sorted(set(offenders)), [],
            "同一层内跨模块引用了私有名——请把它提升为公开名，或移到共享模块"
            "（确属契约的加进 IntraLayerPrivateImportTest.ALLOWED 并写明理由）:\n"
            + "\n".join(sorted(set(offenders))))


class PathAnchorTest(unittest.TestCase):
    """`.parent` 级数必须随目录深度同步修正（搬迁最易踩的静默坑）。"""

    def test_repo_root_anchors_point_to_repo_root(self):
        from drsr_420 import llm
        from drsr_420.knowledge import rag_kb, tool_runner

        expected = Path(_REPO_ROOT).resolve()
        for label, anchor in (("llm._REPO_ROOT", llm._REPO_ROOT),
                              ("rag_kb.REPO_ROOT", rag_kb.REPO_ROOT),
                              ("tool_runner._ROOT", tool_runner._ROOT)):
            with self.subTest(anchor=label):
                self.assertEqual(Path(anchor).resolve(), expected,
                                 f"{label} 指错了目录——相对配置/子进程 cwd 会静默跑偏")


class EvaluationSubsystemTest(unittest.TestCase):
    """阶段 4：评估执行机制归 execution 层，角色文件只留编排。"""

    def test_agent_module_holds_no_execution_mechanism(self):
        src = (Path(_PKG_DIR) / "agents" / "evaluator_agent.py").read_text(encoding="utf-8")
        for forbidden in ("multiprocessing", "Pipe(", "Process(", "exec(program"):
            with self.subTest(token=forbidden):
                self.assertNotIn(
                    forbidden, src,
                    f"agents/evaluator_agent.py 不应再含执行机制痕迹 {forbidden!r}"
                    "（应放在 execution/sandbox.py）")

    def test_mechanism_lives_in_execution_layer(self):
        from drsr_420.agents import evaluator_agent
        from drsr_420.execution import sandbox

        self.assertIs(evaluator_agent.LocalSandbox, sandbox.LocalSandbox)
        self.assertIs(evaluator_agent.Sandbox, sandbox.Sandbox)

    def test_mechanism_symbols_remain_importable_from_agent_module(self):
        """机制符号仍可从角色模块导入（同对象），方便只关心评估流程的调用方。"""
        from drsr_420.agents import evaluator_agent as ea
        from drsr_420.execution import sandbox

        for name in ("LocalSandbox", "Sandbox", "_run_evaluation_task",
                     "_sample_residuals", "_sample_to_program", "_calls_ancestor"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(ea, name), f"evaluator_agent 缺少 {name}")
                self.assertIs(getattr(ea, name), getattr(sandbox, name))


if __name__ == "__main__":
    unittest.main()
