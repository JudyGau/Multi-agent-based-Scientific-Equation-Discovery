"""DRSR 多 Agent 系统 —— 角色层。

本包把 DRSR 主循环中的协作角色拆分为独立的 Agent 类（一个文件一个角色）；
统一契约（``AgentSpec`` 角色卡 + ``BaseAgent`` 基类）定义在
:mod:`drsr_420.agents.base`，Agent 之间的数据契约定义在
:mod:`drsr_420.agents.messages`。

架构速览（由 SPEC 自动生成——运行 ``python -m drsr_420.agents`` 查看最新版）

    DataAnalyzerAgent  ──初次数据分析──▶ residual_analyze.json (sample_order=0)
    CoordinatorAgent   (每个 Sampler-i 线程一个实例)
      ├─▶ SamplerAgent ──▶ ToolCallerAgent ──▶ LLMClient ──▶ MCP 工具
      ├─▶ EvaluatorAgent ──▶ LocalSandbox ──▶ 多起点 least_squares 拟合
      ├─▶ ExperienceSummarizerAgent ──▶ experiences.json
      └─▶ ResidualAnalyzerAgent ──▶ residual_analyze.json
    收尾：find_best_eq()（工具函数，非 Agent）

导入策略
========
本模块用 PEP 562 惰性导出（``__getattr__``）：模块顶层**不 import** 任何 Agent 子模块。
两个原因：

1. 避免循环导入——Agent 之间互相 import，且 ``core.config`` 反向引用
   ``sampler``/``evaluator`` 的类型；
2. 避免仅仅 ``import drsr_420.agents`` 就拉起 numpy/scipy/沙箱等重依赖。

用法
====
    from drsr_420.agents import CoordinatorAgent        # 惰性导入，等价于从子模块导入
    from drsr_420.agents import agent_specs, describe_architecture, check_contracts

命令行
======
    python -m drsr_420.agents            # 打印组织图
    python -m drsr_420.agents --check    # 契约自检（失败时退出码 1）
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:   # 仅供类型检查器（运行时靠 __getattr__ 惰性解析）
    from drsr_420.agents.base import AgentSpec, BaseAgent

#: 7 个 Agent 角色：公开类名 → 定义它的子模块（顺序即组织图的展示顺序）。
_AGENT_CLASSES: tuple[tuple[str, str], ...] = (
    ("CoordinatorAgent", "drsr_420.agents.coordinator_agent"),
    ("SamplerAgent", "drsr_420.agents.sampler_agent"),
    ("ToolCallerAgent", "drsr_420.agents.tool_caller_agent"),
    ("EvaluatorAgent", "drsr_420.agents.evaluator_agent"),
    ("ExperienceSummarizerAgent", "drsr_420.agents.experience_summarizer_agent"),
    ("ResidualAnalyzerAgent", "drsr_420.agents.residual_analyzer_agent"),
    ("DataAnalyzerAgent", "drsr_420.agents.data_analyzer_agent"),
)

#: 契约层的公开名 → 定义它的子模块。
_BASE_EXPORTS: dict[str, str] = {
    "AgentSpec": "drsr_420.agents.base",
    "BaseAgent": "drsr_420.agents.base",
    "PIPELINE": "drsr_420.agents.base",
    "THREAD_PER_SAMPLER": "drsr_420.agents.base",
    "THREAD_SINGLE_SHOT": "drsr_420.agents.base",
    "validate_registry": "drsr_420.agents.base",
    "render_architecture": "drsr_420.agents.base",
}

_AGENT_EXPORTS: dict[str, str] = {**dict(_AGENT_CLASSES), **_BASE_EXPORTS}

#: 组织图中 Agent 的展示顺序（顶层编排在前，单次角色在后）。
AGENT_ORDER: tuple[str, ...] = (
    "coordinator",
    "sampler",
    "tool_caller",
    "evaluator",
    "experience_summarizer",
    "residual_analyzer",
    "data_analyzer",
)


def __getattr__(name: str):
    """PEP 562 惰性导出：首次访问某个名字时才 import 它所在的子模块。"""
    module_path = _AGENT_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value     # 缓存，后续访问不再走 __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_AGENT_EXPORTS))


def agent_specs() -> dict[str, "AgentSpec"]:
    """返回 ``{key: AgentSpec}``——系统里有哪些 Agent，由代码回答而非文档。

    顺序按 :data:`AGENT_ORDER`；未登记的 key 排在末尾（正常不应出现）。
    """
    specs: dict[str, AgentSpec] = {}
    for class_name, module_path in _AGENT_CLASSES:
        cls = getattr(importlib.import_module(module_path), class_name)
        specs[cls.SPEC.key] = cls.SPEC

    ordered = {key: specs[key] for key in AGENT_ORDER if key in specs}
    ordered.update({key: spec for key, spec in specs.items() if key not in ordered})
    return ordered


def describe_architecture() -> str:
    """渲染多 Agent 组织图（由各 Agent 的 SPEC 生成，不会与代码脱节）。"""
    # 函数内导入：模块级 `from ... import` 会破坏本包"顶层不 import 子模块"的约定，
    # 且 PEP 562 的 __getattr__ 不参与函数内的全局名解析（那样会 NameError）。
    from drsr_420.agents.base import render_architecture

    return render_architecture(agent_specs(), AGENT_ORDER)


def check_contracts() -> list[str]:
    """校验 7 个 Agent 的契约与互相引用，返回问题列表（空 = 通过）。"""
    from drsr_420.agents.base import validate_registry

    specs = agent_specs()
    problems = validate_registry(specs)

    missing = [key for key in AGENT_ORDER if key not in specs]
    if missing:
        problems.append(f"AGENT_ORDER 声明的 Agent 未注册: {missing}")
    unexpected = [key for key in specs if key not in AGENT_ORDER]
    if unexpected:
        problems.append(f"发现未登记进 AGENT_ORDER 的 Agent: {unexpected}")
    return problems
