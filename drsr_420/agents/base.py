"""Agent 契约层：所有 Agent 的「角色卡」与基类。

本模块是「这是一个多 Agent 系统」这件事的**代码化表达**：每个 Agent 通过类属性
``SPEC: AgentSpec`` 声明自己是谁、从谁拿输入、给谁输出、产出什么落盘产物；
``BaseAgent.__init_subclass__`` 在**类定义时**就校验契约完整性——因此"漏声明 SPEC"
或"入口方法改名忘同步"会在 import 阶段直接报错，而不是等运行到某个分支才炸。

设计取舍
========
本基类**不**强制统一业务方法签名。7 个 Agent 的输入输出本质不同，强行塞进
``run(**kwargs)`` 只会把现在显式的签名换成新的隐式契约。统一的只有两件事：

1. 元数据（``SPEC``）——可枚举、可校验、可打印成组织图（``python -m drsr_420.agents``）；
2. 自描述（``describe()`` / ``__repr__``）——日志与调试友好。

Agent 之间的**数据**契约由 :mod:`drsr_420.agents.messages` 承担（阶段 2）。

用法
====
    class MyAgent(BaseAgent):
        SPEC = AgentSpec(
            key="my_agent", role="我的角色", mission="一句话使命",
            entrypoints=("do_work",), upstream=("coordinator",), downstream=(),
            consumes=("input: str",), produces=("output: int",),
            artifacts=(), thread_model=THREAD_PER_SAMPLER, llm_task=None,
        )

        def do_work(self, input: str) -> int: ...

实例化 ``MyAgent()`` 时，若 ``SPEC`` 缺失或声明的入口不存在，定义该类时就会抛
``TypeError``。
"""
from __future__ import annotations

import abc
import dataclasses
from typing import Any, ClassVar, Iterable

#: ``AgentSpec.upstream`` / ``downstream`` 中代表"编排层（非 Agent）"的保留 key。
PIPELINE = "pipeline"

#: 生命周期模型取值：每个 Sampler 线程一个实例（Coordinator 及其下游反思类 Agent）。
THREAD_PER_SAMPLER = "per-sampler-thread"
#: 生命周期模型取值：实验启动/收尾时一次性调用（如初次数据分析）。
THREAD_SINGLE_SHOT = "single-shot"

_THREAD_MODELS = (THREAD_PER_SAMPLER, THREAD_SINGLE_SHOT)


@dataclasses.dataclass(frozen=True)
class AgentSpec:
    """一个 Agent 的「角色卡」：纯声明，不含任何逻辑。

    Attributes:
        key: 稳定标识（小写下划线），上下游通过它互相引用，也用于组织图排序。
        role: 中文角色名（如"评估者"），用于文档、日志与组织图。
        mission: 一句话使命。
        entrypoints: 对外入口方法名，**第一个视为规范入口**；至少一个。
        upstream: 上游调用方的 key；编排层用保留值 :data:`PIPELINE` 表示。
        downstream: 下游被调方的 key（只登记 Agent；对非 Agent 的依赖写进 mission）。
        consumes: 输入契约（人类可读的类型名；阶段 2 起指向 ``agents.messages`` 的类型）。
        produces: 输出契约，与 ``consumes`` 同格式。
        artifacts: 本 Agent 直接落盘、或经 Coordinator 落盘的文件名（相对实验目录）。
        thread_model: 生命周期模型，取值见 :data:`THREAD_PER_SAMPLER` /
            :data:`THREAD_SINGLE_SHOT`。
        llm_task: 所使用的 LLM 客户端副本对应的配置 ``tasks.<name>``（决定思考强度等）；
            ``None`` 表示本 Agent 不直接调用 LLM。
        notes: 补充说明（如并发保护、兼容别名等）。
    """

    key: str
    role: str
    mission: str
    entrypoints: tuple[str, ...]
    upstream: tuple[str, ...]
    downstream: tuple[str, ...]
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    artifacts: tuple[str, ...]
    thread_model: str
    llm_task: str | None = None
    notes: str = ""

    @property
    def canonical_entrypoint(self) -> str:
        """规范入口方法名（``entrypoints`` 的第一项）。"""
        return self.entrypoints[0]


class BaseAgent(abc.ABC):
    """所有 Agent 的基类：只负责「身份 + 契约校验 + 自描述」。

    子类**必须**声明 ``SPEC``（:class:`AgentSpec`），且 ``SPEC.entrypoints``
    中列出的每个方法都必须真实存在——这两点在类定义时校验。
    """

    #: 子类必须覆盖为 :class:`AgentSpec` 实例。
    SPEC: ClassVar[AgentSpec]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        spec = cls.__dict__.get("SPEC")
        if spec is None:
            raise TypeError(
                f"{cls.__name__} 必须声明类属性 SPEC（AgentSpec）——"
                f"它是该 Agent 的角色卡，缺失会让多 Agent 架构不可枚举。"
            )
        if not isinstance(spec, AgentSpec):
            raise TypeError(
                f"{cls.__name__}.SPEC 必须是 AgentSpec，实际为 {type(spec).__name__}"
            )
        if not spec.key or not spec.key.replace("_", "").isalnum():
            raise TypeError(f"{cls.__name__}.SPEC.key 非法: {spec.key!r}")
        if not spec.entrypoints:
            raise TypeError(f"{cls.__name__}.SPEC.entrypoints 不能为空")
        missing = [name for name in spec.entrypoints if not callable(getattr(cls, name, None))]
        if missing:
            raise TypeError(
                f"{cls.__name__} 在 SPEC.entrypoints 中声明了不存在的入口: {missing}"
            )
        if spec.thread_model not in _THREAD_MODELS:
            raise TypeError(
                f"{cls.__name__}.SPEC.thread_model 非法: {spec.thread_model!r}"
                f"（可选：{', '.join(_THREAD_MODELS)}）"
            )

    # ------------------------------------------------------------------
    # 身份与自描述
    # ------------------------------------------------------------------
    @property
    def spec(self) -> AgentSpec:
        """本实例的角色卡。"""
        return type(self).SPEC

    @property
    def agent_key(self) -> str:
        """本 Agent 的稳定标识。"""
        return type(self).SPEC.key

    def describe(self) -> str:
        """单行角色卡，供组织图、日志与调试使用。"""
        spec = self.spec
        return f"[{spec.key}] {type(self).__name__}（{spec.role}）: {spec.mission}"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} key={self.spec.key!r} role={self.spec.role!r}>"


def validate_registry(specs: dict[str, AgentSpec]) -> list[str]:
    """校验一组角色卡之间的引用一致性，返回问题列表（空 = 通过）。

    检查项：
    1. 上下游引用的 key 必须存在于注册表，或为保留值 :data:`PIPELINE`；
    2. 每个 ``thread_model == THREAD_PER_SAMPLER`` 的 Agent 必须能从协调者沿
       ``downstream`` 可达——否则它虽标称"每线程一个实例"，却没人启动它
       （注意：可达性而非直接子级，``tool_caller`` 是 ``sampler`` 的下游）；
    3. 每个 ``thread_model == THREAD_SINGLE_SHOT`` 的 Agent 必须由编排层驱动
       （``upstream`` 含 :data:`PIPELINE`）；
    4. ``consumes`` / ``produces`` 不得为空（契约必须写清楚）。
    """
    problems: list[str] = []
    known = set(specs)

    for key, spec in specs.items():
        for field_name, refs in (("upstream", spec.upstream), ("downstream", spec.downstream)):
            for ref in refs:
                if ref != PIPELINE and ref not in known:
                    problems.append(f"{key}.{field_name} 引用了未注册的 key: {ref!r}")
        if not spec.consumes:
            problems.append(f"{key}.consumes 为空——输入契约必须声明")
        if not spec.produces:
            problems.append(f"{key}.produces 为空——输出契约必须声明")
        if spec.thread_model == THREAD_SINGLE_SHOT and PIPELINE not in spec.upstream:
            problems.append(
                f"{key} 是 {THREAD_SINGLE_SHOT}，但 upstream 未包含 {PIPELINE!r}"
                f"（单次角色必须由编排层驱动）"
            )

    coordinator = specs.get("coordinator")
    if coordinator is not None:
        reachable: set[str] = set()
        stack = ["coordinator"]
        while stack:
            key = stack.pop()
            if key in reachable or key not in specs:
                continue
            reachable.add(key)
            stack.extend(specs[key].downstream)
        for key, spec in specs.items():
            if spec.thread_model == THREAD_PER_SAMPLER and key not in reachable:
                problems.append(
                    f"{key} 是 {THREAD_PER_SAMPLER}，但从 coordinator 沿 downstream 不可达"
                    f"——没有任何角色会启动它"
                )

    return problems


def render_architecture(specs: dict[str, AgentSpec], order: Iterable[str] | None = None) -> str:
    """把角色卡渲染成 ASCII 组织图（由 SPEC 生成，不会与代码脱节）。"""
    keys = list(order) if order is not None else list(specs)
    keys = [k for k in keys if k in specs]

    def _line(key: str, prefix: str, connector: str) -> str:
        spec = specs[key]
        llm = f"  LLM={spec.llm_task}" if spec.llm_task else ""
        head = f"{prefix}{connector}[{key}] {spec.role}  ({spec.thread_model}{llm})"
        indent = " " * len(prefix + connector)
        return f"{head}\n{indent}{spec.mission}"

    def _children(key: str) -> list[str]:
        return [c for c in specs[key].downstream if c in specs]

    bar = "═" * 72
    thin = "─" * 72
    out = [f"DRSR 多 Agent 系统（{len(specs)} 个角色）", bar]

    roots = [k for k in keys
             if PIPELINE in specs[k].upstream and specs[k].thread_model == THREAD_PER_SAMPLER]
    seen: set[str] = set()

    def emit(key: str, prefix: str, connector: str, is_root: bool = False) -> None:
        if key in seen:
            return
        seen.add(key)
        out.append(_line(key, prefix, connector))
        children = _children(key)
        if not children:
            return
        # 子级缩进：根节点不缩进；非根节点按其是否为最后一个兄弟决定画竖线还是留白
        child_prefix = prefix + (
            "" if is_root else ("    " if connector.startswith("└") else "│   ")
        )
        for i, child in enumerate(children):
            last = i == len(children) - 1
            emit(child, child_prefix, "└─→ " if last else "├─→ ")

    out.append("编排层（非 Agent）pipeline.main()：初始化共享记忆 → 启动 Sampler-i 线程")
    out.append(thin)
    for root in roots:
        emit(root, "", "", is_root=True)
    out.append(thin)

    others = [k for k in keys if k not in seen]
    if others:
        out.append("编排层直接驱动的单次角色（非 per-sampler 线程）")
        for key in others:
            emit(key, "", "")
        out.append(thin)

    out.append("收尾（非 Agent）：find_best_eq() —— 参数拟合 + 物理解释")
    return "\n".join(out)
