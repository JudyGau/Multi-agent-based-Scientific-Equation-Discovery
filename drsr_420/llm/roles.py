"""角色 → LLM 档案解析：**唯一**回答「哪个 Agent 用哪套配置」的地方。

它解决什么问题
==============
重构前，这件事在代码里没有任何一处声明：只能靠 grep ``clone_for_task('...')``
与 ``load_llm_config("xxx.config")`` 反推。更根本的是，旧结构把「角色 → 配置」
做成了一对一内嵌——一份 ``*.config`` 同时装密钥、生成参数**和**一列角色覆盖
（``tasks`` 字段），再用 CLI 字符串选中它。于是 ``tasks`` 只能在同一模型内改参数，
**换不了模型**：仓库里那两个无人读取的 ``llm_explain.config`` / ``llm_summary.config``
就是有人想给特定角色换模型、却发现无路可走的产物；而 ``analysis/explain.py``
干脆硬编码了一个**不存在**的文件名，异常被吞掉后静默写出空的 ``explain.txt``。

三个正交的问题
==============
一份"配置"其实在回答三个寿命与保密等级都不同的问题：

===========  ==========================  ==========  ==========
问题         内容                        保密等级    是否入库
===========  ==========================  ==========  ==========
Q1           连接谁？用哪把钥匙？        高（密钥）  否
Q2           生成参数是什么？            低          随 Q1 同文件
Q3           哪个角色用哪套？覆盖什么？  无          **是**
===========  ==========================  ==========  ==========

本模块只负责 Q3，并把它落成两处声明：

* ``config/agents.config.json``（**入库**，无密钥）：角色 → 档案 + 参数覆盖；
* :data:`BUILTIN_ROLE_PARAMS`（代码常量）：注册表缺失或未声明该角色时的零配置默认值。

Q1+Q2 仍在 ``config/<提供商>_<模型>.config`` 里，受 ``.gitignore`` 的 ``*.config``
规则保护；**扩展名本身就是保密边界**（``.json`` 可入库、``.config`` 不入库）。

重要约束：代码不得解析文件名
============================
档案文件名的唯一权威是文件内的 ``model`` 字段（``ClientFactory`` 会校验
``provider/model`` 格式）；文件名只是给人看的标签。因此文件名写错**不会**导致
静默连错模型，只会让"读不到文件"这类错误尽早暴露。``tests/test_config_roles.py``
用一条扫描护栏锁死这一点：库代码里不得再出现任何 ``*.config`` 字面量。

解析优先级（显式、可打印、可进快照）
====================================
**档案**（每个角色独立解析，自上而下第一个命中者生效）::

    1. --role-config <role>=<file>              （最高：命令行精确覆盖）
    2. 环境变量 DRSR_ROLE_CONFIG_<ROLE>        （容器 / CI / 子进程传递）
    3. config/agents.config.json 的 roles.<role>.config
    4. --llm_config                            （CLI 指定的**默认**档案）
    5. config/agents.config.json 的 default
    6. 内置 DEFAULT_PROFILE                    （最低：保证永不"无配置"）

注意第 3 步高于第 4 步：``--llm_config`` 的语义已收窄为"**默认**档案"（未绑定档案的
角色共用它），否则注册表里的角色绑定会被 IDE 配置里那句 ``--llm_config`` 永久屏蔽
——那正是旧结构下 explain 绑定失效的原因。要强制所有角色用同一档案，用
``--role-config '*'=<file>``（第 1 步，通配）。

**参数**（按角色合并，后者覆盖前者）::

    内置 BUILTIN_ROLE_PARAMS[role]  <  档案文件的 tasks[role]（旧格式，兼容）  <  注册表 roles[role].params
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from drsr_420.llm.factory import (
    DEFAULT_PROFILE,
    PROFILE_SUFFIX,
    TEMPLATE_SUFFIX,
    config_dir,
    locate_config,
)

#: 角色 → 档案 的注册表文件名（入库，不含任何密钥）。
REGISTRY_FILENAME = "agents.config.json"

#: 系统中所有需要 LLM 的**角色**。
#:
#: 前四项与 ``AgentSpec.llm_task`` 一一对应（``tool_caller`` 复用 ``sampling``）；
#: 后两项不是 Agent，但同样调用 LLM：``explain`` 是收尾分析，``summary`` 是 MCP
#: 文献工具 ``read_paper``（跑在子进程里，靠环境变量拿到解析结果）。
TASKS: tuple[str, ...] = (
    "sampling", "analysis", "experience", "residual", "explain", "summary",
)

#: 内置默认角色参数：注册表文件缺失、或注册表未声明该角色时生效。
#:
#: 这些值原先分成两处硬编码——思考强度在档案文件的 ``tasks`` 字段里，
#: 而经验/残差的 temperature 等直接写死在 ``coordinator_agent`` 的
#: ``clone_llm_client(..., temperature=0.0, ...)`` 实参中。现在集中到这一处：
#: 可 review、可被注册表覆盖、且能被快照记录。**默认值刻意与历史行为逐位相同。**
BUILTIN_ROLE_PARAMS: dict[str, dict] = {
    "sampling":   {"reasoning_effort": "low"},
    "analysis":   {"reasoning_effort": "high"},
    "experience": {"reasoning_effort": "high", "temperature": 0.0,
                   "top_p": 1.0, "frequency_penalty": 0.0},
    "residual":   {"reasoning_effort": "high", "temperature": 0.4,
                   "top_p": 0.9, "frequency_penalty": 0.1},
    "explain":    {"reasoning_effort": "high"},
    "summary":    {"reasoning_effort": "high"},
}

#: 角色级环境变量前缀（``DRSR_ROLE_CONFIG_SAMPLING`` 之类）。
ENV_ROLE_PREFIX = "DRSR_ROLE_CONFIG_"

#: 通配角色：``--role-config '*'=<file>`` 表示"所有角色都用这个档案"。
WILDCARD_ROLE = "*"


# ── 档案路径解析（命名约定与定位规则由 factory 拥有）─────────────

def registry_path() -> Path:
    """注册表文件的绝对路径。"""
    return config_dir() / REGISTRY_FILENAME


def profile_path(profile: str) -> Path:
    """把档案引用解析成绝对路径（薄封装，见 :func:`factory.locate_config`）。"""
    return locate_config(profile)


def list_profiles() -> list[str]:
    """配置目录下已有的档案 ID（不含模板与注册表），按名称排序。"""
    directory = config_dir()
    if not directory.is_dir():
        return []
    return sorted(
        path.name[: -len(PROFILE_SUFFIX)]
        for path in directory.glob(f"*{PROFILE_SUFFIX}")
        if not path.name.endswith(TEMPLATE_SUFFIX)
    )


def list_templates() -> list[str]:
    """配置目录下随仓库分发的模板文件名，按名称排序。"""
    directory = config_dir()
    if not directory.is_dir():
        return []
    return sorted(path.name for path in directory.glob(f"*{TEMPLATE_SUFFIX}"))


# ── 注册表 ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RoleEntry:
    """注册表里一个角色的声明。

    Attributes:
        profile: 该角色使用的档案 ID；``None`` 表示"沿用默认档案"。
        params: 该角色的参数覆盖（如思考强度、temperature）。
    """

    profile: str | None = None
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RoleRegistry:
    """``config/agents.config.json`` 的内存表示。

    Attributes:
        path: 实际读取的文件；``None`` 表示文件不存在（全部走内置默认）。
        default: 未绑定档案的角色共用的默认档案 ID。
        roles: 角色 → :class:`RoleEntry`。
    """

    path: Path | None
    default: str
    roles: dict[str, RoleEntry] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "RoleRegistry":
        """空注册表：默认档案与内置参数全部生效（零配置可跑）。"""
        return cls(path=None, default=DEFAULT_PROFILE, roles={})

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "RoleRegistry":
        """读取注册表；文件不存在时返回 :meth:`empty`。

        Raises:
            ValueError: 文件存在但 JSON 非法、``version`` 不支持，或角色名不在
                :data:`TASKS` 里（尽早暴露拼写错误，而不是静默忽略一个角色）。
        """
        target = Path(path) if path is not None else registry_path()
        if not target.is_absolute() and not target.exists():
            alt = config_dir().parent / target
            if alt.exists():
                target = alt
        if not target.exists():
            return cls.empty()

        with open(target, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"{target} 顶层必须是 JSON 对象")

        version = data.get("version", 1)
        if version != 1:
            raise ValueError(f"{target} 的 version={version!r} 不受支持（当前只支持 1）")

        default = data.get("default") or DEFAULT_PROFILE
        raw_roles = data.get("roles") or {}
        if not isinstance(raw_roles, dict):
            raise ValueError(f"{target} 的 roles 必须是对象")

        roles: dict[str, RoleEntry] = {}
        unknown = [name for name in raw_roles if name not in TASKS]
        if unknown:
            raise ValueError(
                f"{target} 声明了未知角色 {unknown}；可用角色：{list(TASKS)}")
        for name, spec in raw_roles.items():
            spec = spec or {}
            if not isinstance(spec, dict):
                raise ValueError(f"{target} 的 roles.{name} 必须是对象")
            params = spec.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError(f"{target} 的 roles.{name}.params 必须是对象")
            roles[name] = RoleEntry(
                profile=spec.get("config") or None,
                params=dict(params),
            )
        return cls(path=target, default=default, roles=roles)

    def entry(self, role: str) -> RoleEntry:
        """该角色的注册表声明；未声明时返回空声明。"""
        return self.roles.get(role, RoleEntry())


# ── 单个角色的解析结果 ─────────────────────────────────────────────

@dataclass(frozen=True)
class RoleResolution:
    """一个角色最终生效的档案与参数（含生效来源，供快照与自检追溯）。"""

    role: str
    profile: str
    config_path: Path
    params: dict
    source: str

    def as_dict(self) -> dict:
        """进 ``config_snapshot.json`` 的紧凑形式（不含密钥）。"""
        return {
            "profile": self.profile,
            "config": str(self.config_path),
            "params": dict(self.params),
            "source": self.source,
        }


# ── 解析 ───────────────────────────────────────────────────────────

def _check_role_name(role: str) -> None:
    """校验角色名（通配符 ``*`` 例外）——拼错必须报错，不能静默忽略一个角色。"""
    if role != WILDCARD_ROLE and role not in TASKS:
        raise ValueError(
            f"未知角色 {role!r}；可用：{list(TASKS)} 或 '{WILDCARD_ROLE}'")


def _parse_cli_overrides(pairs: Sequence[str] | None) -> dict[str, str]:
    """解析 ``--role-config role=file`` 列表。"""
    overrides: dict[str, str] = {}
    for item in pairs or ():
        if "=" not in item:
            raise ValueError(f"--role-config 需要 <role>=<file> 形式，收到 {item!r}")
        role, _, value = item.partition("=")
        role, value = role.strip(), value.strip()
        if not role or not value:
            raise ValueError(f"--role-config 的 role 与 file 都不能为空：{item!r}")
        _check_role_name(role)
        overrides[role] = value
    return overrides


def resolve_roles(
    *,
    registry: RoleRegistry | None = None,
    cli_default: str | None = None,
    cli_overrides: Mapping[str, str] | Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
    tasks: Sequence[str] = TASKS,
) -> dict[str, RoleResolution]:
    """把每个角色解析成 :class:`RoleResolution`（优先级见模块文档）。

    Args:
        registry: 已加载的注册表；``None`` 表示按默认路径加载。
        cli_default: ``--llm_config`` 的值（**默认**档案，不是"全场唯一档案"）。
        cli_overrides: ``--role-config`` 的解析结果（``{role: file}``）或原始
            ``"role=file"`` 字符串列表；``'*'`` 键对所有角色生效。
        environ: 环境变量映射，默认 ``os.environ``。
        tasks: 要解析的角色集合，默认全部 :data:`TASKS`。

    Returns:
        ``{role: RoleResolution}``，键顺序与 ``tasks`` 一致。
    """
    registry = registry if registry is not None else RoleRegistry.load()
    environ = os.environ if environ is None else environ
    if cli_overrides is None:
        cli_overrides = {}
    elif isinstance(cli_overrides, str):
        cli_overrides = _parse_cli_overrides([cli_overrides])
    elif not isinstance(cli_overrides, Mapping):
        cli_overrides = _parse_cli_overrides(list(cli_overrides))
    else:
        # 已是映射形态（测试/内部调用）时也要校验角色名
        for role in cli_overrides:
            _check_role_name(role)

    resolutions: dict[str, RoleResolution] = {}
    for role in tasks:
        entry = registry.entry(role)
        env_key = f"{ENV_ROLE_PREFIX}{role.upper()}"

        # 1) 命令行精确覆盖（含通配），2) 环境变量，3) 注册表角色绑定，
        # 4) --llm_config 默认档案，5) 注册表 default，6) 内置默认
        if role in cli_overrides:
            profile, source = cli_overrides[role], "cli:--role-config"
        elif WILDCARD_ROLE in cli_overrides:
            profile, source = cli_overrides[WILDCARD_ROLE], "cli:--role-config '*'"
        elif environ.get(env_key):
            profile, source = environ[env_key], f"env:{env_key}"
        elif entry.profile:
            profile, source = entry.profile, f"registry:roles.{role}.config"
        elif cli_default:
            profile, source = cli_default, "cli:--llm_config"
        elif registry.default:
            profile, source = registry.default, "registry:default"
        else:
            profile, source = DEFAULT_PROFILE, "builtin"

        resolutions[role] = RoleResolution(
            role=role,
            profile=_profile_id(profile),
            config_path=profile_path(profile),
            params=resolve_params(role, registry=registry),
            source=source,
        )
    return resolutions


def _profile_id(profile: str) -> str:
    """把档案引用压成给人看的 ID（去掉目录与 ``.config`` 后缀）。"""
    raw = (profile or "").strip().replace("\\", "/")
    name = raw.rsplit("/", 1)[-1]
    if name.endswith(PROFILE_SUFFIX):
        name = name[: -len(PROFILE_SUFFIX)]
    return name


def resolve_params(
    role: str,
    *,
    registry: RoleRegistry | None = None,
    profile_config: Mapping[str, Any] | None = None,
) -> dict:
    """解析角色的参数：内置默认 < 档案 ``tasks[role]``（旧格式） < 注册表 ``params``。"""
    registry = registry if registry is not None else RoleRegistry.load()
    merged: dict = dict(BUILTIN_ROLE_PARAMS.get(role, {}))

    legacy_tasks = (profile_config or {}).get("tasks")
    if isinstance(legacy_tasks, dict) and isinstance(legacy_tasks.get(role), dict):
        merged.update({k: v for k, v in legacy_tasks[role].items() if v is not None})

    merged.update({k: v for k, v in registry.entry(role).params.items() if v is not None})
    return merged


# ── 客户端构造与诊断渲染 ───────────────────────────────────────────
# 本模块只回答"应该用哪套配置"（声明 + 解析）。把解析结果变成可调用的客户端在
# ``llm/role_clients.py``（RoleClients），给人看的渲染与自检在
# ``llm/role_diagnostics.py``（describe_roles / check_roles）——两者都属于
# "怎么用 / 怎么看"，与"解析规则"的变化原因不同，因此分文件。
#
# main() 里对 diagnostics 做**函数级**导入：模块级导入会与本模块形成 import 环
# （diagnostics 需要 RoleRegistry / resolve_roles）。


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m drsr_420.llm.roles [--check] [--ping] [--profiles] [--templates]``。"""
    from drsr_420.llm.role_diagnostics import (
        check_roles,
        describe_roles,
        format_ping,
        ping_problems,
        ping_roles,
    )

    parser = argparse.ArgumentParser(
        prog="python -m drsr_420.llm.roles",
        description="打印与校验「角色 → LLM 档案」的绑定关系",
    )
    parser.add_argument("--check", action="store_true",
                        help="离线自检：档案可解析、文件存在、model 合法、密钥可达（不联网）")
    parser.add_argument("--ping", action="store_true",
                        help="连通性自检：每份档案发一次真实请求（联网，消耗少量 token）")
    parser.add_argument("--llm_config", default=None,
                        help="默认档案（未在注册表中绑定档案的角色用它）")
    parser.add_argument("--role-config", action="append", default=None, metavar="ROLE=FILE",
                        help="按角色覆盖档案；ROLE 用 * 表示所有角色（可重复）")
    parser.add_argument("--profiles", action="store_true", help="列出配置目录下已有的档案")
    parser.add_argument("--templates", action="store_true", help="列出随仓库分发的模板")
    parser.add_argument("--registry", default=None, help="注册表路径（默认 config/agents.config.json）")
    args = parser.parse_args(list(argv) if argv is not None else None)

    registry = RoleRegistry.load(args.registry)

    if args.profiles:
        found = list_profiles()
        print(f"已有档案（{len(found)}）: " + (", ".join(found) if found else "(无)"))
    if args.templates:
        found = list_templates()
        print(f"随仓库分发的模板（{len(found)}）: "
              + (", ".join(found) if found else "(无)"))
    if args.profiles or args.templates:
        if not (args.check or args.ping):
            return 0

    print(describe_roles(registry=registry, cli_default=args.llm_config,
                         cli_overrides=args.role_config))

    if not (args.check or args.ping):
        return 0

    if args.check:
        problems = check_roles(registry=registry, cli_default=args.llm_config,
                               cli_overrides=args.role_config)
        if problems:
            print()
            for problem in problems:
                print(f"[WARN] {problem}")
            return 1
        print()
        print("OK: 全部角色都有可用档案（离线校验）")

    if args.ping:
        outcomes = ping_roles(registry=registry, cli_default=args.llm_config,
                              cli_overrides=args.role_config)
        print()
        print(format_ping(outcomes))
        problems = ping_problems(outcomes)
        if problems:
            print()
            for problem in problems:
                print(f"[WARN] {problem}")
            return 1
        print()
        print("OK: 每份档案都真实响应了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
