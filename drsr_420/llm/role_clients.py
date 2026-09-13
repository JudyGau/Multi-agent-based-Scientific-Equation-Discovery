"""按角色提供已参数化的 LLM 客户端（:class:`RoleClients`）。

角色归属
========
``llm`` 层的**运行时**部件：把 :mod:`drsr_420.llm.roles` 解析出的「角色 → 档案 +
参数」变成真正可调用的客户端实例。``roles.py`` 只回答"应该用哪套"，本模块负责
"把它造出来"，这样"声明什么"与"怎么连"不会混在一个文件里。

为什么不直接每处 ``ClientFactory.from_config``
============================================
两个必须集中的理由：

1. **kwargs 是实例可变状态**。多个角色共用一个 ``LLMClient`` 时，设置
   temperature / reasoning_effort 会互相覆盖——历史上真的发生过：采样/经验/残差
   三个用途的生成参数最后全变成 ``temperature=0.4``。所以 :meth:`RoleClients.get`
   每次返回**独立克隆**（协调者的每个 Sampler 线程各拿一份，互不干扰）。
2. **同一档案只建一个底层客户端**。6 个角色常常共用同一份档案，逐角色新建会造出
   6 套连接；这里按档案路径缓存底层实例，差异只体现在克隆时的参数注入上。

按需构造（为什么不再一启动就连）
================================
``from_registry`` 只做**解析**（角色 → 档案 → 参数合并规则），不读档案、不建连接；
真正的构造发生在第一次 :meth:`get` 时，并按档案缓存。理由是实验启动不该被"这次根本
用不到的角色"拖死：``summary`` 绑定了一份密钥没填好的档案，不该让一次完全不读文献的
实验连启动都起不来。改成按需之后：

* 解析期错误（角色名拼错、``--role-config`` 格式错）仍然**立即致命**；
* 单份档案不可用（缺文件、密钥没填、端点写错）只在**真正用到那个角色时**报错，且
  错误信息带上角色名与档案路径，指得出该改哪个文件；
* ``--check`` 与 CLI 启动时的告警会**提前**列出同一批问题，所以"推迟"不等于"看不见"
  （``cli.llm_setup.build_role_clients`` 启动时跑一遍结构自检并逐条告警、但不退出）。

对外契约
========
* :meth:`RoleClients.from_registry` —— 生产路径（读注册表 + CLI 覆盖，惰性构造）；
* :meth:`RoleClients.single` / ``base_client=`` —— 测试与"只有一份客户端"的旧调用方；
* :meth:`RoleClients.get(role) <RoleClients.get>` —— 取该角色的独立客户端；
* :meth:`RoleClients.describe` / :meth:`RoleClients.resolved_profiles` ——
  分别供 ``config_snapshot.json`` 与子进程环境变量传递。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from drsr_420.llm.factory import ClientFactory, load_llm_config
from drsr_420.llm.roles import (
    RoleRegistry,
    RoleResolution,
    resolve_params,
    resolve_roles,
)


def load_role_config(role: str, **kwargs) -> dict:
    """加载某角色的档案内容（dict）。解析优先级同 :func:`roles.resolve_roles`。

    供**不知道客户端该从哪来**的模块使用——典型是跑在 MCP 子进程里的
    ``read_paper``：它只能靠环境变量 ``DRSR_ROLE_CONFIG_SUMMARY`` 拿到父进程
    解析出的档案。
    """
    return load_llm_config(str(resolve_roles(**kwargs)[role].config_path))


@dataclass
class RoleClients:
    """按角色提供已参数化的 LLM 客户端（按档案缓存 + 每角色独立克隆 + 按需构造）。"""

    resolutions: dict[str, RoleResolution]
    _registry: RoleRegistry | None = None
    _given_base: Any | None = None
    _roles_by_profile: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # 下面两个是**缓存**：首次 get() 某个角色时才填充（见 _base_for_role）
    _base_by_profile: dict[str, Any] = field(default_factory=dict)
    _params_by_role: dict[str, dict] = field(default_factory=dict)

    # -- 构造（只解析，不连接）-----------------------------------------
    @classmethod
    def from_registry(
        cls,
        registry: RoleRegistry | None = None,
        *,
        cli_default: str | None = None,
        cli_overrides: Mapping[str, str] | Sequence[str] | None = None,
        environ: Mapping[str, str] | None = None,
        base_client: Any | None = None,
    ) -> "RoleClients":
        """生产路径：按注册表与 CLI 覆盖逐角色解析。**不构造任何客户端。**

        Args:
            base_client: 给定一个**已构造好**的客户端时，所有角色共用它（只按角色
                注入参数）。测试与"只有一个客户端"的旧调用方走这条路。
        """
        registry = registry if registry is not None else RoleRegistry.load()
        resolutions = resolve_roles(
            registry=registry, cli_default=cli_default,
            cli_overrides=cli_overrides, environ=environ)

        roles_by_profile: dict[str, list[str]] = {}
        for role, resolution in resolutions.items():
            roles_by_profile.setdefault(str(resolution.config_path), []).append(role)

        return cls(
            resolutions=resolutions,
            _registry=registry,
            _given_base=base_client,
            _roles_by_profile={key: tuple(roles)
                               for key, roles in roles_by_profile.items()},
        )

    @classmethod
    def single(cls, client: Any | None, **kwargs) -> "RoleClients":
        """从单个客户端构造（测试与"只有一份客户端"的旧调用方）。

        ``client`` 为 ``None`` 时所有角色都得到 ``None``（不读档案、不建连接）——
        与重构前 ``llm_client=None`` 的行为一致。
        """
        if client is None:
            return cls(resolutions={}, _base_by_profile={}, _params_by_role={})
        return cls.from_registry(base_client=client, **kwargs)

    # -- 使用 ---------------------------------------------------------
    def get(self, role: str):
        """返回该角色专属的客户端实例（每次调用都是独立克隆，可跨线程使用）。

        首次取某个角色时才构造它所在档案的底层客户端（随后按档案缓存）。

        Returns:
            客户端实例，或 ``None``（:meth:`single` 传 ``None`` 的"无客户端"模式）。

        Raises:
            KeyError: 角色不在 :data:`roles.TASKS` 中——尽早暴露拼写错误。
            RuntimeError: 该角色所在档案读不出或构造失败（信息含角色名、档案路径，
                以及 ClientFactory 给出的可操作原因）。
        """
        if not self.resolutions:
            return None                     # 无客户端模式（single(None)）
        if role not in self.resolutions:
            raise KeyError(f"未知角色 {role!r}；可用：{list(self.resolutions)}")
        base = self._base_for_role(role)
        if base is None:
            return None
        return _clone_for_role(base, role, self._params_by_role.get(role, {}))

    def _base_for_role(self, role: str):
        """取该角色所在档案的底层客户端；首次调用时构造并缓存。"""
        if self._given_base is not None:
            # 单客户端模式：不读档案，参数只按注册表解析（与重构前一致）
            if role not in self._params_by_role:
                self._params_by_role[role] = resolve_params(role, registry=self._registry)
            return self._given_base

        key = str(self.resolutions[role].config_path)
        base = self._base_by_profile.get(key)
        if base is None:
            base = self._build_profile_client(key)
        return base

    def _build_profile_client(self, key: str):
        """构造某档案的底层客户端；同时把共用该档案的角色参数**一次算清**。

        参数合并要读档案里的旧格式 ``tasks`` 字段（见 ``roles.resolve_params``），
        所以这里一次性把共享该档案的所有角色都算好——否则第一次取角色 A 建好客户端后，
        再取角色 B 会拿不到 B 自己的参数。
        """
        roles = self._roles_by_profile.get(key, ())
        hint = "；用 `python -m drsr_420.llm.roles --check` 查看详情"
        try:
            config = load_llm_config(key)
        except Exception as exc:                       # noqa: BLE001
            raise RuntimeError(
                f"角色 {', '.join(roles) or '?'} 的档案无法读取：{key}（{exc}）{hint}"
            ) from exc

        group = {role: resolve_params(role, registry=self._registry,
                                      profile_config=config)
                 for role in roles}
        try:
            base = ClientFactory.from_config(config, task_params=group)
        except Exception as exc:                       # noqa: BLE001
            raise RuntimeError(
                f"角色 {', '.join(roles) or '?'} 的档案无法构造客户端：{key}（{exc}）{hint}"
            ) from exc

        self._base_by_profile[key] = base
        self._params_by_role.update(group)
        return base

    def describe(self) -> dict[str, dict]:
        """``{role: {profile, config, params, source}}``——供 ``config_snapshot.json``。"""
        return {role: res.as_dict() for role, res in self.resolutions.items()}

    def resolved_profiles(self) -> dict[str, str]:
        """``{role: 档案绝对路径}``——供子进程传递环境变量。"""
        return {role: str(res.config_path) for role, res in self.resolutions.items()}


def _clone_for_role(base: Any, role: str, params: Mapping[str, Any]):
    """克隆一个客户端并按角色注入参数（真实客户端走 ``clone_for_task``）。"""
    if base is None:
        return None
    if hasattr(base, "clone_for_task"):
        clone = base.clone_for_task(role)      # 内部已重置统计计数
    else:
        clone = copy.copy(base)               # 替身对象：只求 kwargs 独立
        if hasattr(clone, "kwargs"):
            clone.kwargs = dict(getattr(base, "kwargs", None) or {})
    kwargs = getattr(clone, "kwargs", None)
    if isinstance(kwargs, dict):
        kwargs.update({k: v for k, v in params.items() if v is not None})
    return clone


def build_role_client(role: str, **kwargs):
    """便捷函数：直接构造某角色的客户端（收尾分析等"一次性"用途）。

    因为构造是按需的，这里只会建**该角色所在档案**的客户端，不牵连其它角色——
    所以一份坏档案挡不住收尾分析。
    """
    return RoleClients.from_registry(**kwargs).get(role)
