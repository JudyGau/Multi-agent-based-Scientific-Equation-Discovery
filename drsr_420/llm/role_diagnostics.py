"""角色配置的诊断与渲染：`python -m drsr_420.llm.roles` 的输出部分。

角色归属
========
``llm`` 层的**只读诊断**部件：把 :func:`drsr_420.llm.roles.resolve_roles` 的结果
渲染成人看的表格，或逐项校验"这套绑定到底能不能跑起来"。

为什么单独成文件
================
渲染与校验是"给人看/给 CI 看"的事，与"怎么解析"是两种变化原因：解析规则要稳
（它决定行为），而诊断输出会随使用者反馈不断调整措辞与检查项。分开后，
``roles.py`` 只保留会影响运行行为的代码。

对外契约
========
* :func:`describe_roles` —— 对齐文本表格（角色 / 档案 / 生效来源 / 参数）；
* :func:`check_roles` —— 返回问题列表（空 = 通过），供 ``--check`` 与测试使用。

两者都接受与 :func:`resolve_roles` 相同的参数（注册表 / ``--llm_config`` /
``--role-config`` / 环境变量），因此诊断出的就是运行时真正会用的那套。
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from drsr_420.llm.factory import ClientFactory, load_llm_config
from drsr_420.llm.roles import TASKS, RoleRegistry, resolve_roles

#: 优先级说明，渲染在表格底部（与模块文档和 README 保持一致）。
_PRECEDENCE_NOTE = (
    "优先级: --role-config > env DRSR_ROLE_CONFIG_<ROLE> > 注册表角色绑定"
    " > --llm_config > 注册表 default > 内置默认"
)


def describe_roles(
    *,
    registry: RoleRegistry | None = None,
    cli_default: str | None = None,
    cli_overrides: Mapping[str, str] | Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """把「角色 → 档案」渲染成对齐的文本表格（由 :func:`resolve_roles` 生成）。

    列宽**按内容算**而不是写死：自定义提供商上线后档案名从 ``glm_glm-5.3-flash``
    （17 字符）变成 ``deepseek_deepseek-v4-flash``（26 字符）加来源列
    ``registry:roles.explain.config``（30 字符），写死的 26/24 会把两列挤成
    ``...-flashregistry:roles...`` 连在一起。
    """
    registry = registry if registry is not None else RoleRegistry.load()
    resolutions = resolve_roles(
        registry=registry, cli_default=cli_default,
        cli_overrides=cli_overrides, environ=environ)

    rows = [
        (role, res.profile, res.source,
         ", ".join(f"{k}={v}" for k, v in res.params.items()) or "-")
        for role, res in resolutions.items()
    ]
    # +2 = 列间至少留两个空格；表头用中文，宽度按字符算（中文显示更宽，仅影响表头观感）
    w_role = max([len("角色")] + [len(r[0]) for r in rows]) + 2
    w_prof = max([len("档案")] + [len(r[1]) for r in rows]) + 2
    w_src = max([len("生效来源")] + [len(r[2]) for r in rows]) + 2
    rule = "-" * (w_role + w_prof + w_src + 12)

    origin = str(registry.path) if registry.path else "(未找到注册表，使用内置默认)"
    lines = [
        f"LLM 角色配置（{len(resolutions)} 个角色）",
        "=" * len(rule),
        f"注册表: {origin}",
        "=" * len(rule),
        f"{'角色':<{w_role}}{'档案':<{w_prof}}{'生效来源':<{w_src}}参数",
        rule,
    ]
    for role, profile, source, params in rows:
        lines.append(f"{role:<{w_role}}{profile:<{w_prof}}{source:<{w_src}}{params}")
    lines.append(rule)
    lines.append(_PRECEDENCE_NOTE)
    return "\n".join(lines)


def check_roles(
    *,
    registry: RoleRegistry | None = None,
    cli_default: str | None = None,
    cli_overrides: Mapping[str, str] | Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """自检：返回问题列表（空 = 通过）。

    检查项：

    1. 每个角色都有解析结果（``--role-config`` 拼错角色名会在这里暴露）；
    2. 被引用的档案文件存在——缺失时给出 ``cp <name>.config.example`` 的修复提示，
       这条专门防"新克隆的仓库拿不到配置起点"；
    3. 档案能被解析且 ``model`` 字段合法（``provider/model`` 格式）；
    4. 能真的构造出客户端——密钥缺失（提示里带具体环境变量名）、自定义提供商漏写
       ``base_url``、``dialect`` 拼错都在这一步暴露，且透出 ClientFactory 的可操作提示。

    同一档案被多个角色引用时只检查一次（6 个角色常常共用一份默认档案）。
    问题分两类、修复动作不同（建档案 vs 补密钥），所以措辞不统一成 ``cp``：
    缺档案才给 ``cp``，缺密钥给的是"填哪个字段 / 设哪个环境变量"。
    """
    registry = registry if registry is not None else RoleRegistry.load()

    try:
        resolutions = resolve_roles(
            registry=registry, cli_default=cli_default,
            cli_overrides=cli_overrides, environ=environ)
    except Exception as exc:                      # 角色名拼写错误等
        return [f"角色解析失败: {exc}"]

    problems: list[str] = []
    for role in TASKS:
        if role not in resolutions:
            problems.append(f"角色 {role} 没有解析结果")

    seen: set[Path] = set()
    for resolution in resolutions.values():
        path = Path(resolution.config_path)
        if path in seen:
            continue
        seen.add(path)

        if not path.exists():
            problems.append(
                f"档案文件不存在: {path}"
                f"（可执行 `cp {path.name}.example {path}`，"
                f"或用 --templates 查看随仓库分发的模板）")
            continue

        try:
            config = load_llm_config(str(path))
        except Exception as exc:
            problems.append(f"档案 {path} 读取失败: {exc}")
            continue

        try:
            ClientFactory.from_config(dict(config))
        except Exception as exc:
            problems.append(f"档案 {path} 无法构造客户端: {exc}")

    return problems
