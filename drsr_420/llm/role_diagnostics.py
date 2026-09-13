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
* :func:`check_roles` —— **离线**自检，返回问题列表（空 = 通过），供 ``--check`` 与测试使用；
* :func:`ping_roles` / :func:`format_ping` / :func:`ping_problems` —— **联网**连通性自检
  （每份档案一次真实请求），供 ``--ping`` 使用。

两者都接受与 :func:`resolve_roles` 相同的参数（注册表 / ``--llm_config`` /
``--role-config`` / 环境变量），因此诊断出的就是运行时真正会用的那套。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from drsr_420.llm.client import _post_with_retry, gateway_error_detail
from drsr_420.llm.factory import ClientFactory, load_llm_config
from drsr_420.llm.role_clients import RoleClients
from drsr_420.llm.roles import PROFILE_SUFFIX, TASKS, RoleRegistry, resolve_roles

#: 优先级说明，渲染在表格底部（与模块文档和 README 保持一致）。
_PRECEDENCE_NOTE = (
    "优先级: --role-config > env DRSR_ROLE_CONFIG_<ROLE> > 注册表角色绑定"
    " > --llm_config > 注册表 default > 内置默认"
)

#: ``--ping`` 的探测提示词与输出上限：只要能证明"端点 + 密钥 + 方言"这条路走得通。
PING_PROMPT = "Reply with the single word: pong"
PING_MAX_TOKENS = 16


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


# ── 连通性自检（联网）────────────────────────────────────────────────

@dataclass(frozen=True)
class PingOutcome:
    """一份档案的连通性探测结果（按**档案**去重；``roles`` 是共用它的角色）。"""

    profile: str
    config_path: str
    roles: tuple[str, ...]
    ok: bool
    status: object          # HTTP 状态码，或 "ERR"
    seconds: float
    model: str = ""
    endpoint: str = ""
    detail: str = ""        # 失败原因（可操作：给出端点与代理/网关的原话）


def _profile_id_of(path: str) -> str:
    name = Path(path).name
    return name[: -len(PROFILE_SUFFIX)] if name.endswith(PROFILE_SUFFIX) else name


def _clip(text: object, limit: int = 300) -> str:
    """把服务端返回的任意文本压成一行短摘要，并保证**能在 GBK 控制台打出来**。

    这里的文本来自网络（网关的错误原话），内容不可控；Windows 控制台默认 GBK，
    遇到不可编码字符会直接抛 ``UnicodeEncodeError`` 把诊断本身搞崩——诊断工具因为
    "要诊断的东西"而崩掉就很讽刺了。不可编码字符一律降级成 ``?``。
    """
    out = " ".join(str(text).split())
    try:
        out.encode("gbk")
    except UnicodeEncodeError:
        out = out.encode("gbk", "replace").decode("gbk")
    return out[:limit]


def ping_roles(
    *,
    registry: RoleRegistry | None = None,
    cli_default: str | None = None,
    cli_overrides: Mapping[str, str] | Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
    timeout: tuple[int, int] = (10, 60),
    max_tokens: int = PING_MAX_TOKENS,
) -> list[PingOutcome]:
    """对每份被引用的档案发**一次真实请求**，返回逐档案的结果。

    与 :func:`check_roles` 的分工：那个是**离线**结构校验（档案在不在、``model`` 合法、
    密钥可达），本函数真的联网。刻意不并进 ``--check``——CI 与预检不该依赖外网。

    几点设计：

    * **按档案去重**：6 个角色常常共用一份档案，逐角色发请求纯属浪费；每份档案只发一次，
      但用它的**首个角色**（``TASKS`` 顺序），这样连同该角色的参数与请求体方言一起验证；
    * **不重试**：``max_retries=0`` + 明确的连接/读取超时。探测要的是"现在通不通"，
      指数退避只会让一条命令卡上几分钟；
    * **输出上限压到 :data:`PING_MAX_TOKENS`**：探测的是连通性与鉴权，不是生成质量；
    * 任何角色解析期的错误（拼错角色名、``--role-config`` 格式错）也会变成一条结果，
      而不是抛出去——``--ping`` 本身要能干净地跑完并给出退出码。
    """
    registry = registry if registry is not None else RoleRegistry.load()
    try:
        role_clients = RoleClients.from_registry(
            registry=registry, cli_default=cli_default,
            cli_overrides=cli_overrides, environ=environ)
    except Exception as exc:                           # noqa: BLE001
        return [PingOutcome(profile="(解析失败)", config_path="", roles=(), ok=False,
                            status="ERR", seconds=0.0,
                            detail=f"角色解析失败：{_clip(exc)}")]

    grouped: dict[str, list[str]] = {}
    for role, resolution in role_clients.resolutions.items():
        grouped.setdefault(str(resolution.config_path), []).append(role)

    return [_ping_one(role_clients, roles[0], tuple(roles), path, timeout, max_tokens)
            for path, roles in grouped.items()]


def _ping_one(role_clients: RoleClients, role: str, roles: tuple[str, ...],
              path: str, timeout: tuple[int, int], max_tokens: int) -> PingOutcome:
    """对一份档案发一次最小真实请求（用 ``role`` 的参数与方言）。"""
    profile = _profile_id_of(path)
    try:
        client = role_clients.get(role)
    except Exception as exc:                           # noqa: BLE001
        return PingOutcome(profile, path, roles, False, "ERR", 0.0,
                           detail=f"档案不可用：{_clip(exc)}")

    payload = client._build_payload([{"role": "user", "content": PING_PROMPT}])
    for key in ("max_tokens", "max_completion_tokens"):
        if isinstance(payload.get(key), int):
            payload[key] = min(payload[key], max_tokens)
    payload["stream"] = False
    url = f"{client.base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {client.api_key}",
               "Content-Type": "application/json"}

    started = time.time()
    try:
        resp = _post_with_retry(url, headers, payload, max_retries=0,
                                timeout=timeout, stream=False)
    except Exception as exc:                           # noqa: BLE001
        resp = getattr(exc, "response", None)
        detail = _clip(resp.text) if resp is not None else _clip(exc)
        return PingOutcome(profile, path, roles, False,
                           getattr(resp, "status_code", "ERR"), time.time() - started,
                           client.model, client.base_url,
                           f"{client.base_url} 请求失败（{type(exc).__name__}）：{detail}")

    seconds = time.time() - started
    if resp.status_code != 200:
        return PingOutcome(profile, path, roles, False, resp.status_code, seconds,
                           client.model, client.base_url,
                           f"{client.base_url} 返回 HTTP {resp.status_code}："
                           f"{_clip(resp.text)}")

    try:
        data = resp.json()
    except Exception:                                  # noqa: BLE001
        data = None
    if not isinstance(data, dict) or not data.get("choices"):
        # 网关可能把错误包在 HTTP 200 里（如智谱的 {"code":500,"msg":"404 NOT_FOUND"}）
        detail = gateway_error_detail(data) or _clip(resp.text)
        return PingOutcome(profile, path, roles, False, resp.status_code, seconds,
                           client.model, client.base_url,
                           f"{client.base_url} 的响应里没有 choices：{detail}")

    return PingOutcome(profile, path, roles, True, resp.status_code, seconds,
                       client.model, client.base_url)


def ping_problems(outcomes: Sequence[PingOutcome]) -> list[str]:
    """把失败的探测结果转成可操作的问题列表（空 = 全部通过）。"""
    return [f"连通性自检失败 —— 档案 {o.profile}（角色 {', '.join(o.roles) or '?'}）: {o.detail}"
            for o in outcomes if not o.ok]


def format_ping(outcomes: Sequence[PingOutcome]) -> str:
    """把探测结果渲染成对齐表格；失败的行下面附一行原因。"""
    if not outcomes:
        return "连通性自检：没有可探测的档案（注册表为空？）"

    headers = ("结果", "档案", "角色", "模型", "端点", "耗时")
    rows = [
        ("OK" if o.ok else "FAIL", o.profile, ",".join(o.roles) or "-",
         o.model or "-", o.endpoint or "-", f"{o.seconds:.1f}s")
        for o in outcomes
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) + 2
        for i in range(len(headers))
    ]
    rule = "-" * (sum(widths) + 8)
    lines = [
        f"LLM 连通性自检（{len(outcomes)} 份档案，各发一次真实请求）",
        "=" * len(rule),
        "".join(f"{headers[i]:<{widths[i]}}" for i in range(len(headers))),
        rule,
    ]
    for row, outcome in zip(rows, outcomes):
        lines.append("".join(f"{row[i]:<{widths[i]}}" for i in range(len(headers))))
        if not outcome.ok:
            lines.append(f"      └─ {outcome.detail}")
    lines.append(rule)
    lines.append("提示：失败原因里会带上端点的原话；--check 只做离线结构校验，本项才联网。")
    return "\n".join(lines)
