"""命令行入口的 LLM 准备步骤：默认档案、角色客户端池。

角色归属
========
``cli`` 层的**准备**部件：把命令行参数（``--llm_config`` / ``--role-config``）
变成可用的 LLM 客户端。抽出来的理由是 :mod:`drsr_420.cli.main` 已经装了太多
"实验怎么跑"的编排（spec 渲染、CSV 加载、产物快照），而"模型怎么选"的失败模式
完全不同——它几乎总是配置问题，需要的是可操作的提示（缺哪个文件、`cp` 哪条命令），
而不是实验日志。

对外契约
========
* :func:`load_llm_config_file` —— 读默认档案；缺失时报错并列出可用的档案与模板；
* :func:`build_llm_client` —— 由档案 dict 构造客户端，失败即退出；
* :func:`build_role_clients` —— 构造角色客户端池，并把解析结果回写环境变量。

三者中只有**解析期**的"配置没准备好"会以 ``SystemExit`` 终止（``--role-config`` 写错角色名
之类，任何角色都无从谈起）。**单份档案**暂时不可用（缺文件、密钥没填、端点写错）不再拦住
启动：客户端是按需构造的，用不到的档案不该拖死这次实验——但也不能装作没看见，所以
:func:`build_role_clients` 会在启动时跑一遍离线自检并逐条 ``[WARN]``，真正用到该角色时
再带着同样的说明报错。
"""
from __future__ import annotations

import json
import os


def load_llm_config_file(path: str | None) -> dict:
    """读取默认档案；``path`` 为 ``None`` 时用配置注册表的 ``default``。

    重构前这里会**静默写出一份占位配置**（哪怕路径叫 ``no_such.config``），
    随后必然因 api_key 为空 / 模型不存在而失败——制造了一个看起来像配置问题的
    真实困惑点。现在改为：档案不存在就明确报错，并列出可用的档案与模板。
    """
    from drsr_420.llm.factory import CONFIG_DIR_NAME, locate_config
    from drsr_420.llm.roles import list_profiles, list_templates

    try:
        target = locate_config(path)
    except Exception as e:
        print(f"[FATAL] 无法解析 LLM 档案路径: {e}")
        raise SystemExit(1)

    if not target.exists():
        print(f"[FATAL] LLM 档案不存在: {target}")
        profiles = list_profiles()
        print("[FATAL] 已有档案: " + (", ".join(profiles) if profiles else "(无)"))
        templates = list_templates()
        print("[FATAL] 随仓库分发的模板: " + (", ".join(templates) if templates else "(无)"))
        if templates:
            name = templates[0]
            print(f"[FATAL] 可执行：cp {CONFIG_DIR_NAME}/{name} "
                  f"{CONFIG_DIR_NAME}/{name[: -len('.example')]}")
        print("[FATAL] 或运行 `python -m drsr_420.llm.roles --check` 查看角色档案解析情况")
        raise SystemExit(1)

    with open(target, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_llm_client(llm_config: dict):
    """按配置构造"默认"客户端；失败即退出（避免无客户端空转）。

    模型名格式：``provider/model``（如 ``CSTCloud/gpt-oss-120b``）。provider 解析、
    api_key 解析、base_url 与生成参数注入统一由 ``ClientFactory`` 完成。

    注意：本客户端只用于日志与快照的"代表模型"展示；真正分发给各 Agent 的是
    :func:`build_role_clients` 返回的角色客户端池。
    """
    from drsr_420.llm import ClientFactory

    try:
        client = ClientFactory.from_config(llm_config)
        print(f"[INFO] LLM client initialized: provider={client._provider_name()}, "
              f"model={client.model}, kwargs={client.kwargs}")
        return client
    except Exception as e:
        print(f"[FATAL] Failed to init LLM client: {e}")
        print("[FATAL] 请检查 --llm_config / --role-config 指定的档案是否存在、api_key 是否配置正确")
        print("[FATAL] 程序退出，避免在无 LLM 客户端的情况下空转。")
        raise SystemExit(1)


def build_role_clients(llm_config: str | None, role_overrides):
    """按「角色 → 档案」注册表解析各角色的 LLM 客户端（**按需构造**，此处不建连接）。

    角色化的解析见 :mod:`drsr_420.llm.roles`：``--llm_config`` 只是**默认档案**
    （未绑定档案的角色共用它），``--role-config`` 可按角色精确覆盖甚至换模型。
    本函数只做三件事：解析、回写环境变量、把不可用的档案**告警**出来。

    Raises:
        SystemExit: 仅在**解析期**失败时（角色名拼错、``--role-config`` 格式错）。
            单份档案不可用不再退出——见模块文档"按需构造"。
    """
    from drsr_420.llm.role_clients import RoleClients
    from drsr_420.llm.role_diagnostics import check_roles, describe_roles

    try:
        role_clients = RoleClients.from_registry(
            cli_default=llm_config, cli_overrides=role_overrides)
    except Exception as e:
        print(f"[FATAL] 角色档案解析失败: {e}")
        print("[FATAL] 可用角色与档案见 `python -m drsr_420.llm.roles`")
        raise SystemExit(1)

    try:
        print("[INFO] LLM 角色配置：")
        print(describe_roles(cli_default=llm_config, cli_overrides=role_overrides))
    except Exception as e:
        print(f"[WARN] 渲染角色配置失败: {e}")

    # 离线自检：**只告警、不退出**。某份档案现在不可用不该挡住这次实验（可能根本用不到
    # 那个角色），但提前说出来比等到半路报错好；真正用到该角色时 get() 会带同样的说明。
    try:
        for problem in check_roles(cli_default=llm_config, cli_overrides=role_overrides):
            print(f"[WARN] {problem}")
    except Exception as e:
        print(f"[WARN] 角色配置自检失败: {e}")

    # 把解析结果回写环境变量：MCP 子进程（read_paper）只能靠环境变量继承父进程
    # 选定的档案；同时保证本进程内后续任何 resolve_roles() 看到同一份结果。
    for role, path in role_clients.resolved_profiles().items():
        os.environ[f"DRSR_ROLE_CONFIG_{role.upper()}"] = path
    return role_clients
