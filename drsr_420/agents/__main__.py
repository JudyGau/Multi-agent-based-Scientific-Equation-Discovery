"""``python -m drsr_420.agents`` —— 打印多 Agent 组织图 / 契约自检。

用法::

    python -m drsr_420.agents            # 打印组织图（由 AgentSpec 生成）
    python -m drsr_420.agents --check    # 契约自检，发现问题时退出码为 1

组织图不是手写文档：它由每个 Agent 的 ``SPEC``（角色卡）渲染而来，
因此"代码里的调用关系"和"图上的调用关系"不会漂移。
"""
from __future__ import annotations

import sys

from drsr_420.agents import agent_specs, check_contracts, describe_architecture

_USAGE = """用法:
    python -m drsr_420.agents            打印多 Agent 组织图
    python -m drsr_420.agents --check    契约自检（失败时退出码 1）
"""


def _safe_print(text: str) -> None:
    """打印组织图；遇到当前控制台编码无法表示的字符时降级为替换符。

    该项目已有同类踩坑史：`evaluate_on_problems` 的 'R²' 在 GBK 控制台直接抛
    UnicodeEncodeError。入口处统一兜底，避免"打个组织图把 CLI 打崩"。
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if "--help" in args or "-h" in args:
        _safe_print(_USAGE)
        return 0

    if "--check" in args:
        problems = check_contracts()
        for problem in problems:
            _safe_print(f"[FAIL] {problem}")
        if problems:
            _safe_print(f"\n契约自检失败：{len(problems)} 项问题")
            return 1
        _safe_print(f"OK: {len(agent_specs())} 个 Agent 契约齐全，"
                    f"上下游引用与 thread_model 自洽")
        return 0

    _safe_print(describe_architecture())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
