"""经验总结器兼容层（re-export 至 agents 子包）。

原实现已迁移到 drsr_420/agents/experience_summarizer_agent.py。
本模块仅保留旧模块名 ExperienceSummarizer 供外部无缝引用。
"""
from __future__ import annotations

from drsr_420.agents.experience_summarizer_agent import (  # noqa: E402
    ExperienceSummarizerAgent as ExperienceSummarizer,
)
