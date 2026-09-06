"""数据分析器兼容层（re-export 至 agents 子包）。

原实现已迁移到 drsr_420/agents/data_analyzer_agent.py。
本模块仅保留旧模块名 DataAnalyzer 与 API 常量占位供外部无缝引用。
"""
from __future__ import annotations

Port = '5000'

# API 常量占位：pipeline.py 旧逻辑曾做 _dar.API_* 覆盖（已被移除的死代码），
# 保留占位以免任何潜在引用触发 AttributeError。
API_HOST = None
API_KEY = None
API_MODEL = None
MAX_TOKENS = None

from drsr_420.agents.data_analyzer_agent import (  # noqa: E402
    DataAnalyzerAgent as DataAnalyzer,
)
