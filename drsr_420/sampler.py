"""采样与编排兼容层（re-export 至 agents 子包）。

原实现已迁移到 drsr_420/agents/：
- Sampler / LLM / 骨架提取  ->  drsr_420.agents.sampler_agent
- SamplingOrchestrator      ->  drsr_420.agents.coordinator_agent

本模块仅保留旧模块名，供 config.py / pipeline.py / main.py 与外部脚本无缝引用。
"""
from __future__ import annotations

# 旧 API 常量占位（原实现在旧版 sampler.py 中定义，保留以兼容潜在引用）。
# 注意：这里绝不允许放真实密钥——本文件在 git 跟踪内，任何 key 都会随仓库公开
# 并永久留在历史中（与 data_analyse_real.py 的占位风格保持一致，全部置 None）。
Port = None
API_HOST = None
API_KEY = None
API_MODEL = None
MAX_TOKENS = None

from drsr_420.agents.sampler_agent import (  # noqa: E402
    LLM,
    SamplerAgent as Sampler,
)
from drsr_420.agents.skeleton import (  # noqa: E402
    MAX_BODY_RETRIES as _MAX_BODY_RETRIES,
    extract_body as _extract_body,
    extract_code_fragment as _extract_code_fragment,
)
from drsr_420.agents.coordinator_agent import (  # noqa: E402
    CoordinatorAgent as SamplingOrchestrator,
    _SAMPLER_LOCK,
    atomic_write_json as _atomic_write_json,
    clone_llm_client as _clone_llm_client,
)
