"""采样与编排兼容层（re-export 至 agents 子包）。

原实现已迁移到 drsr_420/agents/：
- Sampler / LLM / 骨架提取  ->  drsr_420.agents.sampler_agent
- SamplingOrchestrator      ->  drsr_420.agents.coordinator_agent

本模块仅保留旧模块名，供 config.py / pipeline.py / main.py 与外部脚本无缝引用。
"""
from __future__ import annotations

# 旧 API 常量占位（原实现在旧版 sampler.py 中定义，保留以兼容潜在引用）
Port = '5000'
API_HOST = "api.bltcy.ai"
API_KEY = "sk-1zejrP7CKGPUXASwGpow3vOQ1Pjl5QzeU8xCjMrOEMSbqFQd"
API_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 1024

from drsr_420.agents.sampler_agent import (  # noqa: E402
    LLM,
    SamplerAgent as Sampler,
    _extract_body,
    _extract_code_fragment,
    _MAX_BODY_RETRIES,
)
from drsr_420.agents.coordinator_agent import (  # noqa: E402
    CoordinatorAgent as SamplingOrchestrator,
    _SAMPLER_LOCK,
    atomic_write_json as _atomic_write_json,
    clone_llm_client as _clone_llm_client,
)
