"""各提供商的 LLMClient 子类（仅提供默认 base_url 与旧拼写别名）。"""
import os

from drsr_420.llm.client import LLMClient


class DeepSeekClient(LLMClient):
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.deepseek.com"):
        super().__init__(api_key=api_key, model=model, base_url=base_url)

class SiliconflowClient(LLMClient):
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.siliconflow.cn/v1"):
        super().__init__(api_key=api_key, model=model, base_url=base_url)

class DeepInfraClient(LLMClient):
    """DeepInfra，OpenAI Chat Completions 兼容接口。"""

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.deepinfra.com/v1/openai"):
        super().__init__(api_key=api_key, model=model, base_url=base_url)

class CSTCloudClient(LLMClient):
    """CSTCloud（科技云）提供商，OpenAI Chat Completions 兼容接口。

    默认基址：https://uni-api.cstcloud.cn/v1
    使用示例：model="CSTCloud/gpt-oss-120b" 或 "CSTCloud/qwen3:235b"
    建议环境变量：CSTCLOUD_API_KEY
    """
    def __init__(self, api_key: str, model: str, base_url: str = "https://uni-api.cstcloud.cn/v1"):
        super().__init__(api_key=api_key, model=model, base_url=base_url)

# 兼容旧拼写，避免历史引用报错
SliconflowClient = SiliconflowClient

class OllamaClient(LLMClient):
    def __init__(self, api_key: str, model: str, base_url: str = "http://localhost:11111/v1"):
        super().__init__(api_key=api_key, model=model, base_url=base_url)

class BltClient(LLMClient):
    """BLT（柏拉图）网关，OpenAI Chat Completions 兼容接口。

    默认基址含 /v1，路径将拼接为 /chat/completions。
    """
    def __init__(self, api_key: str, model: str, base_url: str = None):
        base_url = base_url or os.getenv('BLT_API_BASE', 'https://api.bltcy.ai/v1')
        super().__init__(api_key=api_key, model=model, base_url=base_url)

class ZhipuClient(LLMClient):
    """GLM（智谱）提供商，OpenAI Chat Completions 兼容接口。

    默认基址：https://open.bigmodel.cn/api/paas/v4
    使用示例：model="glm/glm-5.3-flash"
    api_key 为智谱开放平台的 'id.secret' 格式；建议环境变量：ZHIPU_API_KEY
    """
    def __init__(self, api_key: str, model: str, base_url: str = None):
        base_url = base_url or os.getenv('ZHIPU_API_BASE', 'https://open.bigmodel.cn/api/paas/v4')
        super().__init__(api_key=api_key, model=model, base_url=base_url)

# 兼容旧拼写
GLMClient = ZhipuClient
