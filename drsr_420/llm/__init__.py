"""LLM 接入层（对外 API 与原根目录 ``llm.py`` 完全一致）。

子模块职责
==========
* ``client.py``       ``LLMClient``：请求/重试/流式/参数适配/token 记账
* ``providers.py``    各提供商子类（默认 base_url 与旧拼写别名）
* ``factory.py``      ``ClientFactory`` / ``load_llm_config`` / ``parse_provider_model``
* ``stats.py``        实验级全局 token 与耗时统计
* ``tools_schema.py`` 工具调用 schema（function-calling 用）

为什么本模块还需要 ``__getattr__``
==================================
旧实现是单模块 ``llm.py``，所以外部与测试会直接访问两类"不在 `__all__` 里"的名字：

1. **私有名**：``llm._post_with_retry``、``llm._accumulate_global_stats``
   （``mock.patch`` / 单测直接用它们）；
2. **会被重新绑定的模块级状态**：``GLOBAL_TIME_SECONDS`` 由 stats 模块
   ``global X; X += ...`` 重新绑定——``from ... import X`` 只会拿到导入那一刻的旧值。

因此这两类名字**不做 eager re-export**，一律经 ``__getattr__`` 从**定义它的子模块**
取；``owner_module()`` 同时供兼容层把属性写入（``mock.patch`` 的打桩）转发到定义处，
否则桩会打在门面上而实现模块看不到。
"""
import importlib

from drsr_420.llm import tools_schema
from drsr_420.llm.client import LLMClient
from drsr_420.llm.factory import (
    ClientFactory,
    load_llm_config,
    normalize_llm_config,
    parse_provider_model,
)
from drsr_420.llm.providers import (
    BltClient,
    CSTCloudClient,
    DeepInfraClient,
    DeepSeekClient,
    GLMClient,
    OllamaClient,
    SiliconflowClient,
    SliconflowClient,
    ZhipuClient,
)
from drsr_420.core.llm_stats import (
    get_global_time,
    get_global_tokens,
    reset_global_time,
    reset_global_tokens,
)
from drsr_420.llm.tools_schema import tools

#: 名字归属查找顺序——"定义该名字的模块"必须排在"只是 import 了它"的模块前面。
_OWNER_MODULES = (
    "drsr_420.llm.client",
    "drsr_420.core.llm_stats",
    "drsr_420.llm.providers",
    "drsr_420.llm.factory",
    "drsr_420.llm.tools_schema",
)

#: 名字 → 归属模块路径，首次解析后记住。
#: 必要性：``mock.patch`` 退出时先 ``delattr`` 再判断 ``hasattr`` 来决定是否回写原值。
#: 删除后按"当前谁还有这个名字"重新查找必然查不到，回写就会落到门面上，
#: 而实现模块（如 client.py）的全局命名空间永远少一个名字 → 后续调用 NameError。
_OWNER_CACHE: dict[str, str] = {}


def owner_module(name: str):
    """返回定义 ``name`` 的子模块；找不到返回 ``None``。

    删除后仍能从 :data:`_OWNER_CACHE` 恢复归属（见其说明）。
    """
    for module_path in _OWNER_MODULES:
        module = importlib.import_module(module_path)
        if hasattr(module, name):
            _OWNER_CACHE[name] = module_path
            return module
    cached = _OWNER_CACHE.get(name)
    if cached is not None:
        return importlib.import_module(cached)
    return None

__all__ = [
    "LLMClient", "ClientFactory", "load_llm_config", "parse_provider_model",
    "normalize_llm_config", "DeepSeekClient", "SiliconflowClient", "SliconflowClient",
    "DeepInfraClient", "CSTCloudClient", "OllamaClient", "BltClient", "ZhipuClient",
    "GLMClient", "reset_global_tokens", "get_global_tokens", "reset_global_time",
    "get_global_time", "tools", "owner_module",
]


def __getattr__(name: str):
    """惰性解析私有名与模块级状态（见模块文档字符串）。"""
    module = owner_module(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
