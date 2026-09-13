"""LLM 接入层（原根目录 ``llm.py`` 的实现，公开 API 保持不变）。

子模块职责
==========
* ``client.py``       ``LLMClient``：请求/重试/流式/参数适配/token 记账
* ``adapt.py``        请求体方言适配（glm/deepseek/ollama/openai，含自定义提供商）
* ``providers.py``    各提供商子类（默认 base_url 与旧拼写别名）
* ``factory.py``      ``ClientFactory`` / ``load_llm_config`` / ``parse_provider_model``
* ``stats.py``        实验级全局 token 与耗时统计
* ``tools_schema.py`` 工具调用 schema（function-calling 用）

为什么本模块还需要 ``__getattr__``
==================================
旧实现是单模块 ``llm.py``，所以外部与单测会直接访问两类"不在 `__all__` 里"的名字：

1. **私有名**：``llm._post_with_retry``、``llm._accumulate_global_stats``；
2. **会被重新绑定的模块级状态**：``GLOBAL_TIME_SECONDS`` 由 stats 模块
   ``global X; X += ...`` 重新绑定——``from ... import X`` 只会拿到导入那一刻的旧值。

因此这两类名字**不做 eager re-export**，一律经 ``__getattr__`` 从**定义它的子模块**
取（:func:`owner_module` 负责定位归属）。

根目录的转发层 ``llm.py`` 已在阶段 7 删除，已无"门面模块"需要把属性写入转发到定义处；
``mock.patch`` 请一律打在**定义处**：``LLMClient.chat`` 是在 ``drsr_420/llm/client.py``
的全局命名空间里查找 ``_post_with_retry`` 的，打在 ``drsr_420.llm`` 上等于没打。
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
    OpenAICompatClient,
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
#: ``adapt`` 排在 ``client`` 前：``adapt_payload`` 定义在 adapt，client 只是 import 了它。
_OWNER_MODULES = (
    "drsr_420.llm.adapt",
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
    "GLMClient", "OpenAICompatClient", "DIALECTS",
    "reset_global_tokens", "get_global_tokens", "reset_global_time",
    "get_global_time", "tools", "owner_module",
    # 角色 → 档案（Q3）
    "roles", "TASKS", "RoleClients", "resolve_roles", "load_role_config",
    "describe_roles", "check_roles",
]


#: 角色 → 档案（Q3）相关名字：**惰性**转发到对应的子模块。
#:
#: 刻意不在这里 eager import 子模块：`python -m drsr_420.llm.roles` 会先执行本
#: `__init__`，若此处已把 roles 放进 sys.modules，runpy 会报
#: "found in sys.modules after import of package ... prior to execution"。
#: 走 PEP 562 惰性解析后，`from drsr_420.llm import RoleClients` 与
#: `llm.roles.describe_roles()` 都照常可用，且 `-m` 入口干净。
_LAZY_ROLES_API: dict[str, str] = {
    # 子模块本身 + 声明与解析（roles.py）
    "roles": "drsr_420.llm.roles",
    "role_clients": "drsr_420.llm.role_clients",
    "role_diagnostics": "drsr_420.llm.role_diagnostics",
    "TASKS": "drsr_420.llm.roles",
    "RoleEntry": "drsr_420.llm.roles",
    "RoleRegistry": "drsr_420.llm.roles",
    "RoleResolution": "drsr_420.llm.roles",
    "resolve_roles": "drsr_420.llm.roles",
    "resolve_params": "drsr_420.llm.roles",
    "profile_path": "drsr_420.llm.roles",
    "registry_path": "drsr_420.llm.roles",
    "list_profiles": "drsr_420.llm.roles",
    "list_templates": "drsr_420.llm.roles",
    "BUILTIN_ROLE_PARAMS": "drsr_420.llm.roles",
    "DEFAULT_PROFILE": "drsr_420.llm.roles",
    "ENV_ROLE_PREFIX": "drsr_420.llm.roles",
    # 客户端构造（role_clients.py）
    "RoleClients": "drsr_420.llm.role_clients",
    "load_role_config": "drsr_420.llm.role_clients",
    "build_role_client": "drsr_420.llm.role_clients",
    # 渲染与自检（role_diagnostics.py）
    "describe_roles": "drsr_420.llm.role_diagnostics",
    "check_roles": "drsr_420.llm.role_diagnostics",
}


#: 属于"子模块本身"的名字（其余按属性从归属模块取）。
#: 必须显式列出而不是后缀匹配——``resolve_roles`` 也以 "roles" 结尾。
_LAZY_SUBMODULES = frozenset({"roles", "role_clients", "role_diagnostics"})


def __getattr__(name: str):
    """惰性解析角色 API、私有名与模块级状态（见模块文档字符串）。"""
    owner = _LAZY_ROLES_API.get(name)
    if owner is not None:
        module = importlib.import_module(owner)
        return module if name in _LAZY_SUBMODULES else getattr(module, name)
    module = owner_module(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_LAZY_ROLES_API))
