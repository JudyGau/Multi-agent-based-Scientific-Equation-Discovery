"""配置加载、模型串解析、提供商归一化与客户端工厂。

本模块负责 **Q1（连接谁 / 用哪把钥匙）与 Q2（生成参数）**，并拥有档案文件的
**命名约定与定位规则**；**Q3（哪个角色用哪套）** 在 :mod:`drsr_420.llm.roles`。

命名约定（唯一权威是文件内的 ``model`` 字段，文件名只是给人看的标签）：

* 目录：``<仓库根>/config/``；
* 档案：``config/<提供商>_<模型>.config``（不入库，含密钥）；
* 模板：``config/<提供商>_<模型>.config.example``（入库，api_key 留空）。

提供商分两类，**接入新服务不必改代码**：

* **内置**（:data:`ClientFactory._PROVIDER_SPECS`）：默认 base_url 与密钥环境变量名
  写在表里，档案只需 ``model``；
* **自定义**：provider 段是代码没见过的名字（如 ``ustc``）时，只要档案里给出
  ``base_url`` 即可——端点属于 Q1（连接谁），本就该在档案里。请求体差异用
  ``dialect`` 字段声明（默认 ``openai``：不认识的一律不发）。
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple, Type

from drsr_420.llm.adapt import DIALECTS, resolve_dialect
from drsr_420.llm.client import LLMClient
from drsr_420.llm.providers import (
    BltClient,
    CSTCloudClient,
    DeepInfraClient,
    DeepSeekClient,
    OllamaClient,
    OpenAICompatClient,
    SiliconflowClient,
    ZhipuClient,
)

# 项目根目录（本文件位于 drsr_420/llm/，上溯 3 级即仓库根）：
# 用于在任意工作目录下定位 config/ 下的档案文件。
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: 配置目录名（相对仓库根）。
CONFIG_DIR_NAME = "config"

#: 档案文件后缀；``<name>.config.example`` 是随仓库分发的模板。
PROFILE_SUFFIX = ".config"
TEMPLATE_SUFFIX = ".config.example"

#: 内置默认档案 ID（**不含**后缀——后缀属于文件命名约定，不属于档案语义）。
DEFAULT_PROFILE = "glm_glm-5.3-flash"


def config_dir() -> Path:
    """配置目录的绝对路径（``<仓库根>/config``）。"""
    return _REPO_ROOT / CONFIG_DIR_NAME


def locate_config(path: str | os.PathLike | None = None) -> Path:
    """把档案引用解析成绝对路径（**只做定位，不解析文件名的语义**）。

    四种写法都接受，便于从任意工作目录调用：

    * 档案 ID：``glm_glm-5.3-flash``  → ``<仓库根>/config/glm_glm-5.3-flash.config``
    * 文件名：``glm_glm-5.3-flash.config``（等价于档案 ID）
    * 相对路径：``config/x.config`` / ``./x.config``（先按 cwd，再按仓库根）
    * 绝对路径：原样返回

    查找顺序：原样 → 补 ``.config`` 后缀 → 仓库根 → ``config/`` 目录。
    """
    if path is None:
        path = DEFAULT_PROFILE
    raw = str(path).strip()
    if not raw:
        raise ValueError("档案名为空：请提供档案 ID（如 glm_glm-5.3-flash）或 .config 路径")

    names = [raw]
    if not raw.endswith(PROFILE_SUFFIX) and not raw.endswith(TEMPLATE_SUFFIX):
        names.append(raw + PROFILE_SUFFIX)

    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate

    for name in names:
        for base in (Path.cwd(), _REPO_ROOT, config_dir()):
            hit = (base / name).resolve()
            if hit.exists():
                return hit
    # 都不存在：返回最可能的位置，让调用方拿到清晰的 FileNotFoundError
    return (config_dir() / names[-1]).resolve()


def load_llm_config(path: str | os.PathLike | None = None) -> dict:
    """读取 LLM 配置文件（JSON）。

    - 相对路径找不到时自动回退到仓库根与 ``config/`` 目录，避免依赖当前工作目录；
    - ``path`` 省略时读内置默认档案（:data:`DEFAULT_PROFILE`）；
    - 返回的 dict 可直接传给 ``ClientFactory.from_config``。

    注意：**库代码不应再传字面量文件名**——角色化的选择请走
    :func:`drsr_420.llm.roles.resolve_roles`；本函数的显式参数形态留给调用方
    （如 ``--role-config``）传入。
    """
    with open(locate_config(path), "r", encoding="utf-8") as f:
        return json.load(f)



def _derive_api_key_env(provider: str) -> str:
    """按 provider 段派生密钥环境变量名（``ustc`` -> ``USTC_API_KEY``）。

    自定义提供商没有内置表可查，Key 又常常由环境变量注入（部署机不落盘密钥），
    所以需要一条确定的派生规则——而不是让每个调用方各猜一个名字。
    """
    safe = re.sub(r'[^0-9A-Za-z]+', '_', provider).strip('_').upper()
    return f"{safe}_API_KEY" if safe else "LLM_API_KEY"


def parse_provider_model(model_str: str) -> Tuple[str, str]:
    """
    解析模型字符串为 (provider, model)。

    规则：第一个 '/' 之前为提供商（大小写不敏感），之后的全部为模型名（大小写敏感，允许包含 '/').
    示例：
    - "deepseek/deepseek-chat" -> ("deepseek", "deepseek-chat")
    - "SiliconFlow/Qwen/Qwen3-8B" -> ("siliconflow", "Qwen/Qwen3-8B")
    - "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct" -> ("deepinfra", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    - "ollama/llama3.1:8b" -> ("ollama", "llama3.1:8b")
    """
    if not isinstance(model_str, str) or '/' not in model_str:
        raise ValueError("缺少模型提供商：请使用 'provider/model' 格式，例如 'CSTCloud/gpt-oss-120b'")
    provider, model = model_str.split('/', 1)
    return provider.lower(), model



def normalize_llm_config(config: dict) -> dict:
    """统一规范化 LLM 配置：拒绝已废弃的 ``host``、补齐 scheme、校验 ``model``。

    - ``host`` 曾是 ``base_url`` 的**同一字段的另一种拼写**，现已统一为 ``base_url``：
      出现 ``host`` 即报错并给出改名提示。刻意不做静默兼容——内置提供商自带默认端点，
      忽略 ``host`` 会让请求悄悄打到默认地址，而不是用户写的那一个（自建/代理端点
      尤其致命，且完全没有信号）；
    - 无 scheme 的地址自动补 ``https://``，避免 requests 报 "No scheme supplied"；
    - 校验 ``model`` 必须是 'provider/model' 格式，配置错误尽早暴露。

    返回规范化后的新 dict，不改动入参。
    """
    cfg = dict(config)
    if 'host' in cfg:
        raise ValueError(
            f'配置键 "host" 已废弃，请改名为 "base_url"（收到 host={cfg["host"]!r}）。'
            f'两者只是同一字段的两种拼写，现已统一为 base_url。')
    base_url = cfg.get('base_url')
    if isinstance(base_url, str) and base_url.strip() and '://' not in base_url:
        cfg['base_url'] = 'https://' + base_url.strip()
    if 'model' in cfg:
        parse_provider_model(cfg['model'])  # 校验 provider 前缀，无效则抛出
    return cfg



class ClientFactory:
    """LLM 客户端工厂：按 'provider/model' 解析提供商并构造对应客户端。

    提供商规格集中在 ``_PROVIDER_SPECS``（类→环境变量→默认 base_url）与
    ``_PROVIDER_ALIASES``（别名→规范名）两张表，新增提供商只需加一行，
    无需在 from_config 内维护 if/elif 分支。
    """

    # 规范提供商 -> (客户端类, api_key 环境变量名, 默认 base_url)
    #   env_var 为 None 表示不强制要求 key（如 ollama 本地部署）。
    #   default_base_url 为 None 表示由客户端类自行从环境变量兜底（如 blt）。
    _PROVIDER_SPECS = {
        'deepseek':   (DeepSeekClient,    'DEEPSEEK_API_KEY',   'https://api.deepseek.com'),
        'siliconflow':(SiliconflowClient, 'SILICONFLOW_API_KEY','https://api.siliconflow.cn/v1'),
        'deepinfra':  (DeepInfraClient,   'DEEPINFRA_API_KEY',  'https://api.deepinfra.com/v1/openai'),
        'ollama':     (OllamaClient,      None,                 'http://localhost:11111/v1'),
        'blt':        (BltClient,         'BLT_API_KEY',        None),
        'cstcloud':   (CSTCloudClient,    'CSTCLOUD_API_KEY',   'https://uni-api.cstcloud.cn/v1'),
        'glm':        (ZhipuClient,       'ZHIPU_API_KEY',      'https://open.bigmodel.cn/api/paas/v4'),
    }

    # 提供商别名 -> 规范名（大小写不敏感的 provider 段经别名归一）
    _PROVIDER_ALIASES = {
        'silicon-flow': 'siliconflow', 'sflow': 'siliconflow',
        'deep-infra': 'deepinfra',
        'bltcy': 'blt', 'plato': 'blt',
        'cst': 'cstcloud', 'cst-cloud': 'cstcloud', 'keji': 'cstcloud', 'keji-yun': 'cstcloud',
        'glm4': 'glm', 'zhipu': 'glm', 'bigmodel': 'glm', 'big-model': 'glm',
    }

    @staticmethod
    def _unsupported_provider_message(provider: str) -> str:
        """未知 provider 段的报错：既列出内置项，也给出"不改代码接进来"的路子。

        内置列表由 ``_PROVIDER_SPECS`` 生成而非手写——手写的那份在加提供商时必然漂移。
        """
        builtin = "、".join(f"'{name}'" for name in ClientFactory._PROVIDER_SPECS)
        dialects = "|".join(DIALECTS)
        return (
            f"不支持的提供商: {provider}（内置：{builtin}；常见别名会自动归一）。"
            f"接入自建或第三方 OpenAI Chat Completions 兼容服务**无需改代码**——"
            f"在档案里给出 base_url 即可，例如 "
            f'{{"model": "{provider}/<模型名>", "base_url": "https://<主机>/v1"}}；'
            f'若该服务的请求体方言与某个内置家族一致，可另加 "dialect": "{dialects}"。'
        )

    @staticmethod
    def from_config(config: dict, task_params: dict | None = None):
        """
        基于 'provider/model' 创建具体客户端。

        必填：config['model']（形如 'provider/model'）。
        选填：config['api_key']、config['base_url']。
        配置先经 normalize_llm_config 统一规范化（补齐 scheme；``host`` 已废弃并会报错）。

        **自定义提供商**：provider 段不在内置表里时，只要给了 ``base_url`` 就照常构造
        （走 :class:`OpenAICompatClient`）。另有两个只对自定义提供商有意义的字段：

        * ``api_key_env``：密钥环境变量名，缺省按 provider 段派生（``ustc`` ->
          ``USTC_API_KEY``）；
        * ``api_key_required``：显式声明是否需要密钥。自定义提供商默认为 ``true``；
          本地免鉴权服务写 ``false``；
        * ``dialect``：请求体方言（``openai`` / ``glm`` / ``deepseek`` / ``ollama``），
          缺省为 ``openai``（见 :mod:`drsr_420.llm.adapt`）。

        Args:
            task_params: 覆盖/补充配置里的 ``tasks`` 字段，形如
                ``{"sampling": {"reasoning_effort": "low"}}``。由
                :mod:`drsr_420.llm.roles` 按「角色 → 档案」解析后传入；
                不传时退化为只读配置文件的 ``tasks``（旧格式，仍兼容）。
        """
        config = normalize_llm_config(config)
        if 'model' not in config:
            raise ValueError("缺少必要字段: model")

        provider, model = parse_provider_model(config['model'])
        api_key_cfg = config.get('api_key')
        base_url = config.get('base_url')

        # api_key 支持：
        # 1) 字符串（兼容旧格式）
        # 2) 字典：可按 provider 或完整 model（'provider/model'）配置不同 key
        api_key = None
        if isinstance(api_key_cfg, dict):
            def _get_case_insensitive(d: dict, k: str):
                for kk, vv in d.items():
                    try:
                        if str(kk).lower() == str(k).lower():
                            return vv
                    except Exception:
                        pass
                return None
            # 优先匹配完整模型名，其次按提供商名
            api_key = _get_case_insensitive(api_key_cfg, config.get('model', '')) or _get_case_insensitive(api_key_cfg, provider)
        elif isinstance(api_key_cfg, str):
            api_key = api_key_cfg
        else:
            api_key = None

        def _require_api_key(value: str, env_name: str) -> str:
            resolved = value or os.getenv(env_name, '')
            if not isinstance(resolved, str) or not resolved.strip():
                raise ValueError(
                    f"LLM provider '{provider}' 缺少 API key：请在对应档案"
                    f"（config/<提供商>_<模型>.config）的 api_key 字段或环境变量 "
                    f"{env_name} 中配置。"
                )
            return resolved

        # 提供商规格：内置表命中则用它；否则走**自定义提供商**——端点属于 Q1
        # （连接谁），本就该在档案里给出，因此这里只要求 base_url，不动代码表。
        canonical = ClientFactory._PROVIDER_ALIASES.get(provider, provider)
        spec = ClientFactory._PROVIDER_SPECS.get(canonical)
        if spec is None:
            if not base_url:
                raise ValueError(ClientFactory._unsupported_provider_message(provider))
            client_cls, spec_env_var, default_base_url = OpenAICompatClient, None, None
            spec_requires_key = True      # 自定义提供商默认需要密钥（本地免鉴权写 false）
        else:
            client_cls, spec_env_var, default_base_url = spec
            spec_requires_key = spec_env_var is not None   # 只有 ollama 这类本地部署为 None

        # 密钥环境变量名：档案显式声明 > 内置表 > 按 provider 段派生。
        # derived 兜底保证报错信息里永远有一个**可设置的具体变量名**，而不是 "None"。
        env_var = config.get('api_key_env') or spec_env_var or _derive_api_key_env(canonical)
        api_key_required = config.get('api_key_required')
        if api_key_required is None:
            api_key_required = spec_requires_key
        resolved_key = _require_api_key(api_key, env_var) if api_key_required else (api_key or '')

        # base_url：传入优先，否则用 spec 默认；spec 默认为 None 时由客户端类
        # 自行从环境变量兜底（如 BltClient 读 BLT_API_BASE）
        final_base_url = base_url or default_base_url
        client = client_cls(api_key=resolved_key, model=model, base_url=final_base_url)

        # 必须写别名规范化后的 canonical：_provider_name() 只在 provider 为空时才
        # 从 URL 推断，直接写原始别名（'zhipu'/'bigmodel'/'glm4'…）会让
        # 方言适配的所有分支（glm 的 max_tokens 改名、thinking、
        # deepseek 等）静默跳过——别名用户丢失按任务注入的 reasoning_effort。
        client.provider = canonical
        # 方言同理必须在构造处定死：自定义提供商的 provider 段（如 'ustc'）不是方言名，
        # 靠 _provider_name() 推断只会落到"其余"分支——显式解析可让档案复用某个
        # 已知家族的分支（"dialect": "deepseek"），也让内置提供商行为逐位不变。
        client.dialect = resolve_dialect(canonical, config.get('dialect'))
        # 统一从配置注入生成参数（temperature/top_p/max_tokens 等），
        # 调用方无需再手动 client.kwargs.update，避免各入口重复拼装同一套字段。
        # extra_body 也在其中：它的语义是"并入请求体"，是自定义端点接私有能力
        # （如 vLLM 的 chat_template_kwargs）的唯一出口。
        for k in ('max_tokens', 'max_completion_tokens', 'temperature', 'top_p', 'top_k',
                  'frequency_penalty', 'presence_penalty', 'stream', 'n', 'stop',
                  'extra_body'):
            if k in config and config[k] is not None:
                client.kwargs[k] = config[k]
        # 任务级私有参数（如思考强度），供 clone_for_task 按任务注入：
        # tasks: {"sampling": {"reasoning_effort": "low"}, "analysis": {"reasoning_effort": "high"}}
        # 来源有二：档案文件里的 tasks 字段（旧格式，8.6 起从模板中移除）与调用方
        # 传入的 task_params（roles.py 按角色注册表解析，优先级更高）。
        merged_tasks: dict = {}
        legacy_tasks = config.get('tasks')
        if isinstance(legacy_tasks, dict):
            merged_tasks.update(legacy_tasks)
        if isinstance(task_params, dict):
            merged_tasks.update(task_params)
        if merged_tasks:
            client.task_params = merged_tasks
        return client
