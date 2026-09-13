"""配置加载、模型串解析、提供商归一化与客户端工厂。

本模块负责 **Q1（连接谁 / 用哪把钥匙）与 Q2（生成参数）**，并拥有档案文件的
**命名约定与定位规则**；**Q3（哪个角色用哪套）** 在 :mod:`drsr_420.llm.roles`。

命名约定（唯一权威是文件内的 ``model`` 字段，文件名只是给人看的标签）：

* 目录：``<仓库根>/config/``；
* 档案：``config/<提供商>_<模型>.config``（不入库，含密钥）；
* 模板：``config/<提供商>_<模型>.config.example``（入库，api_key 留空）。
"""
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Type

from drsr_420.llm.client import LLMClient
from drsr_420.llm.providers import (
    BltClient,
    CSTCloudClient,
    DeepInfraClient,
    DeepSeekClient,
    OllamaClient,
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
    """统一规范化 LLM 配置，消除 host/base_url 双键与缺 scheme 等历史混乱。

    - ``base_url`` 与 ``host`` 兼容：两者都存在时 ``base_url`` 优先，输出统一为 ``base_url``；
    - 无 scheme 的地址自动补 ``https://``，避免 requests 报 "No scheme supplied"；
    - 校验 ``model`` 必须是 'provider/model' 格式，配置错误尽早暴露。

    返回规范化后的新 dict，不改动入参。
    """
    cfg = dict(config)
    if not cfg.get('base_url') and cfg.get('host'):
        cfg['base_url'] = cfg['host']
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
    def from_config(config: dict, task_params: dict | None = None):
        """
        基于 'provider/model' 创建具体客户端。

        必填：config['model']（形如 'provider/model'）。
        选填：config['api_key']、config['base_url']（或兼容别名 'host'）。
        配置先经 normalize_llm_config 统一规范化（host/base_url 兼容、补齐 scheme）。

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

        # 设置默认 base_url 并构造对应客户端（表驱动，替代原先 7 分支 if/elif）
        canonical = ClientFactory._PROVIDER_ALIASES.get(provider, provider)
        spec = ClientFactory._PROVIDER_SPECS.get(canonical)
        if spec is None:
            raise ValueError(
                f"不支持的提供商: {provider}，请使用 'deepseek'、'siliconflow'、"
                f"'deepinfra'、'blt'、'cstcloud'、'glm' 或 'ollama'")
        client_cls, env_var, default_base_url = spec

        # ollama 不强制 api_key；其余提供商走 _require_api_key（缺失时回退环境变量）
        if env_var is None:
            resolved_key = api_key or ''
        else:
            resolved_key = _require_api_key(api_key, env_var)

        # base_url：传入优先，否则用 spec 默认；spec 默认为 None 时由客户端类
        # 自行从环境变量兜底（如 BltClient 读 BLT_API_BASE）
        final_base_url = base_url or default_base_url
        client = client_cls(api_key=resolved_key, model=model, base_url=final_base_url)

        # 必须写别名规范化后的 canonical：_provider_name() 只在 provider 为空时才
        # 从 URL 推断，直接写原始别名（'zhipu'/'bigmodel'/'glm4'…）会让
        # _adapt_payload 的所有提供商分支（glm 的 max_tokens 改名、thinking、
        # deepseek 等）静默跳过——别名用户丢失按任务注入的 reasoning_effort。
        client.provider = canonical
        # 统一从配置注入生成参数（temperature/top_p/max_tokens 等），
        # 调用方无需再手动 client.kwargs.update，避免各入口重复拼装同一套字段。
        for k in ('max_tokens', 'max_completion_tokens', 'temperature', 'top_p', 'top_k',
                  'frequency_penalty', 'presence_penalty', 'stream', 'n', 'stop'):
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
