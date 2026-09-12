"""配置加载、模型串解析、提供商归一化与客户端工厂。"""
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
# 用于在任意工作目录下定位 llm.config / glm_*.config 等配置文件。
_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_llm_config(path: str = "glm_glm-5.3-flash.config") -> dict:
    """读取 LLM 配置文件（JSON）。

    - 相对路径找不到时自动回退到项目根目录，避免依赖当前工作目录；
    - 返回的 dict 可直接传给 ``ClientFactory.from_config``。
    """
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        cand = _REPO_ROOT / p
        if cand.exists():
            p = cand
    with open(p, "r", encoding="utf-8") as f:
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
    def from_config(config: dict):
        """
        基于 'provider/model' 创建具体客户端。

        必填：config['model']（形如 'provider/model'）。
        选填：config['api_key']、config['base_url']（或兼容别名 'host'）。
        配置先经 normalize_llm_config 统一规范化（host/base_url 兼容、补齐 scheme）。
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
                    f"LLM provider '{provider}' 缺少 API key："
                    f"请在配置文件（如 glm_glm-5.3-flash.config）的 api_key 字段或环境变量 {env_name} 中配置。"
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
        task_params = config.get('tasks')
        if isinstance(task_params, dict):
            client.task_params = task_params
        return client
