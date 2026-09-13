"""RAG 档案的定位、读取与校验（``config/rag.config``）。

为什么单独成文件
================
原先这些都在 ``rag_kb.py`` 顶部，与"嵌入模型 / 分块 / Chroma 读写"挤在一起。两者的
变化原因完全不同：配置随"命名约定与校验规则"变（这轮就加了旧键拒绝与端点格式校验），
而运行时随"嵌入后端与向量库 API"变。分开后 ``rag_kb.py`` 回到 500 行预算以内，
配置侧也才放得下完整的错误提示。

定位规则与 LLM 档案**共用同一套**（``llm.factory.locate_config``）：依次尝试当前工作
目录、仓库根与 ``config/``，因此库代码里不需要出现 ``*.config`` 字面量（有护栏拦这点）。
"""
from __future__ import annotations

import json
from pathlib import Path

#: 内置默认配置（档案缺失时全用它；档案给的值会覆盖同名键）。
DEFAULT_CONFIG = {
    "backend": "local",                # local | api
    "model": "BAAI/bge-small-zh-v1.5",  # 本地模型名，或 api 后端模型名
    "api_base_url": "",
    "api_key": "",
    "api_model": "bge-m3",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "persist_dir": "knowledge_base/chroma_db",
    "collection": "literature",
    "k": 5,
    "query_prefix": "",
    # "" 表示"按问题自动推导检索词"（调用方回退到背景文本/自变量）。
    # 此前放的是固定 MRF 关键词，且因 load_config 恒合并本默认值，
    # 导致任何非 MRF 问题也永远用这串中文检索、调用方的按问题回退成为死代码。
    # 需要固定检索词时在 rag.config 里显式配置 default_query。
    "default_query": "",
    # "embed_batch_size": 64,
    # "embed_max_retries": 6,
    # "embed_backoff_base": 1.0,
    # "embed_batch_interval": 0.3,
}

#: RAG 档案名（不含后缀）。文件定位统一交给 ``llm.factory.locate_config``。
_CONFIG_NAME = "rag"

# 项目根目录：知识库持久化目录等相对路径统一锚定到这里，避免不同 cwd 下产生多个库。
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_config_path(path: str | None = None) -> Path:
    """解析配置路径，交由 ``llm.factory.locate_config`` 统一在仓库根与 ``config/`` 下定位。"""
    from drsr_420.llm.factory import locate_config

    return locate_config(path or _CONFIG_NAME)


#: 已废弃的配置键 -> 现用键。
#: RAG 端点与 LLM 档案统一用 ``base_url`` 命名（``*_host`` 这类拼写全部下线）。
_RENAMED_KEYS = {"api_host": "api_base_url"}


def _reject_renamed_keys(loaded: dict, source) -> None:
    """档案里出现已废弃的键名时报错，并把改名方法写在错误里。"""
    for old, new in _RENAMED_KEYS.items():
        if old in loaded:
            raise ValueError(
                f"{source} 里的配置键 {old!r} 已废弃，请改名为 {new!r}"
                f"（收到 {old}={loaded[old]!r}）。端点统一用 base_url 命名。")


def _validate_endpoint(loaded: dict, source) -> None:
    """校验 API 后端的端点：必须是**完整 URL**，不接受裸主机域名。

    与 LLM 档案同一条规则（见 ``llm/client.require_absolute_url``）：统一写成
    ``https://<主机>/v1`` 这种带路径的完整端点。

    只在 ``backend == "api"`` 时检查——local 后端根本不用端点，写了也不该被拦。
    空端点尤其要拦：拼接出的 URL 会变成 ``/embeddings``，报一个与"键名/取值写错"
    毫无关系的 ``MissingSchema``。
    """
    if loaded.get("backend", DEFAULT_CONFIG["backend"]) != "api":
        return
    url = str(loaded.get("api_base_url") or "").strip()
    if not url:
        raise ValueError(
            f"{source}: backend=\"api\" 时必须给出 api_base_url"
            f"（完整 URL，如 https://api.siliconflow.cn/v1）")
    if not url.startswith(("http://", "https://")):
        raise ValueError(
            f"{source}: api_base_url 必须是完整 URL（以 http:// 或 https:// 开头），"
            f"收到 {url!r}；不要写裸主机域名")


def load_config(path: str | None = None) -> dict:
    """读取配置档案，缺失时使用内置默认值。

    Raises:
        ValueError: 档案里用了已废弃的键名（见 :data:`_RENAMED_KEYS`），或 ``backend="api"``
            却没给出合法的端点。静默忽略旧键会让嵌入请求打到一个空地址，所以这里必须
            响亮地失败并给出改法。
    """
    cfg = dict(DEFAULT_CONFIG)
    resolved = _resolve_config_path(path)
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except FileNotFoundError:
        print(f"[RAG] 配置文件不存在: {resolved}，使用默认配置")
        return cfg
    except Exception as e:
        print(f"[RAG] 读取 {path} 失败，使用默认配置: {e}")
        return cfg
    _reject_renamed_keys(loaded, resolved)
    _validate_endpoint(loaded, resolved)
    cfg.update(loaded)
    return cfg
