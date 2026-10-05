"""嵌入后端：把文本变成向量（本地 sentence-transformers 或远端 API）。

角色归属
--------
``knowledge`` 层 RAG 子系统的**向量化后端**，从 :mod:`drsr_420.knowledge.rag_kb` 拆出。

为什么单独成模块
----------------
"文本怎么变成向量"与"知识库怎么存、怎么检索"是两件事：前者要处理 HTTP 重试、
环境变量取密钥、本地模型懒加载；后者处理分块、持久化、相似度。混在一个文件里时，
想改"换一个嵌入端点"必须翻过 800 行的检索代码。

进程内单例
----------
:func:`get_embedder` 带双重检查锁定（``_embedder_lock`` + ``_embedder_state`` 原子替换）：
并发首调只构造一次模型（本地模型加载昂贵）。这把锁是本模块的私有实现细节，不外露。
"""
from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod

import numpy as np
import requests

from drsr_420.knowledge.rag_config import DEFAULT_CONFIG, load_config

# ── 嵌入模型 ──────────────────────────────────────────────────────────
class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        ...


class SentenceTransformerEmbedder(EmbeddingModel):
    """本地嵌入（sentence-transformers），懒加载模型。"""

    def __init__(self, model_name: str, query_prefix: str = ""):
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts, is_query: bool = False):
        # query_prefix 只作用于查询侧：bge 等模型的指令前缀是按"仅查询"训练的，
        # 旧实现对文档嵌入也加前缀，会把查询指令烧进知识库向量、静默劣化检索
        if self._query_prefix and is_query:
            texts = [self._query_prefix + t for t in texts]
        vecs = self._get_model().encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32).tolist()


class APIEmbedder(EmbeddingModel):
    """OpenAI 兼容 /embeddings API 后端（如智谱、SiliconFlow、OpenAI）。

    内置限流保护：429/5xx 会按指数退避自动重试（优先遵循 Retry-After 响应头），
    并可在请求批次间加入固定间隔，避免触发服务端 429。
    """

    def __init__(self, api_base_url: str, api_key: str, api_model: str,
                 batch_size: int = 64,
                 max_retries: int = 6,
                 backoff_base: float = 1.0,
                 batch_interval: float = 0.3,
                 query_prefix: str = ""):
        self._api_base_url = api_base_url.rstrip("/")
        self._api_key = api_key
        self._api_model = api_model
        self._api_prefix = query_prefix or ""
        self._batch_size = int(batch_size or 64)
        self._max_retries = int(max_retries or 6)
        self._backoff_base = float(backoff_base or 1.0)
        self._batch_interval = float(batch_interval or 0.0)

    def _post(self, url: str, headers: dict, payload: dict):
        """带无限流退避重试的 POST，返回成功响应。"""
        import requests
        import time
        last_exc = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=60)
            except requests.exceptions.RequestException as e:
                # 网络类错误也退避重试
                last_exc = e
                time.sleep(self._backoff_base * (2 ** attempt))
                continue

            # 限流(429)或服务端错误(5xx)：退避重试，优先用 Retry-After
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = self._backoff_base * (2 ** attempt)
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except ValueError:
                        pass
                if attempt >= self._max_retries:
                    resp.raise_for_status()
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp

        if last_exc is not None:
            raise last_exc
        # 理论不可达
        raise requests.exceptions.Timeout("embedding 请求最终失败")

    def embed(self, texts, is_query: bool = False):
        import time
        # api 后端同样支持 query_prefix（此前仅 local 生效，两后端行为分叉）
        if self._api_prefix and is_query:
            texts = [self._api_prefix + t for t in texts]
        url = f"{self._api_base_url}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        out = []
        n = len(texts)
        for i in range(0, n, self._batch_size):
            batch = texts[i:i + self._batch_size]
            resp = self._post(url, headers, {"model": self._api_model, "input": batch})
            data = resp.json()["data"]
            out.extend(d["embedding"] for d in sorted(data, key=lambda x: x["index"]))
            # 批次之间小睡，降低触发限流的概率
            if self._batch_interval and i + self._batch_size < n:
                time.sleep(self._batch_interval)
        return out


_embedder_state = None  # (key, embedder) 元组，整体原子替换（见 get_embedder）
_embedder_lock = threading.Lock()


def _env_key_for_base_url(api_base_url: str) -> str:
    """按 API 端点推断提供商环境变量名并读取密钥（api_key 留空时回退）。"""
    url = (api_base_url or "").lower()
    if "bigmodel" in url or "zhipu" in url:
        return os.getenv("ZHIPU_API_KEY", "")
    if "siliconflow" in url:
        return os.getenv("SILICONFLOW_API_KEY", "")
    if "deepseek" in url:
        return os.getenv("DEEPSEEK_API_KEY", "")
    if "deepinfra" in url:
        return os.getenv("DEEPINFRA_API_KEY", "")
    if "openai" in url:
        return os.getenv("OPENAI_API_KEY", "")
    return ""


def get_embedder(config=None) -> EmbeddingModel:
    """进程级懒加载单例（仅首次调用才构造/加载模型，双重检查锁定）。"""
    global _embedder_state
    cfg = config or load_config()
    api_key = cfg.get("api_key") or _env_key_for_base_url(cfg.get("api_base_url", ""))
    key = (cfg.get("backend"), cfg.get("model"), cfg.get("api_base_url"),
           api_key, cfg.get("api_model"), cfg.get("query_prefix"))
    state = _embedder_state
    if state is not None and state[0] == key:
        return state[1]
    with _embedder_lock:
        # 锁内复查：并发首调时防止重复加载 torch/模型
        state = _embedder_state
        if state is not None and state[0] == key:
            return state[1]
        if cfg.get("backend") == "api":
            embedder = APIEmbedder(
                cfg.get("api_base_url", ""), api_key, cfg.get("api_model", "bge-m3"),
                batch_size=cfg.get("embed_batch_size", 64),
                max_retries=cfg.get("embed_max_retries", 6),
                backoff_base=cfg.get("embed_backoff_base", 1.0),
                batch_interval=cfg.get("embed_batch_interval", 0.3),
                query_prefix=cfg.get("query_prefix", ""))
        else:
            embedder = SentenceTransformerEmbedder(
                cfg.get("model", DEFAULT_CONFIG["model"]), cfg.get("query_prefix", ""))
        # 以单一元组原子发布 (key, embedder)，杜绝无锁快路径读到"新实例+旧键"的撕裂
        _embedder_state = (key, embedder)
    return _embedder_state[1]


def reset_embedder():
    global _embedder_state
    with _embedder_lock:
        _embedder_state = None

