"""RAG 文献知识库：嵌入、存储与检索。

设计要点：
- 嵌入模型可配置双后端：本地 sentence-transformers（默认 BAAI/bge-small-zh-v1.5）或
  任意 OpenAI 兼容 /embeddings 的 API 后端。
- 向量存储用 ChromaDB PersistentClient，全程显式传 embeddings，不依赖其默认嵌入函数。
- 所有模型/客户端均懒加载，避免 import 或进程启动时加载 torch 等重依赖。
"""
import json
import os
import re
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

# ── 配置 ──────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "backend": "local",                # local | api
    "model": "BAAI/bge-small-zh-v1.5",  # 本地模型名，或 api 后端模型名
    "api_host": "",
    "api_key": "",
    "api_model": "bge-m3",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "persist_dir": "knowledge_base/chroma_db",
    "collection": "literature",
    "k": 5,
    "query_prefix": "",
    "default_query": "磁流变 颗粒 本构 屈服应力 压缩",
}

_CONFIG_PATH = "rag.config"


# 项目根目录（用于在任意工作目录下都能定位 rag.config）
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_config_path(path: str) -> Path:
    """解析配置路径：NotFound 时优先从项目根目录兜底，避免依赖当前工作目录。"""
    p = Path(path)
    if p.is_absolute() or p.exists() or not path:
        return p
    candidate = _REPO_ROOT / p
    return candidate if candidate.exists() else p


def load_config(path=_CONFIG_PATH) -> dict:
    """读取配置文件，缺失时使用内置默认值。"""
    cfg = dict(DEFAULT_CONFIG)
    resolved = _resolve_config_path(path)
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except FileNotFoundError:
        print(f"[RAG] 配置文件不存在: {resolved}，使用默认配置")
    except Exception as e:
        print(f"[RAG] 读取 {path} 失败，使用默认配置: {e}")
    return cfg


# ── 嵌入模型 ──────────────────────────────────────────────────────────
class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
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

    def embed(self, texts):
        if self._query_prefix:
            texts = [self._query_prefix + t for t in texts]
        vecs = self._get_model().encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32).tolist()


class APIEmbedder(EmbeddingModel):
    """OpenAI 兼容 /embeddings API 后端（如 SiliconFlow、OpenAI）。"""

    def __init__(self, api_host: str, api_key: str, api_model: str):
        self._api_host = api_host.rstrip("/")
        self._api_key = api_key
        self._api_model = api_model

    def embed(self, texts):
        import requests
        url = f"{self._api_host}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        out = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = requests.post(
                url, json={"model": self._api_model, "input": batch},
                headers=headers, timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            out.extend(d["embedding"] for d in sorted(data, key=lambda x: x["index"]))
        return out


_embedder = None
_embedder_key = None
_embedder_lock = threading.Lock()


def get_embedder(config=None) -> EmbeddingModel:
    """进程级懒加载单例（仅首次调用才构造/加载模型）。"""
    global _embedder, _embedder_key
    cfg = config or load_config()
    key = (cfg.get("backend"), cfg.get("model"), cfg.get("api_host"),
           cfg.get("api_key"), cfg.get("api_model"), cfg.get("query_prefix"))
    if _embedder is not None and _embedder_key == key:
        return _embedder
    with _embedder_lock:
        if cfg.get("backend") == "api":
            _embedder = APIEmbedder(
                cfg.get("api_host", ""), cfg.get("api_key", ""), cfg.get("api_model", "bge-m3"))
        else:
            _embedder = SentenceTransformerEmbedder(
                cfg.get("model", DEFAULT_CONFIG["model"]), cfg.get("query_prefix", ""))
        _embedder_key = key
    return _embedder


def reset_embedder():
    global _embedder, _embedder_key
    _embedder = None
    _embedder_key = None


# ── 文本处理 ──────────────────────────────────────────────────────────
def extract_pdf_text(path: str) -> str:
    """用 pymupdf 逐页提取 PDF 全文。"""
    import pymupdf
    doc = pymupdf.open(path)
    parts = []
    try:
        for i in range(doc.page_count):
            parts.append(f"\n--- Page {i + 1} ---\n{doc.load_page(i).get_text()}")
    finally:
        doc.close()
    return "".join(parts)


def _recover_doi(stem: str) -> str | None:
    """从去斜杠文件名恢复 DOI（如 10.1016j.jmmm.2020.166652 -> 10.1016/j.jmmm.2020.166652）。"""
    m = re.match(r"^10\.\d+", stem)
    if not m:
        return None
    prefix, rest = m.group(0), stem[m.end():]
    return prefix + "/" + rest if rest else prefix


def _safe_id(name: str) -> str:
    """把任意字符串转成 Chroma 合法 id 片段。"""
    return re.sub(r"[^0-9A-Za-z_-]", "_", name)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """按空行分段，合并到约 chunk_size；相邻块保留 overlap 字符。"""
    text = (text or "").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        while len(para) > chunk_size:  # 单段超长硬切
            if current:
                chunks.append(current)
            chunks.append(para[:chunk_size])
            para = para[chunk_size:]
        if current and len(current) + len(para) + 1 > chunk_size:
            tail = current[-overlap:] if overlap > 0 else ""
            chunks.append(current)
            current = (tail + "\n" + para) if tail else para
        else:
            current = (current + "\n" + para) if current else para
    if current:
        chunks.append(current)
    return chunks


# ── 知识库 ────────────────────────────────────────────────────────────
class RagKB:
    """ChromaDB 文献知识库。"""

    def __init__(self, config=None):
        self.cfg = config or load_config()
        self._client = None
        self._collection = None

    # 懒加载（避免 import/启动时加载 chroma）
    def _get_client(self):
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.cfg.get("persist_dir", DEFAULT_CONFIG["persist_dir"]))
        return self._client

    def _get_collection(self):
        if self._collection is None:
            col_name = self.cfg.get("collection", DEFAULT_CONFIG["collection"])
            client = self._get_client()
            try:
                # Chroma 1.x 推荐 configuration 写法；旧版用 metadata
                self._collection = client.get_or_create_collection(
                    name=col_name, configuration={"hnsw": {"space": "cosine"}})
            except TypeError:
                self._collection = client.get_or_create_collection(
                    name=col_name, metadata={"hnsw:space": "cosine"})
        return self._collection

    # 写入
    def add_text(self, text: str, source_file: str = "", doi: str = "", title: str = "") -> int:
        """切块、嵌入并写入知识库，返回 chunk 数。id 幂等（upsert）。"""
        chunks = chunk_text(text, self.cfg.get("chunk_size", 500), self.cfg.get("chunk_overlap", 50))
        if not chunks:
            return 0
        base = _safe_id(doi or source_file or "doc")
        ids = [f"{base}::{i}" for i in range(len(chunks))]
        metadatas = [
            {"doi": doi, "title": title, "source_file": source_file, "chunk_index": i}
            for i in range(len(chunks))
        ]
        embeddings = get_embedder(self.cfg).embed(chunks)
        self._get_collection().upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        return len(chunks)

    def add_pdf(self, pdf_path: str, doi: str = "", title: str = "") -> int:
        """把单个 PDF 嵌入知识库，返回 chunk 数。"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(pdf_path)
        text = extract_pdf_text(pdf_path)
        if not text.strip():
            return 0
        source_file = os.path.basename(pdf_path)
        stem = os.path.splitext(source_file)[0]
        if not doi:
            doi = _recover_doi(stem) or ""
        if not title:
            try:
                import pymupdf
                doc = pymupdf.open(pdf_path)
                title = ((doc.metadata or {}).get("title") or "").strip()
                doc.close()
            except Exception:
                title = ""
        title = title or doi or stem
        return self.add_text(text, source_file=source_file, doi=doi, title=title)

    def ingest_dir(self, dir_path: str = "pdf_downloads", limit: int | None = None) -> dict:
        """批量入库目录下所有 PDF；已入库（按 source_file 判重）自动跳过。"""
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(dir_path)
        files = sorted(f for f in os.listdir(dir_path) if f.lower().endswith(".pdf"))
        if limit is not None:
            files = files[:limit]
        col = self._get_collection()
        results = {"ingested": 0, "skipped": 0, "failed": 0, "chunks": 0}
        for fname in files:
            if col.get(where={"source_file": fname}).get("ids"):
                results["skipped"] += 1
                continue
            path = os.path.join(dir_path, fname)
            try:
                n = self.add_pdf(path)
                results["ingested"] += 1
                results["chunks"] += n
                print(f"[RAG] 已入库 {fname}: {n} chunks")
            except Exception as e:
                results["failed"] += 1
                print(f"[RAG] 入库失败 {fname}: {e}")
        return results

    # 检索
    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.count() == 0:
            return []
        q_vec = get_embedder(self.cfg).embed([query])[0]
        res = self._get_collection().query(
            query_embeddings=[q_vec], n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out = []
        for text, meta, dist in zip(docs, metas, dists):
            meta = meta or {}
            out.append({
                "text": text,
                "doi": meta.get("doi", ""),
                "title": meta.get("title", ""),
                "source_file": meta.get("source_file", ""),
                "chunk_index": meta.get("chunk_index"),
                "distance": dist,
            })
        return out

    def get_context(self, query: str, k: int = 5, max_chars: int = 1500) -> str:
        """检索并拼接为 prompt 可注入的文献上下文（截断到 max_chars）。"""
        results = self.search(query, k=k)
        parts, total = [], 0
        for r in results:
            head = r.get("title") or r.get("doi") or r.get("source_file") or "文献"
            block = f"【{head}】\n{r['text']}\n"
            if total + len(block) > max_chars:
                block = block[:max_chars - total]
            parts.append(block)
            total += len(block)
            if total >= max_chars:
                break
        return "\n".join(parts)

    def count(self) -> int:
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    def reset_collection(self):
        """删除并重建 collection（换嵌入模型维度变化时使用）。"""
        name = self.cfg.get("collection", DEFAULT_CONFIG["collection"])
        try:
            self._get_client().delete_collection(name)
        except Exception:
            pass
        self._collection = None


_kb = None
_kb_lock = threading.Lock()


def get_kb(config=None) -> RagKB:
    """进程级单例，供 pipeline / find_best_eq / MCP 复用。"""
    global _kb
    if _kb is None:
        with _kb_lock:
            if _kb is None:
                _kb = RagKB(config)
    return _kb
