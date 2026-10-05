"""RAG 文献知识库：嵌入、存储与检索。

设计要点：
- 嵌入模型可配置双后端：本地 sentence-transformers（默认 BAAI/bge-small-zh-v1.5）或
  任意 OpenAI 兼容 /embeddings 的 API 后端。
- 向量存储用 ChromaDB PersistentClient，全程显式传 embeddings，不依赖其默认嵌入函数。
- 所有模型/客户端均懒加载，避免 import 或进程启动时加载 torch 等重依赖。

档案（``config/rag.config``）的定位、读取与校验在 :mod:`drsr_420.knowledge.rag_config`；
这里仍**转发**那几个名字，历史导入路径（``rag_kb.load_config`` / ``rag_kb.DEFAULT_CONFIG``）
全部照旧可用。
"""
import os
import re
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from drsr_420.knowledge.rag_config import (      # noqa: F401  （转发：对外契约不变）
    CONFIG_NAME,
    DEFAULT_CONFIG,
    REPO_ROOT,
    load_config,
)

from drsr_420.knowledge.pdf_metadata import (      # noqa: F401  （转发：对外契约不变）
    doi_from_pdf_text,
    extract_pdf_text,
    first_page_title,
    is_placeholder_title,
    resolve_doi,
    resolve_title,
    safe_id,
    same_doi_text,
)
from drsr_420.knowledge.rag_embedder import (      # noqa: F401  （转发：对外契约不变）
    APIEmbedder,
    EmbeddingModel,
    SentenceTransformerEmbedder,
    get_embedder,
    reset_embedder,
)
from drsr_420.knowledge.text_chunking import chunk_text  # noqa: F401  （转发）


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
            persist = self.cfg.get("persist_dir", DEFAULT_CONFIG["persist_dir"])
            # 相对路径统一解析到项目根目录，避免在不同 cwd 下产生多个知识库；绝对路径尊重原样
            p = Path(persist)
            if not p.is_absolute():
                persist = str(REPO_ROOT / p)
            self._client = chromadb.PersistentClient(path=persist)
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
        base = safe_id(doi or source_file or "doc")
        ids = [f"{base}::{i}" for i in range(len(chunks))]
        metadatas = [
            {"doi": doi, "title": title, "source_file": source_file, "chunk_index": i}
            for i in range(len(chunks))
        ]
        embeddings = get_embedder(self.cfg).embed(chunks)
        self._get_collection().upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        return len(chunks)

    def add_pdf(self, pdf_path: str, doi: str = "", title: str = "") -> int:
        """把单个 PDF 嵌入知识库，返回 chunk 数。

        DOI 与标题只在调用方没显式给出时才推断，推断链见 :func:`resolve_doi`
        （论文自己印的 DOI，权威 → 无歧义的文件名恢复 → 空串）与
        :func:`resolve_title`（元数据 → 首页最大字号 → 文件名 → 空串）。列表页
        不能再出现"用文件名/DOI 冒充标题"的元数据——实测 report.md 的参考文献
        因此显示成 ``10.216561000-0887.380021.pdf``。
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(pdf_path)
        text = extract_pdf_text(pdf_path)
        if not text.strip():
            return 0
        source_file = os.path.basename(pdf_path)
        stem = os.path.splitext(source_file)[0]
        if not doi:
            doi = resolve_doi(text, stem)
        meta_title = ""
        page_title = ""
        if not title:
            try:
                import pymupdf
                doc = pymupdf.open(pdf_path)
                try:
                    meta_title = ((doc.metadata or {}).get("title") or "").strip()
                finally:
                    doc.close()  # 异常路径也要释放文件句柄（Windows 上句柄不关会锁文件）
            except Exception:
                meta_title = ""
            # 元数据已给出真标题时不再开第二次 PDF（首页字号启发式是兜底手段）
            if is_placeholder_title(meta_title):
                page_title = first_page_title(pdf_path)
        title = resolve_title(title, meta_title, page_title, stem)
        return self.add_text(text, source_file=source_file, doi=doi, title=title)

    def ingest_dir(self, dir_path: str = "pdf_downloads", limit: int | None = None) -> dict:
        """批量入库目录下所有 PDF；已入库（按 source_file 判重）自动跳过。"""
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(dir_path)
        files = sorted(f for f in os.listdir(dir_path) if f.lower().endswith(".pdf"))
        if limit is not None:
            files = files[:limit]
        col = self._get_collection()
        # 文件间节流：rag.config 里的 embed_file_interval 此前没有任何代码读取，
        # 用户以为限流生效、实际每个文件的请求批次背靠背打向 API
        file_interval = float(self.cfg.get("embed_file_interval", 0) or 0)
        results = {"ingested": 0, "skipped": 0, "failed": 0, "chunks": 0}
        for idx, fname in enumerate(files):
            if file_interval and idx:
                import time
                time.sleep(file_interval)
            try:
                # 判重查询也纳入容错：一次 chroma 读失败不应中断整批入库
                if col.get(where={"source_file": fname}).get("ids"):
                    results["skipped"] += 1
                    continue
                path = os.path.join(dir_path, fname)
                n = self.add_pdf(path)
                results["ingested"] += 1
                results["chunks"] += n
                print(f"[RAG] 已入库 {fname}: {n} chunks")
            except Exception as e:
                results["failed"] += 1
                print(f"[RAG] 入库失败 {fname}: {e}")
        return results

    # 元数据修复
    def repair_metadata(self, pdf_dir: str = "pdf_downloads",
                        dry_run: bool = False) -> dict:
        """就地修复已入库文献的 doi/title 元数据（**不重嵌入、不改 chunk 文本**）。

        历史入库用"从去斜杠文件名恢复 DOI + ``title or doi or stem``"的兜底，给
        知识库留下两类坏元数据（实测 report.md 参考文献显示成
        ``10.216561000-0887.380021.pdf`` 这类"标题"）：
        1. **假 DOI**——文件名抹掉的是哪个 ``/`` 无从判断，旧恢复规则猜出的
           DOI 指向错误文章（``10.11221/.3479045`` 实为 ``10.1122/1.3479045``）；
        2. **假标题**——元数据为空时退化成 doi/stem。

        修复判据与 :meth:`add_pdf` 现行推断链一致（权威度从高到低）：
        - DOI：论文正文印刷的 DOI（:func:`doi_from_pdf_text`，扫描到参考文献之前）
          → 无歧义文件名恢复（:func:`_recover_doi`，后缀含 ``-`` 或以数字开头一律
          拒绝）→ 原值保留；原值恰是旧规则从文件名猜的（去斜杠后 == 主干）时按新
          规则重判。文件名是 DOI 形态而印刷值与它对不上时不信印刷值（见
          :func:`resolve_doi`）。
        - 标题：PDF 元数据 → 首页最大字号（:func:`first_page_title`）→
          非 DOI 形态的文件名 → 空串；新链给不出且旧值非占位物时保留旧值。

        PDF 缺失的条目无法复核，原样跳过；修复按 source_file 幂等，重复执行
        第二次应为 0 修复。

        Args:
            pdf_dir: 原始 PDF 所在目录（按 source_file 对名）。
            dry_run: True 时只报告将做的修改，不写库。

        Returns:
            dict：``files``（库内文献数）/ ``repaired`` / ``unchanged`` /
            ``skipped``（PDF 缺失）/ ``changes``（逐文件的新旧值对照）。
        """
        col = self._get_collection()
        data = col.get(include=["metadatas"])
        ids = data.get("ids") or []
        metas = data.get("metadatas") or []
        by_file: dict[str, dict] = {}
        for cid, meta in zip(ids, metas):
            meta = meta or {}
            sf = str(meta.get("source_file") or "")
            if not sf:
                continue
            entry = by_file.setdefault(sf, {"ids": [], "doi": "", "title": "",
                                            "chunk_index": {}})
            entry["ids"].append(cid)
            entry["doi"] = entry["doi"] or str(meta.get("doi") or "")
            entry["title"] = entry["title"] or str(meta.get("title") or "")
            entry["chunk_index"][cid] = meta.get("chunk_index")

        summary = {"files": len(by_file), "repaired": 0, "unchanged": 0,
                   "skipped": 0, "changes": []}
        for sf, entry in sorted(by_file.items()):
            pdf_path = os.path.join(pdf_dir, sf)
            if not os.path.isfile(pdf_path):
                summary["skipped"] += 1
                continue
            stem = os.path.splitext(sf)[0]
            text = extract_pdf_text(pdf_path)

            # DOI 推断链与 add_pdf 同源（resolve_doi）：印刷值 → 无歧义文件名恢复 → 原值。
            # 原值恰是旧规则从文件名猜的（去斜杠后 == 主干）时按新规则重判。
            new_doi = resolve_doi(text, stem, entry["doi"])
            if same_doi_text(new_doi, entry["doi"]):
                new_doi = entry["doi"]   # 只有大小写差异：不算修改，避免无意义写入

            # 标题：现行推断链重算
            meta_title, page_title = "", ""
            try:
                import pymupdf
                doc = pymupdf.open(pdf_path)
                try:
                    meta_title = ((doc.metadata or {}).get("title") or "").strip()
                finally:
                    doc.close()
            except Exception:
                meta_title = ""
            if is_placeholder_title(meta_title):
                page_title = first_page_title(pdf_path)
            new_title = resolve_title("", meta_title, page_title, stem)
            if new_title == "" and not is_placeholder_title(entry["title"]):
                new_title = entry["title"]

            if new_doi == entry["doi"] and new_title == entry["title"]:
                summary["unchanged"] += 1
                continue
            summary["changes"].append({
                "source_file": sf,
                "doi": [entry["doi"], new_doi],
                "title": [entry["title"], new_title],
                "chunks": len(entry["ids"]),
            })
            if not dry_run:
                col.update(
                    ids=entry["ids"],
                    metadatas=[{"doi": new_doi, "title": new_title,
                                "source_file": sf,
                                "chunk_index": entry["chunk_index"][cid]}
                               for cid in entry["ids"]],
                )
            summary["repaired"] += 1
        return summary

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
