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
    DEFAULT_CONFIG,
    _CONFIG_NAME,
    _REPO_ROOT,
    load_config,
)

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


#: 论文正文里印出来的 DOI（含页脚 "http://dx.doi.org/10.xxxx/yyy" 形态）。
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>()\[\],;]+")


def _looks_like_doi(name: str) -> bool:
    """字符串是不是 DOI/DOI 去掉斜杠后的形态（用来拒绝把 DOI 当标题）。"""
    return bool(re.match(r"^10\.\d{4,9}", str(name or "").strip()))


def doi_from_pdf_text(text: str, head_chars: int = 3000) -> str | None:
    """从 PDF 文本里取**印出来的 DOI**，取不到返回 ``None``。

    这是恢复 DOI 的权威来源。去斜杠文件名无法唯一还原（``10.216561000-0887.380021``
    实际是 ``10.21656/1000-0887.380021``），而论文自己印的 DOI 没有歧义。只看开头
    若干字符：正文靠前的 DOI 属于本文，参考文献里的 DOI 是别人的。
    """
    if not text:
        return None
    m = _DOI_RE.search(str(text)[:head_chars])
    return m.group(0).rstrip(".,;)]") if m else None


def _recover_doi(stem: str) -> str | None:
    """从去斜杠文件名恢复 DOI（如 10.1016j.jmmm.2020.166652 -> 10.1016/j.jmmm.2020.166652）。

    只在**无歧义**时恢复：文件名抹掉的是哪一个 ``/``（注册号几位）无从判断，后缀里
    一旦出现 ``-``（``10.216561000-0887.380021``、``10.10880964-17262412125005``），
    它既可能是 DOI 自带的连字符、也可能是被抹掉的斜杠，猜出来的 DOI 多半是错的——
    实测这两种都进了 knowledge base 的元数据。宁可留空，也不要写一个假 DOI。
    """
    m = re.match(r"^10\.\d{4,9}", stem)
    if not m:
        return None
    prefix, rest = m.group(0), stem[m.end():]
    if not rest or "-" in rest:
        return None
    return prefix + "/" + rest


def _is_placeholder_title(text: str) -> bool:
    """判断一段"标题"其实是占位物：排版软件痕迹、文件名或 DOI。"""
    s = str(text or "").strip()
    if not s:
        return True
    low = s.lower()
    return (low.startswith("microsoft word") or low.endswith(".pdf")
            or low.endswith(".doc") or _looks_like_doi(s))


def _first_page_title(pdf_path: str, limit: int = 300) -> str:
    """用首页**最大字号**的那几行当标题（元数据 title 常为空或是排版软件文件名）。

    与 ``knowledge/tools/read_paper._first_page_title_candidate`` 同类启发式；这里另起
    一份是为了不让知识库模块反向依赖 knowledge/tools（那边顶层 import 了 requests/llm，
    而本模块的既有约定是重依赖一律懒加载）。
    """
    try:
        import pymupdf
        doc = pymupdf.open(pdf_path)
    except Exception:
        return ""
    try:
        data = doc.load_page(0).get_text("dict")
    except Exception:
        return ""
    finally:
        doc.close()
    largest, lines = 0.0, []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans") or []
            text = "".join((s.get("text") or "") for s in spans).strip()
            if not text:
                continue
            size = max((s.get("size") or 0.0) for s in spans) if spans else 0.0
            if size > largest + 0.1:
                largest, lines = size, [text]
            elif lines and abs(size - largest) <= 0.1:
                lines.append(text)
    return " ".join(lines).strip()[:limit]


def _resolve_title(explicit: str, meta_title: str, page_title: str, stem: str) -> str:
    """标题回退链：显式传入 → PDF 元数据 → 首页最大字号 → 文件名（不像 DOI 时）→ 空串。

    **绝不用 DOI 冒充标题**：历史实现在两者都取不到时退化成 ``doi or stem``，于是
    explain.md 的参考文献出现 ``10.216561000-0887.380021.pdf`` 这种"标题"（实测），
    读者无从判断是哪篇文献，模型也无法据此判断相关性。
    """
    for cand in (explicit, meta_title, page_title):
        if not _is_placeholder_title(cand):
            return str(cand).strip()
    if stem and not _looks_like_doi(stem):
        return stem
    return ""


def _safe_id(name: str) -> str:
    """把任意字符串转成 Chroma 合法 id 片段。"""
    return re.sub(r"[^0-9A-Za-z_-]", "_", name)


def _is_section_heading(line: str) -> bool:
    """判断一行是否为文献小节标题（启发式，宁缺勿滥：误判只会多切一刀，漏判回退整段合并）。

    识别五类常见形态：
    - Markdown 标题（``## Methods``）；
    - 数字编号（``1. Introduction`` / ``2.1 Materials``）；
    - 罗马数字编号（``II. EXPERIMENTAL``）；
    - 全大写行（PDF 提取常见的 ``INTRODUCTION``）；
    - 常见节名整行（Abstract / Conclusions / References ...，大小写不敏感）。
    """
    s = line.strip()
    if not s or len(s) > 80:                      # 标题都是短行，长行是正文/列表项
        return False
    if re.match(r"^---\s*Page \d+\s*---$", s):    # extract_pdf_text 的页码标记
        return False
    if s.rstrip(".").isdigit():                   # 纯数字行是页码/公式，不是标题
        return False
    if any(p.match(s) for p in _SECTION_HEADING_RES):
        return True
    return bool(_SECTION_NAME_RE.match(s))


#: 小节标题的形态清单（顺序无关；_is_section_heading 里统一加长度/页码护栏）
_SECTION_HEADING_RES = (
    re.compile(r"^#{1,6}\s+\S"),                              # Markdown 标题
    # 数字编号：编号后必须是字母开头的真实文字——否则 PDF 提取的公式/页码碎片
    # （如 "1 2"、"0. 8"）会被当成标题，切出一堆 3 字符的垃圾块（实测 15% 的块 <120 字符）
    re.compile(r"^\d+(?:\.\d+)*[.)]?\s+[A-Za-z][A-Za-z\s\-']{2,}"),
    # 罗马数字：同样要求后跟字母文字（"II. EXPERIMENTAL"）
    re.compile(r"^(?:I{1,3}|IV|V|VI{0,3}|IX|X|XI|XII)\.\s+[A-Za-z]"),
    re.compile(r"^[A-Z][A-Z0-9 ,\-]{4,60}$"),                 # 全大写行
)
#: 常见节名整行（可带编号前缀与冒号）
_SECTION_NAME_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?\s+)?"
    r"(?:abstract|introduction|background|motivation|methods?|materials\s+and\s+methods|"
    r"experimental(?:\s+section)?|results?(?:\s+and\s+discussion)?|discussion|"
    r"conclusions?|summary|references|acknowledg?ments?|appendix[ a-z]*)[.:]?\s*$",
    re.IGNORECASE)


def _merge_paragraphs(paragraphs: list[str], chunk_size: int, overlap: int) -> list[str]:
    """把段落贪心合并到 ≤ chunk_size；超长段落硬切（相邻片段保留 overlap）。

    即旧版 chunk_text 的主体行为，现作为"无小节结构"的回退与"超长小节"的
    二级切分器复用。
    """
    overlap = max(0, min(int(overlap), int(chunk_size) - 1))
    step = chunk_size - overlap  # 硬切步长：相邻硬切片段之间保留 overlap
    chunks, current = [], ""
    for para in paragraphs:
        if len(para) > chunk_size:
            # 先冲刷未完成的 current：旧实现把它在每次硬切前重复 append 却不清空，
            # 导致同一短块在知识库里出现多次、污染检索结果
            if current:
                chunks.append(current)
                current = ""
            while len(para) > chunk_size:  # 单段超长硬切（带重叠）
                chunks.append(para[:chunk_size])
                para = para[step:]
        if current and len(current) + len(para) + 1 > chunk_size:
            tail = current[-overlap:] if overlap > 0 else ""
            chunks.append(current)
            current = (tail + "\n" + para) if tail else para
        else:
            current = (current + "\n" + para) if current else para
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """按文献小节/段落语义分块（取代旧的固定大小分块）。

    切分策略（自顶向下）：
    1. 先按小节标题切（Markdown / 数字编号 / 罗马数字 / 全大写行 / 常见节名），
       一个小节一个语义块——检索命中的片段天然自带"它属于论文哪一节"的语境；
    2. 小节超过 chunk_size 时，在**小节内部**按空行分段贪心合并，仍保持单块
       ≤ chunk_size，且每个子块都带上小节标题前缀（子块自带上下文）；
    3. 超长段落（无空行）硬切，相邻片段保留 overlap；
    4. 全文检测不到任何小节标题时，退回旧的"空行分段 + 合并"行为。

    注意：换分块策略后必须重建知识库（``rag_build --ingest --rebuild``），
    否则旧 chunk 与新 chunk 混存、``ingest_dir`` 按文件判重会跳过重切。
    """
    text = (text or "").strip()
    if not text:
        return []
    chunk_size = max(1, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))

    lines = text.split("\n")
    sections: list[list[str]] = [[]]      # 每个小节是行列表（首行可能是标题行）
    for line in lines:
        if _is_section_heading(line):
            sections.append([line])
        else:
            sections[-1].append(line)

    if len(sections) == 1:
        # 无小节结构：退回旧行为（空行分段 + 合并 + 超长硬切）
        return _merge_paragraphs(
            [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()],
            chunk_size, overlap)

    chunks: list[str] = []
    for sec_lines in sections:
        sec = "\n".join(sec_lines).strip()
        if not sec:
            continue
        first = sec_lines[0].strip()
        heading = first if _is_section_heading(first) else ""
        if heading:
            body = "\n".join(sec_lines[1:]).strip()
            if not body:
                chunks.append(heading)     # 只有标题的空节：标题本身也值得可检索
                continue
        else:
            body = sec
        # 子块预算扣除标题前缀长度，保证"标题 + 正文"整体仍 ≤ chunk_size
        budget = max(1, chunk_size - len(heading) - 1) if heading else chunk_size
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        for piece in _merge_paragraphs(paras, budget, overlap):
            chunks.append(f"{heading}\n{piece}" if heading else piece)
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
            persist = self.cfg.get("persist_dir", DEFAULT_CONFIG["persist_dir"])
            # 相对路径统一解析到项目根目录，避免在不同 cwd 下产生多个知识库；绝对路径尊重原样
            p = Path(persist)
            if not p.is_absolute():
                persist = str(_REPO_ROOT / p)
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
        """把单个 PDF 嵌入知识库，返回 chunk 数。

        DOI 与标题只在调用方没显式给出时才推断，推断链见 :func:`doi_from_pdf_text`
        （论文自己印的 DOI，权威）→ :func:`_recover_doi`（仅无歧义时）与
        :func:`_resolve_title`（元数据 → 首页最大字号 → 文件名 → 空串）。列表页
        不能再出现"用文件名/DOI 冒充标题"的元数据——实测 explain.md 的参考文献
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
            doi = doi_from_pdf_text(text) or _recover_doi(stem) or ""
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
            if _is_placeholder_title(meta_title):
                page_title = _first_page_title(pdf_path)
        title = _resolve_title(title, meta_title, page_title, stem)
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
