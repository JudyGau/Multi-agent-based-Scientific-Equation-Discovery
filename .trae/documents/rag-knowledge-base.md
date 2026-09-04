# RAG 文献知识库实现计划

## Context（背景与目标）

项目是多智能体科学方程发现系统，`pdf_downloads/` 已有约 140 篇磁流变（MRF）文献 PDF，`read_paper.py` 能用 pymupdf 提取全文，但**没有任何检索/记忆能力**：LLM 每次分析数据、解释公式都无法利用已有文献。

`pipeline.py` 第 135-142 行留有被注释的 RAG 设计（chromadb + `BAAI/bge-small-zh-v1.5` + `get_context`），但从未实现。

目标：新增 RAG 知识库，把文献嵌入向量库并支持检索，集成进三处入口——CLI 批处理 + Python API、MCP 工具（LLM agent 运行中可入库/检索）、pipeline 检索增强（初次数据分析 / 公式解释时注入相关文献上下文）。

**用户已确认的选型**：
- 嵌入模型：可配置双后端（默认本地 `BAAI/bge-small-zh-v1.5`；预留 OpenAI 兼容 `/embeddings` API 后端）
- 向量存储：ChromaDB `PersistentClient`，路径 `knowledge_base/chroma_db`
- 集成入口：CLI + Python API、MCP 工具、pipeline 检索增强（三者都要）

## 新建文件

### 1. `drsr_420/rag_kb.py`（核心模块，自包含，不 import read_paper/llm 避免重依赖）

- **配置**：`DEFAULT_CONFIG` 内置默认值 + 读取项目根 `rag.config`（`.gitignore` 已有 `*.config`）。字段：`backend(local|api)`、`model`、`api_host/api_key/api_model`、`chunk_size(500)`、`chunk_overlap(50)`、`persist_dir("knowledge_base/chroma_db")`、`collection("literature")`、`default_query`、`query_prefix("")`。
- **EmbeddingModel 抽象**：
  - `SentenceTransformerEmbedder(model, query_prefix)` — `encode(texts, normalize_embeddings=True)`，懒加载 `SentenceTransformer`（首次调用才加载，避免 import/启动拖慢）
  - `APIEmbedder(api_host, api_key, api_model)` — `POST {host}/embeddings`，OpenAI 兼容 body `{"model", "input"}`，Authorization Bearer，分批（每批 64）
  - `get_embedder(config=None)`：进程级懒加载单例（threading.Lock 保护）
- **工具函数**：
  - `extract_pdf_text(path)` — pymupdf 逐页 `get_text()`，拼接 `\n--- Page N ---\n`（复用 read_paper.py:154-160 写法，但**自行实现**，不 import read_paper）
  - `_recover_doi(stem)` — `re.match(r"^10\.\d+", stem)` 匹配前缀后插入 `/`（如 `10.1016j.jmmm.2020.166652` → `10.1016/j.jmmm.2020.166652`；仅作展示 metadata，唯一键仍用文件名）
  - `chunk_text(text, chunk_size, overlap)` — 按 `\n\n` 段落后合并到约 chunk_size，相邻块保留 overlap 字符
- **`RagKB` 类**：
  - `_client()` / `_collection()` 懒建 `PersistentClient` + `get_or_create_collection`（不传 embedding_function，全程显式传 `embeddings=`；`configuration={"hnsw":{"space":"cosine"}}`，TypeError 时回退旧 `metadata` 写法；因向量已归一化，默认 L2 排序也等价）
  - `add_text(text, source_file, doi="", title="") -> int` — 切块 → 嵌入 → `col.upsert(ids=[f"{safe_doi}::{i}"], ...)`（**用 upsert 保证幂等**，容忍中断重跑）
  - `add_pdf(pdf_path, doi="", title="") -> int` — 提取全文 → title 三级回退（metadata.title → 恢复的 doi → 文件名）→ `source_file=basename`
  - `ingest_dir(dir_path="pdf_downloads", limit=None) -> dict` — 遍历 `*.pdf`；`col.get(where={"source_file": f})` 非空则跳过（增量幂等）
  - `search(query, k=5) -> list[dict]` — 嵌入查询 → `col.query(query_embeddings=..., n_results=k, include=["documents","metadatas","distances"])`，解析嵌套 list
  - `get_context(query, k=5, max_chars=1500) -> str` — 拼接 `【标题/DOI】\n正文\n`，供 prompt 注入
  - `count()`、`reset_collection()`（换模型维度变化时重建，配 CLI `--rebuild`）
- **`get_kb(config=None)`**：进程级单例，供 pipeline / find_best_eq / MCP 复用。

### 2. `drsr_420/rag_build.py`（CLI，单进程写入，避开并发）

- `python -m drsr_420.rag_build --ingest --dir pdf_downloads [--limit N] [--rebuild]`
- `python -m drsr_420.rag_build --query "椭球 颗粒 磁流变 本构 屈服应力" [--k 5]`
- argparse + `if __name__ == "__main__"`，无模块级副作用。

### 3. `rag.config`（项目根，JSON）

```json
{
  "backend": "local",
  "model": "BAAI/bge-small-zh-v1.5",
  "api_host": "",
  "api_key": "",
  "api_model": "bge-m3",
  "chunk_size": 500,
  "chunk_overlap": 50,
  "persist_dir": "knowledge_base/chroma_db",
  "collection": "literature",
  "k": 5,
  "default_query": "磁流变 颗粒 本构 屈服应力 压缩"
}
```

## 修改文件

### 4. `requirements.txt`
追加（Python 3.12 / Windows / numpy==1.26.4 兼容版本）：
```
chromadb==1.5.9
sentence-transformers==5.6.1
```
不要上 sentence-transformers 6.x（transformers v5 下限有 numpy>=2 冲突）。安装会带 torch（CPU 版约 2-3GB）。

### 5. `drsr_420/tools/mcp_server.py`
注册两个工具，**函数体内懒加载** `get_kb()`（服务启动零开销）：
```python
def ingest_paper(pdf_path: str, doi: str = "", title: str = "") -> str
def search_kb(query: str, k: int = 5) -> str
```
返回 `json.dumps(..., ensure_ascii=False)`。相对 pdf_path 相对 MCP 子进程 cwd（=项目根）。

### 6. `drsr_420/tools/tools_description.py`
追加两个 OpenAI 兼容 schema（`ingest_paper`：pdf_path 必填，doi/title 可选；`search_kb`：query 必填，k 默认 5），与 MCP 工具名一致。注意：加入此列表 = 采样 LLM 运行中可调用（llm.py:144 将其作为 `tools` 下发）。

### 7. `drsr_420/prompt_config.py`
`render_initial_analysis_prompt(self, literature_context: str | None = None)` — 非 None 时在返回串末尾追加 `\n\n### 相关文献参考 ###\n{literature_context}`（返回串仍含 `{csv_data}` 占位符，DataAnalyzer 只 replace 该占位符，不受影响）。

### 8. `drsr_420/pipeline.py`
替换 135-142 注释块为实际逻辑（约 145 行 `initial_analysis_prompt` 之前）：
```python
lit = None
try:
    rag_query = (kwargs.get('rag_query')
                 or (prompt_ctx.background_text if prompt_ctx else None)
                 or cfg.get('default_query', ''))
    if rag_query:
        from drsr_420.rag_kb import get_kb
        kb = get_kb()
        if kb.count() > 0:          # 先 count（只开 sqlite，不加载模型）
            lit = kb.get_context(rag_query, k=3)
except Exception as e:
    print(f"[RAG] 文献检索增强未启用: {e}")
initial_analysis_prompt = prompt_ctx.render_initial_analysis_prompt(literature_context=lit) if prompt_ctx else None
```
整体 try/except 守卫，RAG 失败不影响方程发现流程。

### 9. `drsr_420/find_best_eq.py`
在约 178-181 行构造解释 `content` 时，head/eq/thinking 之后、tail 之前注入文献上下文（同样 try/except + `count()>0` 守卫），查询词用 `f"磁流变 本构 {dependent} {independent}"`。

### 10. `.gitignore`
追加 `knowledge_base/`。

## 并发注意（重要）

- Chroma 官方仅支持**单进程**访问同一 path。pipeline 主进程（get_kb 单例）与 MCP 子进程（agent 调 search_kb）可能同时持有客户端 → 有锁冲突风险。
- 规避：**入库只用 CLI 单进程**（在跑 pipeline 前完成）；MCP `ingest_paper` 仅在 pipeline 未运行时使用；pipeline 内检索放在采样开始前的初始分析（无 MCP 子进程存活的安全窗口）。
- 失败模式被包含：tool_runner 把调用异常包成 `{"error":...}` 返回给 LLM，sampler 的 try/except 吞掉，不会崩 pipeline。
- 若实测频繁锁冲突，后续切 `chroma run --path knowledge_base/chroma_db` + `mode:"http"`（RagKB 预留 `HttpClient` 支持）。

## 其他

- 清理探索期临时文件：`_mods_check.py`、`_mods.txt`、`_ollama_check.py`、`_ollama.txt`。
- 首次下载 bge 模型若被墙，设 `HF_ENDPOINT=https://hf-mirror.com` 预下载，或 config 指向本地模型目录。

## 验证（端到端）

1. **装依赖**：`.venv\Scripts\python.exe -m pip install chromadb==1.5.9 sentence-transformers==5.6.1`；`python -c "import chromadb, sentence_transformers; print(chromadb.__version__, sentence_transformers.__version__)"`
2. **CLI 建库**：`python -m drsr_420.rag_build --ingest --dir pdf_downloads --limit 5` → 打印每文件 chunk 数与 `count()`；再全量跑一遍验证增量跳过；`python -m drsr_420.rag_build --query "椭球 颗粒 磁流变 本构 屈服应力" --k 5` → 检查命中是否语义相关
3. **Python API**：`python -c "from drsr_420.rag_kb import get_kb; kb=get_kb(); print(kb.count()); print(kb.get_context('磁流变 屈服应力', k=3))"`
4. **MCP**：`python -c "from drsr_420.tool_runner import mcp_call_tool; print(mcp_call_tool('search_kb', {'query':'磁流变 本构','k':3}))"`；`ingest_paper` 传 `{"pdf_path":"pdf_downloads/10.1002smll.202410011.pdf"}`
5. **pipeline 注入**：跑一次实验，确认初次分析提示词末尾出现"相关文献参考"段落、结束后 explain.txt 含文献上下文；故意损坏/改名 `knowledge_base` 验证 pipeline 仅 WARN 后继续运行
