# ── MCP 服务器：将 drsr_420/tools 下的工具暴露为 MCP 工具 ───────
# 运行方式（需在项目根目录 c:\ResearchCode\drsr-main 下）：
#   python -m drsr_420.tools.mcp_server          # stdio 传输（MCP 协议默认）
#   python -m drsr_420.tools.mcp_server --http   # 单 streamable HTTP 端点，127.0.0.1:8000/mcp
import json

from mcp.server.mcpserver import MCPServer

from drsr_420.tools.search_paper import search_paper as _search_paper_impl
from drsr_420.tools.read_paper import read_paper as _read_paper_impl

mcp = MCPServer("drsr-tools")


@mcp.tool(
    description=(
        "Search academic papers (Chinese/English) by keywords and return paper metadata "
        "as a JSON string; the paper content itself is not returned."
    )
)
def search_paper(query: str, num: int = 10) -> str:
    """搜索中/英文论文，返回文献元数据 JSON 字符串。"""
    return _search_paper_impl(query=query, num=num)


@mcp.tool(
    description=(
        "Download a paper by its DOI link and return its content "
        "(optionally summarized by an LLM). "
        "title_doi is a list of (title, doi) pairs."
    )
)
def read_paper(title_doi: list[list[str]]) -> str:
    """下载论文并获取论文内容，返回结构化文本的 JSON 字符串。"""
    # 兼容原始签名 list[tuple[str, str]] | tuple[str, str]
    payload = [tuple(pair) for pair in title_doi] if isinstance(title_doi, list) else title_doi
    return _read_paper_impl(payload)


@mcp.tool(
    description=(
        "Embed an existing local PDF literature file into the RAG knowledge base. "
        "pdf_path is the PDF file path (relative to the project root is OK, "
        "e.g. 'pdf_downloads/xxx.pdf'). doi and title are optional. "
        "Returns the number of chunks ingested."
    )
)
def ingest_paper(pdf_path: str, doi: str = "", title: str = "") -> str:
    """将 PDF 文献嵌入 RAG 知识库。"""
    try:
        from drsr_420.rag_kb import get_kb
        n = get_kb().add_pdf(pdf_path, doi=doi, title=title)
        return json.dumps({"ok": True, "chunks": n, "pdf_path": pdf_path}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@mcp.tool(
    description=(
        "Search the RAG knowledge base for literature chunks semantically related to query "
        "and return the top-k hits (title, DOI, source file, text, similarity distance). "
        "The knowledge base must be populated first via ingest_paper or the CLI."
    )
)
def search_kb(query: str, k: int = 5) -> str:
    """在 RAG 知识库中检索相关文献片段。"""
    try:
        from drsr_420.rag_kb import get_kb
        hits = get_kb().search(query, k=k)
        return json.dumps(hits, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


if __name__ == "__main__":
    import sys

    if "--http" in sys.argv:
        # 暴露为单 Streamable HTTP 端点
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
