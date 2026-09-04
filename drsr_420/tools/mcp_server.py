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
        "根据关键词 搜索中/英文论文，返回文献的元数据(DOI、标题、期刊/会议名称、作者、年份、被引量等)，"
        "但无法直接获取文献内容。"
    )
)
def search_paper(query: str, num: int = 10) -> str:
    """搜索中/英文论文，返回文献元数据 JSON 字符串。"""
    return _search_paper_impl(query=query, num=num)


@mcp.tool(
    description=(
        "根据论文的 doi 链接来下载论文并获取论文内容（并可做 LLM 总结）。"
        "入参 title_doi 为 (论文标题, doi) 键值对的列表。"
    )
)
def read_paper(title_doi: list[list[str]]) -> str:
    """下载论文并获取论文内容，返回结构化文本的 JSON 字符串。"""
    # 兼容原始签名 list[tuple[str, str]] | tuple[str, str]
    payload = [tuple(pair) for pair in title_doi] if isinstance(title_doi, list) else title_doi
    return _read_paper_impl(payload)


@mcp.tool(
    description=(
        "将本地已有的 PDF 文献文件嵌入到 RAG 文献知识库。"
        "入参 pdf_path 为 PDF 文件路径（可相对项目根目录，如 'pdf_downloads/xxx.pdf'），"
        "doi 与 title 可选。返回入库的片段数量。"
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
        "在 RAG 文献知识库中检索与 query 语义相关的文献片段，返回 top-k 条命中"
        "（标题、DOI、来源文件、正文、相似度距离）。知识库需已用 ingest_paper 或 CLI 入库。"
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
        mcp.run(transport="http")
    else:
        mcp.run()