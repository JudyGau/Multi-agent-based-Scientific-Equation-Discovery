# ── MCP 服务器：将 drsr_420/tools 下的工具暴露为 MCP 工具 ───────
# 运行方式（需在项目根目录 c:\ResearchCode\drsr-main 下）：
#   python -m drsr_420.tools.mcp_server          # stdio 传输（MCP 协议默认）
#   python -m drsr_420.tools.mcp_server --http   # 单 streamable HTTP 端点，127.0.0.1:8000/mcp
import mcp.server.fastmcp as FastMCP

from drsr_420.tools.search_paper import search_paper
from drsr_420.tools.read_paper import read_paper

mcp = FastMCP("drsr-tools")


@mcp.tool(
    description=(
        "根据关键词 搜索中/英文论文，返回文献的元数据(DOI、标题、期刊/会议名称、作者、年份、被引量等)，"
        "但无法直接获取文献内容。"
    )
)
def search_paper(query: str, num: int = 10) -> str:
    """搜索中/英文论文，返回文献元数据 JSON 字符串。"""
    return search_paper(query=query, num=num)


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
    return read_paper(payload)


if __name__ == "__main__":
    import sys

    if "--http" in sys.argv:
        # 暴露为单 Streamable HTTP 端点
        mcp.run(transport="http")
    else:
        mcp.run()