# ── MCP 服务器：把 knowledge/tools 下的文献工具暴露为 MCP 工具 ───────
# 运行方式（需在项目根目录下）：
#   python -m drsr_420.knowledge.tools.mcp_server          # stdio 传输（MCP 协议默认）
#   python -m drsr_420.knowledge.tools.mcp_server --http   # 单 streamable HTTP 端点，127.0.0.1:8000/mcp
import json
import os
import sys
import traceback

from mcp.server.mcpserver import MCPServer

from drsr_420.knowledge.tools.search_paper import search_paper as _search_paper_impl
from drsr_420.knowledge.tools.read_paper import MCP_SERVER_ENV
from drsr_420.knowledge.tools.read_paper import read_paper as _read_paper_impl

mcp = MCPServer("drsr-tools")


def _mark_server_process() -> None:
    """给本进程打上"MCP 服务器子进程"标记（``DRSR_MCP_SERVER=1``）。

    stdio 传输下 stdout 是 JSON-RPC 通道：子进程启动时解释器按管道把 ``sys.stdout``
    设成块缓冲，SDK 随后又把 fd 1 重定向到 stderr，但 Python 层的缓冲模式不变——
    ``print`` 出来的内容会滞留缓冲区、进程被强杀收尾时整块丢失。工具实现据此标记
    把库内打印改道 stderr（见 ``read_paper._stats_to_stderr``），文献总结的
    token/耗时统计才能像其它日志一样显示出来。

    必须在 ``mcp.run()`` **之前**、且只在本进程里设置（不能在模块导入时设置：
    父进程/测试导入本模块会连带被标记，从而误改 stdout 流向）。
    """
    os.environ[MCP_SERVER_ENV] = "1"


#: 实验目录环境变量。与评估 worker 共用同一个名字（定义在
#: ``evaluation.sandbox.WORKER_LOG_ENV``）——knowledge 层按分层规则不能 import
#: evaluation，只能在需要处重述一次；它是"主进程 ↔ 子进程"的约定，改一处要改两处。
RESULTS_ROOT_ENV = "DRSR_RESULTS_ROOT"


class _StderrTee:
    """子进程侧 stderr 分流：控制台 + 实验 ``run.err``。

    每次 write 都 flush：服务器子进程随时可能被父进程强杀，留在缓冲区里的内容会直接
    丢掉——而刚刚那次下载/总结的现场恰恰是最想留下的。语义与 ``cli.main._Tee``、
    ``sandbox._WorkerStderrTee`` 一致（控制台优先、旁路写入失败不影响主流程）。
    """

    def __init__(self, console, fp):
        self._console = console
        self._fp = fp

    def write(self, data):
        for stream in (self._console, self._fp):
            try:
                stream.write(data)
            except Exception:
                pass
        try:
            self._fp.flush()
        except Exception:
            pass

    def flush(self):
        for stream in (self._console, self._fp):
            try:
                stream.flush()
            except Exception:
                pass

    def fileno(self):
        # 第三方库会拿 sys.stderr.fileno() 探测终端/给子进程用，透传控制台 fd
        # （run.err 的旁路写入不受影响）。
        return self._console.fileno()

    def isatty(self):
        try:
            return bool(self._console.isatty())
        except Exception:
            return False

    def close(self):
        """关闭旁路文件句柄（进程退出时由解释器兜底，主要供测试收尾）。"""
        try:
            self._fp.close()
        except Exception:
            pass


def attach_server_stderr() -> None:
    """把本子进程的 stderr 也旁路进实验目录的 ``run.err``。

    子进程的 fd 2 由父进程传给 ``stdio_client`` 的 ``errlog`` 决定（``tool_runner``
    用的是 ``sys.__stderr__``），只到控制台；而 ``run.err`` 是主进程在 Python 层替换
    ``sys.stderr`` 得到的，子进程根本看不到。于是文献总结的 token/耗时统计
    （``read_paper._stats_to_stderr``）与下载失败原因只闪在终端里，实验产物查无此物
    ——"这次摘要到底用了哪个模型"因此无法事后核实。

    办法与评估 worker 相同：主进程把实验目录写进环境变量（``RESULTS_ROOT_ENV``，
    由 ``tool_runner._ENV_PREFIXES`` 透传），子进程自己开一路追加写。

    失败一律静默降级：这只是诊断通道，任何环境问题都不该影响工具本身。
    """
    root = os.environ.get(RESULTS_ROOT_ENV)
    if not root or isinstance(sys.stderr, _StderrTee):
        return
    try:
        fp = open(os.path.join(root, "run.err"), "a", encoding="utf-8")
    except (OSError, ValueError):   # 路径非法时 open 抛 ValueError 而不是 OSError
        return
    sys.stderr = _StderrTee(sys.stderr, fp)


def _error_json(context: str, exc: Exception) -> str:
    """统一的工具错误返回格式（{"error": ...} JSON）。

    mcp SDK 对抛出的异常只回传固定的 "Error executing tool <name>" 文本、
    吞掉真实错误信息（tools/base.py），调用方 agent 无从判断原因；
    这里在处理器内捕获，把错误文本放进正常结果通道，并把 traceback 打到
    服务器 stderr 留痕。与 ingest_paper/search_kb 的错误风格保持一致。
    """
    traceback.print_exc(file=sys.stderr)
    return json.dumps({"error": f"{context} 失败: {exc}"}, ensure_ascii=False)


@mcp.tool(
    description=(
        "Search academic papers (Chinese/English) by keywords and return paper metadata "
        "as a JSON string; the paper content itself is not returned."
    )
)
def search_paper(query: str, num: int = 10) -> str:
    """搜索中/英文论文，返回文献元数据 JSON 字符串。"""
    try:
        return _search_paper_impl(query=query, num=num)
    except Exception as e:
        return _error_json("search_paper", e)


@mcp.tool(
    description=(
        "Download a paper by its DOI link and return its content "
        "(optionally summarized by an LLM). "
        "title_doi is a list of (title, doi) pairs; each DOI must be copied verbatim "
        "from a search_paper result. The downloaded PDF's title is verified against the "
        "requested title, and a 'title mismatch' notice is returned instead of a summary "
        "when they disagree."
    )
)
def read_paper(title_doi: list[list[str]]) -> str:
    """下载论文并获取论文内容，返回结构化文本的 JSON 字符串。"""
    try:
        # 条目合法性（二元组等）由 read_paper 实现统一校验并逐条回传错误
        return _read_paper_impl(title_doi)
    except Exception as e:
        return _error_json("read_paper", e)


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
        from drsr_420.knowledge.rag_kb import get_kb
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
        from drsr_420.knowledge.rag_kb import get_kb
        hits = get_kb().search(query, k=k)
        return json.dumps(hits, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    """启动 MCP 服务器；``--http`` 时暴露为单 Streamable HTTP 端点。

    独立成函数（而不是把逻辑写在 ``if __name__ == "__main__"`` 里）的原因：
    ``tool_runner`` 会以 ``python -m drsr_420.knowledge.tools.mcp_server`` 拉起它，
    启动逻辑必须可被显式调用；写在 ``__main__`` 里会让这条命令静默起不来
    ——MCP 客户端只能等到 120s 超时。
    """
    args = list(sys.argv[1:] if argv is None else argv)
    _mark_server_process()
    attach_server_stderr()
    if "--http" in args:
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
