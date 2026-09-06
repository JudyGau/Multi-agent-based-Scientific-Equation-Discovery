# ── 共享 MCP client：让业务程序通过 stdio 调用 MCP 服务器上的工具 ──
# 用法：
#   from drsr_420.tool_runner import mcp_call_tool
#   result = mcp_call_tool("search_paper", {"query": "磁流变液", "num": 3})
#
# 说明：
#   - 底层用 mcp 官方 SDK 的 stdio client 拉起并复用单个服务器子进程（懒连接），
#     避免每次都新建 python 进程。
#   - 服务器端对应脚本 drsr_420/tools/mcp_server.py，cwd 固定为项目根目录，
#     确保 read_paper 内的 './glm_glm-5.3-flash.config' 等相对路径可用。
#   - 依赖：mcp>=1.0（见 requirements.txt）。
import json
import sys
import threading
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_ROOT = Path(__file__).resolve().parent.parent

# stdio 服务器命令（需在项目根目录下执行）
_SERVER_ARGS = ["-m", "drsr_420.tools.mcp_server"]


class MCPStdioClient:
    """通过 stdio 连接、并在后台事件循环中复用单个 MCP 服务器子进程的同步客户端。"""

    def __init__(self, command=None, server_args=None, cwd=None):
        # 默认使用与主进程相同的 Python 解释器，避免 PATH 中的 python 缺少项目依赖
        if command is None:
            command = sys.executable
        self._params = StdioServerParameters(
            command=command,
            args=list(server_args or _SERVER_ARGS),
            cwd=str(cwd or _ROOT),
            # mcp 的 stdio_client 只透传白名单环境变量，PYTHONUTF8/PYTHONIOENCODING
            # 会被丢弃，导致服务子进程退回 GBK 编码，中文/tqdm 进度条在控制台乱码。
            # 这里显式补上，保证子进程所有输出按 UTF-8 编码。
            env={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        self._loop = None
        self._session = None
        self._error = None
        self._ready = threading.Event()
        self._connect_lock = threading.Lock()

    # ── 连接管理 ─────────────────────────────
    def _ensure_connected(self):
        if self._session is not None:
            return
        with self._connect_lock:
            if self._session is not None:
                return
            import asyncio

            self._loop = asyncio.new_event_loop()
            self._ready.clear()
            t = threading.Thread(target=self._run_loop, daemon=True)
            t.start()
            self._ready.wait(60)
            if self._error:
                raise RuntimeError(f"MCP 服务器连接失败: {self._error}")
            if self._session is None:
                raise RuntimeError("MCP 服务器连接超时")

    def _run_loop(self):
        import asyncio

        asyncio.set_event_loop(self._loop)

        async def _connect():
            try:
                async with stdio_client(self._params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self._session = session
                        self._ready.set()
                        # 保持事件循环存活，直到进程结束
                        while True:
                            await asyncio.sleep(3600)
            except Exception as e:  # noqa: BLE001
                self._error = e
                self._ready.set()

        self._loop.run_until_complete(_connect())

    # ── 工具调用 ─────────────────────────────
    def call_tool(self, name, arguments=None):
        """同步调用指定工具，返回 text 内容（str）。出错时返回 {"error": ...} 的 JSON 字符串。"""
        self._ensure_connected()
        import asyncio

        future = asyncio.run_coroutine_threadsafe(
            self._do_call(name, arguments or {}), self._loop
        )
        return future.result(timeout=600)

    async def _do_call(self, name, arguments):
        try:
            resp = await self._session.call_tool(name, arguments=arguments)
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": f"调用 {name} 失败: {e}"})
        parts = [
            getattr(c, "text", None)
            for c in (resp.content or [])
            if getattr(c, "text", None) is not None
        ]
        text = "\n".join(parts)
        # mcp>=2.x 中 CallToolResult 字段为 snake_case 的 is_error（旧版 1.x 为 isError）
        if resp.is_error:
            return json.dumps({"error": text})
        return text


# ── 进程级共享的单例（懒初始化为共享 MCP client）────
_client = None
_client_lock = threading.Lock()


def get_client():
    global _client, _client_lock
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = MCPStdioClient(cwd=str(_ROOT))
    return _client


def mcp_call_tool(name, arguments):
    """进程内的统一工具调用入口：走 MCP 服务器，而不是直接调用 tools 函数。"""
    return get_client().call_tool(name, arguments)