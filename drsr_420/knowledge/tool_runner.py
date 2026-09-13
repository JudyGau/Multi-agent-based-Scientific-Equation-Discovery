# ── 共享 MCP client：让业务程序通过 stdio 调用 MCP 服务器上的工具 ──
# 用法：
#   from drsr_420.knowledge.tool_runner import mcp_call_tool
#   result = mcp_call_tool("search_paper", {"query": "磁流变液", "num": 3})
#
# 说明：
#   - 底层用 mcp 官方 SDK 的 stdio client 拉起并复用单个服务器子进程（懒连接），
#     避免每次都新建 python 进程。
#   - 服务器端对应脚本 drsr_420/knowledge/tools/mcp_server.py，cwd 固定为项目根目录，
#     确保 read_paper 内的 './glm_glm-5.3-flash.config' 等相对路径可用。
#   - 依赖：mcp>=1.0（见 requirements.txt）。
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_ROOT = Path(__file__).resolve().parents[2]

# stdio 服务器命令（需在项目根目录下执行）。
# 用规范路径启动：子进程按 -m 解析模块名，兼容层已于阶段 6 清退。
_SERVER_ARGS = ["-m", "drsr_420.knowledge.tools.mcp_server"]

# mcp SDK 为子进程构造环境时只透传系统级白名单（PATH/TEMP/...），提供商 API key
# 环境变量会被剥离——配置里 api_key 留空、靠环境变量回退的提供商在服务子进程中
# 必然鉴权失败。这里把项目用到的密钥类变量显式并入 server.env（env 是合并而非替换）。
_ENV_PASSTHROUGH = (
    "ZHIPU_API_KEY", "SILICONFLOW_API_KEY", "DEEPSEEK_API_KEY",
    "DEEPINFRA_API_KEY", "OPENAI_API_KEY", "UNPAYWALL_EMAIL",
)


def _server_env() -> dict:
    env = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    for name in _ENV_PASSTHROUGH:
        value = os.environ.get(name)
        if value:
            env[name] = value
    return env


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
            # 编码变量 + 提供商密钥变量显式并入（见 _server_env）：mcp 的 stdio_client
            # 只透传系统级白名单，不补这些变量子进程会退回 GBK 编码且拿不到 API key。
            env=_server_env(),
        )
        self._loop = None
        self._session = None
        self._error = None
        self._connecting = False
        self._ready = threading.Event()
        self._connect_lock = threading.Lock()

    # ── 连接管理 ─────────────────────────────
    def _ensure_connected(self):
        if self._session is not None:
            return
        with self._connect_lock:
            if self._session is not None:
                return
            # 清掉上一次失败的残留错误：否则一次瞬时失败后，即使本次连接成功
            # 也会在错误检查处再次抛出，把已建好的 session/事件循环/子进程孤儿化
            self._error = None
            if not self._connecting:
                import asyncio

                self._loop = asyncio.new_event_loop()
                self._ready.clear()
                self._connecting = True
                threading.Thread(target=self._run_loop, daemon=True).start()
            # 连接线程只会有一个：超时后再次等待同一线程的结果，绝不新建第二个
            # （旧实现会在慢连接后重建循环，两个循环线程争写 _loop/_session，
            # 且 session 与 _loop 可能配对到不同的事件循环，导致 anyio 跨循环挂死）
            deadline = time.monotonic() + 120
            while not self._ready.wait(1.0):
                if time.monotonic() >= deadline:
                    raise RuntimeError("MCP 服务器连接超时")
            if self._error:
                raise RuntimeError(f"MCP 服务器连接失败: {self._error}")
            if self._session is None:
                raise RuntimeError("MCP 服务器连接未就绪")

    def _run_loop(self):
        import asyncio

        loop = self._loop  # 本地引用：不动态读 self._loop，避免与重建流程互相踩写
        asyncio.set_event_loop(loop)

        async def _connect():
            try:
                # errlog 显式绑定 sys.__stderr__：mcp 的默认参数 errlog=sys.stderr
                # 在 mcp.client.stdio 导入瞬间求值，而本模块可能被 tool_caller_agent
                # 在 drsr_420/cli/main.py 把 sys.stderr 换成无 fileno 的 _Tee 之后才首次导入——
                # 届时默认值即 _Tee，子进程 spawn 直接 AttributeError 起不来。
                async with stdio_client(self._params, errlog=sys.__stderr__) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self._session = session
                        self._ready.set()
                        # 保持事件循环存活，直到进程结束
                        while True:
                            await asyncio.sleep(3600)
            except Exception as e:  # noqa: BLE001
                self._error = e
            finally:
                self._connecting = False
                self._ready.set()

        loop.run_until_complete(_connect())

    # ── 工具调用 ─────────────────────────────
    def call_tool(self, name, arguments=None):
        """同步调用指定工具，返回 text 内容（str）。出错时返回 {"error": ...} 的 JSON 字符串。"""
        self._ensure_connected()
        import asyncio

        future = asyncio.run_coroutine_threadsafe(
            self._do_call(name, arguments or {}), self._loop
        )
        try:
            return future.result(timeout=600)
        except concurrent.futures.TimeoutError:
            # 本方法契约是"出错返回 error JSON"而非抛异常：抛出去会让上层
            # tool_caller_agent 把整条样本连同已成功的前几轮工具结果一起丢掉
            future.cancel()
            return json.dumps({"error": f"调用 {name} 超时(600s)"}, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            future.cancel()
            return json.dumps({"error": f"调用 {name} 失败: {e}"}, ensure_ascii=False)

    async def _do_call(self, name, arguments):
        try:
            resp = await self._session.call_tool(name, arguments=arguments)
        except Exception as e:  # noqa: BLE001
            # 传输层异常（子进程死亡/管道断裂）：丢弃当前 session，下一次调用重建连接，
            # 否则整个实验余下时间所有工具调用永久报错
            self._session = None
            return json.dumps({"error": f"调用 {name} 失败: {e}"}, ensure_ascii=False)
        parts = [
            getattr(c, "text", None)
            for c in (resp.content or [])
            if getattr(c, "text", None) is not None
        ]
        text = "\n".join(parts)
        # mcp>=2.x 中 CallToolResult 字段为 snake_case 的 is_error；旧版 1.x 为 isError。
        # requirements 声明 mcp>=1.0，故同时兼容两种拼写，缺失时按"非错误"处理。
        is_error = getattr(resp, "is_error", getattr(resp, "isError", False))
        if is_error:
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