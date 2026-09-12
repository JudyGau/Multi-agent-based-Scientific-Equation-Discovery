"""线程安全的控制台输出工具。

多采样器线程并行流式输出时，无换行的增量 print(end='') 会与其他线程的日志
半行互相嵌合。LineStreamPrinter 将流式增量按行缓冲，完整行加锁原子输出并带
线程前缀，保证并行日志整行完整、互不穿插。
"""
import threading

_PRINT_LOCK = threading.Lock()


class LineStreamPrinter:
    """按行缓冲并原子输出的流式打印器：内容攒到换行才打印，每行加线程前缀。"""

    def __init__(self):
        self._buf = ""
        self._prefix = f"[{threading.current_thread().name}] "

    def write(self, delta: str) -> None:
        """写入流式增量：完整行立即原子输出，不完整行缓存在缓冲区。"""
        self._buf += delta
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)

    def write_line(self, text: str) -> None:
        """从新行开始输出完整一行（带线程前缀），避免与未完成的流式行拼接。"""
        self.newline()
        self._emit(text)

    def newline(self) -> None:
        """强制结束当前未完成的缓冲行：若缓冲非空先原子输出，保证后续内容从新行开始。"""
        if self._buf:
            self._emit(self._buf)
            self._buf = ""

    def flush(self) -> None:
        """输出剩余未换行的缓冲内容。"""
        if self._buf:
            self._emit(self._buf)
            self._buf = ""

    def _emit(self, line: str) -> None:
        with _PRINT_LOCK:
            if line:
                print(f"{self._prefix}{line}", flush=True)
            else:
                print(flush=True)


class StreamDeltaPrinter:
    """LLM 流式增量打印器：把 chat(on_delta=...) 回调的 reasoning/content 实时按行输出。

    在思考段与正文段交界处插入 [思考]/[正文] 视觉分隔，底色由 LineStreamPrinter
    保证多线程并行下整行完整、带线程前缀。

    封装原先在 tool_caller_agent / experience_summarizer_agent /
    residual_analyzer_agent / data_analyzer_agent 中逐字重复的 _on_delta 闭包。
    """

    def __init__(self):
        self._stream = LineStreamPrinter()
        self._shown = 0  # 已实时打印的字符数（reasoning 在前、content 在后拼接）
        self._think_label_printed = False
        self._content_label_printed = False

    def on_delta(self, chunk: dict) -> None:
        """流式回调：chunk 含 reasoning_content / content 字段，按到达顺序打印增量。"""
        reasoning = chunk.get('reasoning_content') or ''
        content = chunk.get('content') or ''
        text = reasoning + content
        if len(text) > self._shown:
            if self._shown < len(reasoning) and not self._think_label_printed:
                self._stream.write("[思考]\n")
                self._think_label_printed = True
            elif not self._content_label_printed:
                self._stream.write_line("[正文]")
                self._content_label_printed = True
            self._stream.write(text[self._shown:])
            self._shown = len(text)

    def flush(self) -> None:
        """输出剩余未换行的缓冲内容。"""
        self._stream.flush()


def print_block(text) -> None:
    """带线程前缀完整输出多行文本块。

    Args:
        text: 待输出的文本（多行字符串或可转字符串对象）。
    """
    prefix = f"[{threading.current_thread().name}] "
    lines = str(text).splitlines()
    with _PRINT_LOCK:
        for line in lines:
            print(f"{prefix}{line}", flush=True)
