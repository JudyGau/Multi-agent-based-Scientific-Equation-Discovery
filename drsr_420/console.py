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
