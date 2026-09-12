"""全局 token 与耗时统计（实验级累加，多 Sampler 线程共享）。

``+=`` 非原子：无锁会让 progress.json 的 llm_tokens / llm_time_seconds 少记。
"""
import threading
from typing import Dict


# 单次实验级别的全局 token 统计（需由调用方在实验开始前手动 reset）
GLOBAL_TOKENS = {
    'prompt': 0,    # 提示词（prompt）部分 token
    'thinking': 0,  # 推理/思维链部分 token（reasoning_tokens）
    'content': 0,   # 可见输出部分 token（completion_tokens - reasoning_tokens）
    'total': 0,     # provider 返回的总 token（通常含 prompt + completion）
}
GLOBAL_TIME_SECONDS: float = 0.0

# 多 Sampler 线程并发累加（+= 非原子），无锁会让 progress.json 的
# llm_tokens / llm_time_seconds 少记；统计写入另有 try/except 兜底。
_GLOBAL_STATS_LOCK = threading.Lock()

def _accumulate_global_stats(prompt: int, thinking: int, content: int,
                             total: int, elapsed: float) -> None:
    with _GLOBAL_STATS_LOCK:
        global GLOBAL_TIME_SECONDS
        GLOBAL_TOKENS['prompt'] += prompt
        GLOBAL_TOKENS['thinking'] += thinking
        GLOBAL_TOKENS['content'] += content
        GLOBAL_TOKENS['total'] += total
        GLOBAL_TIME_SECONDS += elapsed

def reset_global_tokens():
    """重置本次实验的全局 token 统计。"""
    with _GLOBAL_STATS_LOCK:
        GLOBAL_TOKENS['prompt'] = 0
        GLOBAL_TOKENS['thinking'] = 0
        GLOBAL_TOKENS['content'] = 0
        GLOBAL_TOKENS['total'] = 0

def get_global_tokens() -> Dict[str, int]:
    """获取本次实验的全局 token 统计（thinking/content/total）。"""
    return dict(GLOBAL_TOKENS)


def reset_global_time():
    """重置本次实验的大模型总耗时统计（秒）。"""
    global GLOBAL_TIME_SECONDS
    with _GLOBAL_STATS_LOCK:
        GLOBAL_TIME_SECONDS = 0.0


def get_global_time() -> float:
    """获取本次实验的大模型总耗时（秒）。"""
    return float(GLOBAL_TIME_SECONDS)
