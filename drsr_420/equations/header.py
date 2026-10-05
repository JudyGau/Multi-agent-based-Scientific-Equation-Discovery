r"""样本文本头部的解析：从样本函数字符串里取因变量与自变量名。

为什么放在 core 层
------------------
``Dependent:`` / ``Independents:`` 是**采样侧与收尾侧共用的词汇**：采样侧用它给
骨架族去重（``agents.prompt_injection``），收尾侧用它做剪枝、曲线与物理解释
（``analysis.*``），表达式代入也要它（``analysis.expr_parse``）。分层规则不允许
``agents`` 与 ``analysis`` 互相 import，两边唯一能共享的落点就是 core
（与 :mod:`drsr_420.equations.records` 同理）。

归一化说明
----------
原先 4 处各写一份正则，其中 ``analysis.explain`` 用的是 ``Independents:\\s+``
（要求至少一个空格），其余用 ``\s*``。这里统一取 ``\s*``——它是前者的**严格超集**，
对"冒号后有空格"这一既存形态结果完全一致，只是不再拒绝无空格写法。
"""
from __future__ import annotations

import re

_DEPENDENT_RE = re.compile(r"Dependent:\s*(\w+)")
_INDEPENDENTS_RE = re.compile(r"Independents:\s*(.*)")

#: 自变量名的分隔符：英文逗号 / 中文逗号 / 任意空白。
_NAME_SPLIT_RE = re.compile(r"[,，\s]+")


def parse_dependent(text: str) -> str | None:
    """取 ``Dependent:`` 后的因变量名；没有则 ``None``。"""
    match = _DEPENDENT_RE.search(text or "")
    return match.group(1) if match else None


def parse_independents_text(text: str) -> str | None:
    """取 ``Independents:`` 后的**原始文本**（已去首尾空白）；没有则 ``None``。

    需要原样文本的调用方用它（例如解释提示词要把自变量列表原样写进正文）；
    只要名字列表时用 :func:`parse_symbols`。
    """
    match = _INDEPENDENTS_RE.search(text or "")
    return match.group(1).strip() if match else None


def split_names(text: str) -> list[str]:
    """把自变量声明文本切成名字列表（兼容逗号 / 中文逗号 / 空白分隔）。"""
    return [name.strip() for name in _NAME_SPLIT_RE.split(text or "") if name.strip()]


def parse_symbols(text: str) -> tuple[str, list[str]] | None:
    """解析 ``(因变量名, 自变量名列表)``；任一项缺失或自变量为空则 ``None``。"""
    dependent = parse_dependent(text)
    names_text = parse_independents_text(text)
    if dependent is None or names_text is None:
        return None
    names = split_names(names_text)
    if not names:
        return None
    return dependent, names


__all__ = ["parse_dependent", "parse_independents_text", "split_names", "parse_symbols"]