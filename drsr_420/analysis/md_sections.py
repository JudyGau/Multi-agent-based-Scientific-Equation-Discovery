"""report.md 里**机器生成小节**的剥离与回填（幂等）。

角色归属
--------
``analysis`` 层的共享文本工具。收尾报告由"LLM 正文 + 若干机器小节"拼成；机器小节的
数字必须由系统算（不能让 LLM 转述出第二套数字），因此每个小节在装配前都要做两件事：

1. **剥离**正文里同名的小节（模型可能自己写了一份）；
2. **回填**权威版本到固定锚点（参考文献之前），并保证重复回填逐字节相同。

此前这两件事在 5 个模块里各写了一遍（``holdout`` ×2、``progress_curve`` ×1、
``explain`` ×2），13 行同构循环复制 5 份——每新增一个机器小节就要再复制一次。
本模块把它们收敛成两个函数。

一处**确实不同**的语义
----------------------
``progress_curve`` 的训练进度小节有两个额外约束（它比其他小节多出 ``###`` 子标题）：

* 用**前缀**匹配标题（该小节标题改过一次，用完整标题匹配会让老报告的旧标题小节留在
  原地，再插一节新的 → 同报告出现两节）；
* 只把 **h1/h2** 当作小节结束，``###`` 子标题属于本节内容。

这两点用 ``end_check`` 回调与"传前缀作 heading"表达，不做特例分支。
"""
from __future__ import annotations

import re

#: 连续 3 个以上换行收敛成 2 个（回填的幂等要求之一）。
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def is_h1_or_h2(line: str) -> bool:
    """小节结束判据：任何 Markdown 标题（h1-h6）。"""
    return line.startswith("#")


def only_h1_or_h2_ends_section(line: str) -> bool:
    """小节结束判据：只认 h1/h2，``###`` 及更深属本节内容。"""
    return line.startswith("#") and not line.startswith("###")


def strip_section(text: str, heading: str, *, end_check=None, rstrip: bool = True) -> str:
    """剥掉从 ``heading`` 起、到下一个同级或更高级标题为止的整节。

    Args:
        heading: 小节标题（按 ``line.strip().startswith(heading)`` 匹配，故可传前缀）。
        end_check: 判定"小节结束"的谓词，收到**原始未 strip 的行**；默认任何 ``#``
            开头的行都结束本节。
        rstrip: 是否去掉结果末尾空白。
    """
    if not text or heading not in text:
        return text
    is_end = end_check or is_h1_or_h2
    kept: list[str] = []
    skipping = False
    for line in text.splitlines():
        if line.strip().startswith(heading):
            skipping = True
            continue
        if skipping and is_end(line):
            skipping = False
        if not skipping:
            kept.append(line)
    out = "\n".join(kept)
    return out.rstrip() if rstrip else out


def upsert_section(text: str, section: str, *, heading: str, anchor: str,
                   end_check=None) -> str:
    """把 ``section`` 写进报告正文：已有同名小节则**整节替换**，否则插到 ``anchor`` 之前。

    幂等：先整节剥掉旧小节再按固定锚点插回，最后把连续空行收敛成一行——重复回填得到
    逐字节相同的结果（同一个报告不会出现两节，也不会每次多一个空行）。锚点缺失
    （正文没有该锚点小节）时追加到末尾，保证小节不丢。

    Args:
        anchor: 插入锚点（"参考文献"小节的标题）。
        end_check: 透传给 :func:`strip_section`。
    """
    if not section:
        return text
    lines = strip_section(text, heading, end_check=end_check).splitlines()
    anchor_at = next((i for i, line in enumerate(lines)
                      if line.strip().startswith(anchor)), None)
    if anchor_at is None:
        head, tail = "\n".join(lines).rstrip(), ""
    else:
        head = "\n".join(lines[:anchor_at]).rstrip()
        tail = "\n".join(lines[anchor_at:]).rstrip()
    merged = f"{head}\n\n{section}" if head else section
    if tail:
        merged += f"\n\n{tail}"
    return _BLANK_LINES_RE.sub("\n\n", merged).rstrip() + "\n"


__all__ = ["strip_section", "upsert_section", "is_h1_or_h2", "only_h1_or_h2_ends_section"]