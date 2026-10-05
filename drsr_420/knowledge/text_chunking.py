"""文本分块：按小节结构切块（无结构时回退到定长滑窗 + overlap）。

角色归属
--------
``knowledge`` 层 RAG 子系统的**切块内核**，从 :mod:`drsr_420.knowledge.rag_kb` 拆出。

为什么单独成模块
----------------
切块质量直接决定检索命中率，而它的判据全是**版面启发式**（哪一行像小节标题、
段落怎么合并、超长段怎么硬切且保留 overlap）。这些规则要能被单独测试，
不该埋在知识库类旁边。

两种策略
--------
* 有 ``##`` 风格小节标题时按小节切，超长小节再交给滑窗；
* 没有小节结构（纯文本/乱排版）时整篇走滑窗（即旧版 ``chunk_text`` 的主体行为）。
"""
from __future__ import annotations

import re


def _is_section_heading(line: str) -> bool:
    """判断一行是否为文献小节标题（启发式，宁缺勿滥：误判只会多切一刀，漏判回退整段合并）。

    识别五类常见形态：
    - Markdown 标题（``## Methods``）；
    - 数字编号（``1. Introduction`` / ``2.1 Materials``）；
    - 罗马数字编号（``II. EXPERIMENTAL``）；
    - 全大写行（PDF 提取常见的 ``INTRODUCTION``）；
    - 常见节名整行（Abstract / Conclusions / References ...，大小写不敏感）。
    """
    s = line.strip()
    if not s or len(s) > 80:                      # 标题都是短行，长行是正文/列表项
        return False
    if re.match(r"^---\s*Page \d+\s*---$", s):    # extract_pdf_text 的页码标记
        return False
    if s.rstrip(".").isdigit():                   # 纯数字行是页码/公式，不是标题
        return False
    if any(p.match(s) for p in _SECTION_HEADING_RES):
        return True
    return bool(_SECTION_NAME_RE.match(s))


#: 小节标题的形态清单（顺序无关；_is_section_heading 里统一加长度/页码护栏）
_SECTION_HEADING_RES = (
    re.compile(r"^#{1,6}\s+\S"),                              # Markdown 标题
    # 数字编号：编号后必须是字母开头的真实文字——否则 PDF 提取的公式/页码碎片
    # （如 "1 2"、"0. 8"）会被当成标题，切出一堆 3 字符的垃圾块（实测 15% 的块 <120 字符）
    re.compile(r"^\d+(?:\.\d+)*[.)]?\s+[A-Za-z][A-Za-z\s\-']{2,}"),
    # 罗马数字：同样要求后跟字母文字（"II. EXPERIMENTAL"）
    re.compile(r"^(?:I{1,3}|IV|V|VI{0,3}|IX|X|XI|XII)\.\s+[A-Za-z]"),
    re.compile(r"^[A-Z][A-Z0-9 ,\-]{4,60}$"),                 # 全大写行
)
#: 常见节名整行（可带编号前缀与冒号）
_SECTION_NAME_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?\s+)?"
    r"(?:abstract|introduction|background|motivation|methods?|materials\s+and\s+methods|"
    r"experimental(?:\s+section)?|results?(?:\s+and\s+discussion)?|discussion|"
    r"conclusions?|summary|references|acknowledg?ments?|appendix[ a-z]*)[.:]?\s*$",
    re.IGNORECASE)


def _merge_paragraphs(paragraphs: list[str], chunk_size: int, overlap: int) -> list[str]:
    """把段落贪心合并到 ≤ chunk_size；超长段落硬切（相邻片段保留 overlap）。

    即旧版 chunk_text 的主体行为，现作为"无小节结构"的回退与"超长小节"的
    二级切分器复用。
    """
    overlap = max(0, min(int(overlap), int(chunk_size) - 1))
    step = chunk_size - overlap  # 硬切步长：相邻硬切片段之间保留 overlap
    chunks, current = [], ""
    for para in paragraphs:
        if len(para) > chunk_size:
            # 先冲刷未完成的 current：旧实现把它在每次硬切前重复 append 却不清空，
            # 导致同一短块在知识库里出现多次、污染检索结果
            if current:
                chunks.append(current)
                current = ""
            while len(para) > chunk_size:  # 单段超长硬切（带重叠）
                chunks.append(para[:chunk_size])
                para = para[step:]
        if current and len(current) + len(para) + 1 > chunk_size:
            tail = current[-overlap:] if overlap > 0 else ""
            chunks.append(current)
            current = (tail + "\n" + para) if tail else para
        else:
            current = (current + "\n" + para) if current else para
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """按文献小节/段落语义分块（取代旧的固定大小分块）。

    切分策略（自顶向下）：
    1. 先按小节标题切（Markdown / 数字编号 / 罗马数字 / 全大写行 / 常见节名），
       一个小节一个语义块——检索命中的片段天然自带"它属于论文哪一节"的语境；
    2. 小节超过 chunk_size 时，在**小节内部**按空行分段贪心合并，仍保持单块
       ≤ chunk_size，且每个子块都带上小节标题前缀（子块自带上下文）；
    3. 超长段落（无空行）硬切，相邻片段保留 overlap；
    4. 全文检测不到任何小节标题时，退回旧的"空行分段 + 合并"行为。

    注意：换分块策略后必须重建知识库（``rag_build --ingest --rebuild``），
    否则旧 chunk 与新 chunk 混存、``ingest_dir`` 按文件判重会跳过重切。
    """
    text = (text or "").strip()
    if not text:
        return []
    chunk_size = max(1, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))

    lines = text.split("\n")
    sections: list[list[str]] = [[]]      # 每个小节是行列表（首行可能是标题行）
    for line in lines:
        if _is_section_heading(line):
            sections.append([line])
        else:
            sections[-1].append(line)

    if len(sections) == 1:
        # 无小节结构：退回旧行为（空行分段 + 合并 + 超长硬切）
        return _merge_paragraphs(
            [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()],
            chunk_size, overlap)

    chunks: list[str] = []
    for sec_lines in sections:
        sec = "\n".join(sec_lines).strip()
        if not sec:
            continue
        first = sec_lines[0].strip()
        heading = first if _is_section_heading(first) else ""
        if heading:
            body = "\n".join(sec_lines[1:]).strip()
            if not body:
                chunks.append(heading)     # 只有标题的空节：标题本身也值得可检索
                continue
        else:
            body = sec
        # 子块预算扣除标题前缀长度，保证"标题 + 正文"整体仍 ≤ chunk_size
        budget = max(1, chunk_size - len(heading) - 1) if heading else chunk_size
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        for piece in _merge_paragraphs(paras, budget, overlap):
            chunks.append(f"{heading}\n{piece}" if heading else piece)
    return chunks

