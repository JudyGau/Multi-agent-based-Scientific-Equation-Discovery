"""参考文献：检索、去重、渲染，以及从工具返回里抽取条目。

角色归属
--------
物理解释（:mod:`drsr_420.reporting.explain`）的**文献子域**。用户对 report.md 有一条
硬性要求：**必须列出参考文献**，且正文只负责用 ``[n]`` 标注，文末清单由程序从
"知识库检索命中 + 解释过程中工具检索命中"机器生成（带 DOI/来源），杜绝 LLM 编造文献；
一条都没检索到时写明"本次未获取到可引用的文献"，而不是交给模型自由发挥。

为什么单独成模块
----------------
这四件事（检索 / 去重合并 / 条目渲染 / 工具返回解析）彼此紧密、与"构造解释提示词"
和"渲染报告小节"却几乎无关。原先它们散在 1200 行的 ``explain`` 里，改一条文献渲染
规则要在一个只有 1/10 相关的文件里定位。抽出后本模块可单独测试（去重键、字段补齐、
题名回退等都有明确契约）。

对外契约
--------
``merge_references`` 的编号必须建立在**去重后的清单**上——否则提示词里的【文献 n】
与文末清单会对不上号；本模块的所有渲染入口都先经它。
"""
from __future__ import annotations

import json
import re

#: 文末参考文献小节的标题。整份 report.md 只有这一处清单（正文若自己写了，
#: 会被 :func:`strip_reference_section` 去掉后替换成机器生成的权威清单）。
REFERENCE_HEADING = "## 参考文献"

#: 一条文献都没检索到时的说明（诚实告知，而不是让模型自由发挥）。
EMPTY_REFERENCES_NOTE = ("（本次未获取到可引用的文献：RAG 知识库为空或检索失败，"
                         "且解释过程中未检索到文献。）")

#: 匹配"参考文献"小节标题行（Markdown 标题 / 加粗 / 纯文本 + 可选冒号）。
_REF_HEADING_RE = re.compile(
    r'^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?参考文献(?:\*\*)?[ \t]*[:：]?[ \t]*$', re.M)


def _ref_keys(ref: dict) -> list[str]:
    """一条文献可用的全部去重键（DOI / 题名 / 来源文件），按特异性排序。

    同一篇文献可能"知识库命中有 DOI、工具检索结果只有题名"，因此把 DOI 与题名
    两个键都登记，任一命中即视为重复，避免同一篇文献在清单里出现两次。
    """
    keys = []
    doi = (ref.get("doi") or "").strip().lower()
    if doi:
        keys.append(f"doi:{doi}")
    title = re.sub(r"\s+", " ", (ref.get("title") or "")).strip().lower()
    if title:
        keys.append(f"title:{title}")
    source = (ref.get("source_file") or "").strip().lower()
    if source:
        keys.append(f"src:{source}")
    return keys


def merge_references(*groups) -> list[dict]:
    """按顺序合并多组文献并去重（DOI / 题名 / 来源文件任一相同即同一篇）。

    同一篇文献常以多个**片段**出现（知识库按小节切块，一次检索可能命中同一 PDF 的
    好几块）：这里合并而不是丢弃——正文片段拼在一起，缺失的字段（题名/DOI/年份）
    从后续条目补齐。编号必须建立在**去重后的清单**上，否则提示词里的【文献 n】
    与文末清单会对不上号。
    """
    merged: list[dict] = []
    index_of: dict[str, int] = {}
    for group in groups:
        for ref in (group or []):
            if not isinstance(ref, dict):
                continue
            keys = _ref_keys(ref)
            if not keys:
                continue
            hit = next((index_of[k] for k in keys if k in index_of), None)
            if hit is None:
                merged.append(dict(ref))
                for k in keys:
                    index_of[k] = len(merged) - 1
                continue
            entry = merged[hit]
            text = (ref.get("text") or "").strip()
            if text and text not in (entry.get("text") or ""):
                entry["text"] = ((entry.get("text") or "").strip() + "\n" + text).strip()
            for field in ("title", "doi", "journal", "year", "authors", "source_file"):
                if not entry.get(field) and ref.get(field):
                    entry[field] = ref[field]
            for k in keys:
                index_of.setdefault(k, hit)
    return merged


def format_reference_entry(ref: dict) -> str:
    """把一条文献渲染成单行条目——**只用工具/知识库真实返回的字段**，不编造。

    题名缺失时退回 PDF 文件名（比退回 DOI 更容易人工追溯——知识库里有不少
    DOI 是从文件名猜出来的，形如 ``10.216561000/-0887.380021``）。
    """
    title = (ref.get("title") or "").strip()
    doi = (ref.get("doi") or "").strip()
    journal = (ref.get("journal") or "").strip()
    year = ref.get("year")
    source = (ref.get("source_file") or "").strip()

    # 知识库的 title 常是"从文件名回推的 DOI"（add_pdf: title = title or doi or stem），
    # 与 doi 字段一字不差时改用来源 PDF 名——对读者更有追溯价值。
    if title and doi and title.lower() == doi.lower():
        title = ""

    authors = [str(a).strip() for a in (ref.get("authors") or []) if str(a).strip()]
    head = ""
    if authors:
        head = ", ".join(authors[:3]) + (" 等" if len(authors) > 3 else "") + ". "

    body = f"{head}{title or source or doi or '（无题名）'}"
    venue = ", ".join(x for x in (journal, str(year) if year else "") if x)
    if venue:
        body += f". {venue}"
    if doi:
        body += f". DOI: {doi}"
    if source and title:
        body += f"（知识库来源: {source}）"
    return body + "."


def render_reference_section(refs: list[dict]) -> str:
    """渲染文末参考文献小节（机器生成，杜绝 LLM 自编文献）。

    内部先做一次去重：任何调用方（管线或补跑脚本）直接丢进检索原始命中，
    都不会出现"同一篇文献列三遍"。
    """
    refs = merge_references(refs)
    lines = [REFERENCE_HEADING, ""]
    if not refs:
        lines.append(EMPTY_REFERENCES_NOTE)
    else:
        lines.extend(f"[{i}] {format_reference_entry(r)}" for i, r in enumerate(refs, 1))
    return "\n".join(lines) + "\n"


def strip_reference_section(text: str) -> str:
    """删掉正文里自己写的最后一个"参考文献"小节（清单由本模块统一附加）。

    只看**最后一个**标题行：正文中间提到"参考文献"（例如"与参考文献中的
    结论一致"）不会被误删。
    """
    matches = list(_REF_HEADING_RE.finditer(text or ""))
    if not matches:
        return text
    return text[:matches[-1].start()].rstrip() + "\n"


def collect_tool_refs(tool_refs: list | None, fn_name: str, args: dict, result: str) -> None:
    """从一次 MCP 工具返回里抽取文献条目（尽力而为：解析失败就静默跳过）。

    覆盖三个工具：``search_paper``（CrossRef 元数据列表）、``search_kb``（知识库
    命中，含 doi/title/source_file）、``read_paper``（入参就是 (title, doi) 对）。
    """
    if tool_refs is None:
        return
    try:
        if fn_name == "read_paper":
            for pair in (args or {}).get("title_doi") or []:
                if isinstance(pair, (list, tuple)) and pair:
                    tool_refs.append({
                        "title": str(pair[0]),
                        "doi": str(pair[1]) if len(pair) > 1 else "",
                        "source_file": "",
                    })
            return
        payload = json.loads(result)
        if fn_name == "search_paper" and isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    tool_refs.append({
                        "title": item.get("title", ""),
                        "doi": item.get("doi", ""),
                        "journal": item.get("journal", ""),
                        "year": item.get("year"),
                        "authors": item.get("authors", []),
                    })
        elif fn_name == "search_kb" and isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    tool_refs.append({
                        "title": item.get("title", ""),
                        "doi": item.get("doi", ""),
                        "source_file": item.get("source_file", ""),
                    })
    except Exception:
        return


def retrieve_rag(query: str, k: int = 5) -> list[dict]:
    """检索知识库命中（含 title/doi/source_file），失败/库为空返回空列表。

    比 ``RagKB.get_context`` 多要一件事：结构化元数据。解释提示词要按 [n] 编号
    引用、文末要生成带 DOI 的清单，光有拼接后的文本做不到。
    """
    try:
        from drsr_420.knowledge.rag_kb import get_kb, load_config
        cfg = load_config()
        return list(get_kb().search(query, k=cfg.get("k", k) or k))
    except Exception as e:
        print(f"[RAG] 解释阶段文献检索失败（跳过）: {e}")
        return []


def explain_query(independent: str) -> str:
    """解释阶段的检索词：rag.config 的 default_query 优先（跨模式中立），否则自变量名。"""
    try:
        from drsr_420.knowledge.rag_kb import load_config
        return load_config().get("default_query") or independent
    except Exception:
        return independent


def numbered_rag_context(refs: list[dict], max_chars: int = 1500) -> str:
    """把知识库命中拼成带编号的文献上下文（编号与文末参考文献清单一一对应）。"""
    parts, total = [], 0
    for i, ref in enumerate(refs, 1):
        head = ref.get("title") or ref.get("doi") or ref.get("source_file") or "文献"
        block = f"【文献 {i}】{head}\n{ref.get('text', '')}\n"
        if total + len(block) > max_chars:
            block = block[:max(0, max_chars - total)]
        parts.append(block)
        total += len(block)
        if total >= max_chars:
            break
    return "\n".join(parts)


def format_reference_list(refs: list[dict]) -> str:
    """渲染进提示词的"可引用的文献清单"块（正文用 [n] 标注）。"""
    if not refs:
        return ""
    lines = ["### 可引用的文献清单（正文用 [n] 标注；不得引用清单之外的文献） ###\n"]
    lines.extend(f"[{i}] {format_reference_entry(r)}" for i, r in enumerate(refs, 1))
    return "\n".join(lines)


__all__ = [
    "REFERENCE_HEADING", "EMPTY_REFERENCES_NOTE",
    "merge_references", "format_reference_entry", "render_reference_section",
    "strip_reference_section", "collect_tool_refs", "retrieve_rag",
    "explain_query", "numbered_rag_context", "format_reference_list",
]