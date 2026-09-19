"""物理解释：让 LLM 对最优公式做逐项力学解释，并落盘 ``explain.md``。

角色归属
--------
收尾分析（analysis）阶段的**可读性产物**：把"数学上最优"翻译成"物理上讲得通"。

协作
----
* 输入：``experiences.json`` 里该样本的 Good 条目（含模型的思考过程与含参公式）、
  ``config_snapshot.json`` 的问题背景（材料体系与自变量定义的准绳），以及
  ``find_best_eq`` 传入的**剪枝摘要**（剪枝前/后表达式、被移除项与敏感度、剪枝前后
  在训练数据上的拟合对比）；
* LLM：通过 ReAct 循环（``explain_re_act``）调用，模型可自行发起 MCP 检索工具；
* 增强：RAG 知识库注入相关文献摘要（库为空或检索失败则静默跳过）；
* 产物：``<results_root>/explain.md`` = LLM 正文（含剪枝分析）+ **由本模块附加的
  权威参考文献清单**。

两条硬性要求（用户明确指定，见下面对应的实现与测试）
----------------------------------------------------
1. **必须列出参考文献**：正文只负责用 ``[n]`` 标注，文末清单由本模块从"知识库检索
   命中 + 解释过程中工具检索命中"里机器生成（带 DOI/来源），杜绝 LLM 编造文献；
   若一条都没检索到，就写明"本次未获取到可引用的文献"，而不是交给模型自由发挥。
2. **必须解释剪枝后的表达式**，并讲清剪枝去掉了哪些项、这样剪枝为什么合理：提示词
   里因此同时给出剪枝前/后表达式、被移除项及敏感度、剪枝前后在训练数据上的 MSE
   对比（后者由 :mod:`drsr_420.analysis.prune_report` 实测），要求 LLM 逐项论证。
   这也是 ``find_best_eq`` 必须**先剪枝再解释**的原因。

失败策略：任一环节（无经验文件 / 无匹配条目 / 提示词构造失败 / LLM 初始化失败 /
保存失败）都只告警并返回，绝不抛出——收尾流程后面还有剪枝与可视化要做。
"""
from __future__ import annotations

import json
import os
import re

from drsr_420.core.console import LineStreamPrinter, print_block
from drsr_420.core import prompt_config as pc
import drsr_420.llm as llm
from drsr_420.analysis.prune_report import format_fit_summary
from drsr_420.knowledge.tool_runner import mcp_call_tool
from drsr_420.analysis.holdout import (render_holdout_section,
                                       strip_holdout_section)

#: 单个表达式/被移除项在提示词里的最大字符数（剪枝后的表达式有时很长，
#: 无节制地塞进提示词只会挤掉真正需要模型读的推导过程）。
_EXPR_CHAR_LIMIT = 2000

#: 文末参考文献小节的标题。整份 explain.md 只有这一处清单（正文若自己写了，
#: 会被 :func:`_strip_reference_section` 去掉后替换成机器生成的权威清单）。
REFERENCE_HEADING = "## 参考文献"

#: 一条文献都没检索到时的说明（诚实告知，而不是让模型自由发挥）。
EMPTY_REFERENCES_NOTE = ("（本次未获取到可引用的文献：RAG 知识库为空或检索失败，"
                         "且解释过程中未检索到文献。）")

#: 匹配"参考文献"小节标题行（Markdown 标题 / 加粗 / 纯文本 + 可选冒号）。
_REF_HEADING_RE = re.compile(
    r'^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?参考文献(?:\*\*)?[ \t]*[:：]?[ \t]*$', re.M)


def _clip(text, limit: int = _EXPR_CHAR_LIMIT) -> str:
    """超长文本截断并标注（提示词预算保护）。"""
    text = "" if text is None else str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f" …（已截断，共 {len(text)} 字符）"


def explain_re_act(client: llm.LLMClient, content: str, tool_refs: list | None = None) -> str | None:
    """ReAct 循环：流式对话，模型调工具就执行并回传，直到它给出最终答复。

    Args:
        tool_refs: 可选的列表；模型在本轮里通过 ``search_paper`` / ``search_kb`` /
            ``read_paper`` 检索到的文献会被追加进去，供调用方生成参考文献清单。
    """
    if client is None:
        return None
    try:
        messages = [
            {"role": "system", "content": pc.sampling_system_prompt},
            {"role": "user", "content": content},
        ]

        while True:
            # 流式迭代：reasoning 与 content 按到达顺序实时打印增量（网络层已是 SSE 流式）
            resp = None
            stream = LineStreamPrinter()
            shown = 0  # 已实时打印的字符数（reasoning 在前、content 在后拼接）
            think_label_printed = False
            content_label_printed = False
            for chunk in client.chat_stream(messages):
                if chunk.get('final'):
                    resp = {k: v for k, v in chunk.items() if k != 'final'}
                    break
                reasoning = chunk.get('reasoning_content') or ''
                # 局部变量名刻意不叫 content：外层 content 是提示词，历史上被这里的
                # 同名赋值覆盖过（ReAct 第二轮再读提示词就会拿到最后一段正文）。
                chunk_content = chunk.get('content') or ''
                text = reasoning + chunk_content
                if len(text) > shown:
                    if shown < len(reasoning) and not think_label_printed:
                        stream.write("[思考]\n")
                        think_label_printed = True
                    elif not content_label_printed:
                        stream.write_line("[正文]")
                        content_label_printed = True
                    stream.write(text[shown:])
                    shown = len(text)
            stream.flush()
            if resp is None:
                return None
            print("\n====================================================\n")

            tool_calls = resp.get('tool_calls', [])
            messages.append({"role": "assistant", "content": resp.get('content', ''), "tool_calls": tool_calls})

            # 如果调了 tool，执行后回传
            if tool_calls:
                print("调用了工具：", tool_calls)

                for tc in tool_calls:
                    fn_name = tc.get('function', {}).get('name', '')
                    args = json.loads(tc.get('function', {}).get('arguments', '{}'))
                    result = mcp_call_tool(fn_name, args)
                    _collect_tool_refs(tool_refs, fn_name, args, result)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get('id', ''),
                        "content": result
                    })
            # 如果未调用，则跳出循环
            else:
                return resp.get('content', '')
    except Exception as e:
        print(f"API请求发生错误: {str(e)}")
        return None


# ── 文献：检索、去重、渲染 ───────────────────────────────────

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


def _strip_reference_section(text: str) -> str:
    """删掉正文里自己写的最后一个"参考文献"小节（清单由本模块统一附加）。

    只看**最后一个**标题行：正文中间提到"参考文献"（例如"与参考文献中的
    结论一致"）不会被误删。
    """
    matches = list(_REF_HEADING_RE.finditer(text))
    if not matches:
        return text
    return text[:matches[-1].start()].rstrip() + "\n"


def _collect_tool_refs(tool_refs: list | None, fn_name: str, args: dict, result: str) -> None:
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


def _explain_query(independent: str) -> str:
    """解释阶段的检索词：rag.config 的 default_query 优先（跨模式中立），否则自变量名。"""
    try:
        from drsr_420.knowledge.rag_kb import load_config
        return load_config().get("default_query") or independent
    except Exception:
        return independent


def _numbered_rag_context(refs: list[dict], max_chars: int = 1500) -> str:
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


def _format_reference_list(refs: list[dict]) -> str:
    if not refs:
        return ""
    lines = ["### 可引用的文献清单（正文用 [n] 标注；不得引用清单之外的文献） ###\n"]
    lines.extend(f"[{i}] {format_reference_entry(r)}" for i, r in enumerate(refs, 1))
    return "\n".join(lines)


# ── 剪枝摘要块 ───────────────────────────────────────────────

def _format_pruning_block(pruning: dict) -> str:
    """渲染剪枝摘要块：判定结论 + 剪枝前/后表达式 + 被移除项（含敏感度）+ 拟合对比。"""
    lines = ["\n\n### 以下是敏感度剪枝的结果 ###\n"]
    before = pruning.get("substituted_expr")
    after = pruning.get("pruned_expr")

    # 判定结论放最前：先讲清"这次剪枝到底做没做"，否则模型会把 simplify 的通分
    # 当成剪枝结果，硬编一段"剪掉了哪些项"的说明。
    verdict = pruning.get("verdict") or {}
    if verdict.get("summary"):
        lines.append(f"剪枝判定：{verdict['summary']}")
    if verdict.get("kind") == "form_only" and pruning.get("simplify_expr"):
        lines.append("（参考，未采用）simplify 若被采用会给出的等价形式："
                     + _clip(pruning.get("simplify_expr"), 600)
                     + "——它只是通分/展开/重排，数学上与剪枝前一致，因此未被采用。")
    lines.append("")

    lines.append(f"剪枝前（参数已代入）：{_clip(before)}")
    if after is None:
        lines.append("剪枝后：本次没有得到剪枝结果（剪枝未执行或失败）。")
    elif after == before:
        if verdict.get("kind") == "degenerate":
            lines.append("剪枝后：沿用剪枝前的公式——上面那次剪枝把自变量全剪掉了"
                         "（结果退化为常数），已判为剪枝失败并回退。")
        else:
            lines.append("剪枝后：与剪枝前完全相同（没有移除任何项）。")
    else:
        lines.append(f"剪枝后：{_clip(after)}")

    rate = pruning.get("prune_rate") or 0.0
    lines.append(
        "剪枝设置与统计：threshold={thr}，sample_range={rng}，采样指标 relative、"
        "聚合 max（敏感度 ≤ 阈值即移除），节点访问 {vis}，剪枝 {cut}，剪枝率 {rate:.1%}".format(
            thr=pruning.get("threshold"), rng=pruning.get("sample_range"),
            vis=pruning.get("nodes_visited"), cut=pruning.get("nodes_pruned"), rate=rate))

    removed = pruning.get("removed") or []
    if removed:
        lines.append("被移除的项（逐项列出）：")
        for i, r in enumerate(removed, 1):
            sens = r.get("sensitivity")
            sens_txt = f"{sens:.3e}" if isinstance(sens, (int, float)) else "未知"
            lines.append(f"  ({i}) 类型={r.get('kind')}，树深度={r.get('depth')}，"
                         f"敏感度={sens_txt}，被移除项：{_clip(r.get('term'), 300)}")
    else:
        lines.append("被移除的项：无 —— 本次剪枝没有移除任何项。")

    lines.append(format_fit_summary(pruning.get("fit")))
    return "\n".join(lines)


#: 用户指定的输出结构：必须解释剪枝后的表达式、讲清剪掉了哪些项、论证剪枝合理性，
#: 并按 [n] 引用文献（清单由系统附加）。
_REQUIRED_STRUCTURE = (
    "\n\n请按下面的结构输出中文 Markdown（缺任何一节都算没完成）：\n"
    "1. 材料体系与问题定位；\n"
    "2. 剪枝前公式的逐项力学解释（每项的物理含义、量纲、在极限处的行为）；\n"
    "3. **剪枝后表达式的逐项力学解释**；若剪枝没有改变表达式（判定为\"未实际剪枝\"），"
    "就写明\"与剪枝前相同\"，并说明为何没有可剪的项；\n"
    "4. **剪枝过程去掉了哪些项**：逐项说明被移除项的物理含义、敏感度与量级，"
    "以及移除它们为什么可以接受；若一项都没移除，说明阈值与各项贡献上的原因；\n"
    "5. **剪枝合理性的论证**：结合上面给出的剪枝前后拟合数值（MSE/NMSE/最大逐点偏差）、"
    "量纲一致性与物理机制，论证剪枝没有削弱模型对数据的解释能力；若某项虽然敏感度低、"
    "但在物理上不可去（例如保证 lambda12 = lambda23 = 1 时退化为立方颗粒基线的项），"
    "必须明确指出并说明应当保留；\n"
    "6. **公式形状/物理先验与实测基线的对照**：把上方的候选骨架基线（NMSE 越小越好）"
    "与你引用的物理机理逐条对照；若某个先验（例如\"响应由整体长细比 lambda12*lambda23 支配\"）"
    "与基线排名冲突，必须显式报告这一冲突，不得把先验写成已被数据证实的事实；"
    "若给出了可辨识性告警（自变量近似共线），必须写明哪些指数在本次数据上不可单独辨识，"
    "不得把它们的相对大小解释成独立的物理发现；\n"
    "7. 结论。\n\n"
    "关于剪枝的硬性约束：只有判定为\"本次实际剪枝\"时，才把它写成剪枝结果；"
    "判定为\"未实际剪枝/仅形式变化\"时，最终公式就是剪枝前的公式，"
    "simplify 的通分/展开形式**不是**剪枝结果，不得据此编造\"被移除的项\"；"
    "判定为\"退化剪枝被拒\"时，那次剪枝已被否决，最终公式仍是剪枝前的公式，"
    "不得把被剪掉的那几项写成\"已经简化掉的项\"。\n\n"
    "引用规范：正文引用文献处用 [n] 标注（n 为下方文献清单的编号）；只允许引用该清单"
    "内的文献，不得编造文献；文末的参考文献列表由系统自动附加，你不需要自己编写。"
)


def _parse_func_header(func: str) -> tuple[str, str] | None:
    """解析样本函数头里的 (因变量, 自变量列表文本)；解析失败返回 None。"""
    dep_match = re.search(r'Dependent:\s*(\w+)', func)
    ind_match = re.search(r'Independents:\s+(.*)', func)
    if not dep_match or not ind_match:
        return None
    return dep_match.group(1), ind_match.group(1).strip()


def build_explain_content(func: str, exp: dict, background: str | None = None,
                          pruning: dict | None = None,
                          references: list | None = None,
                          holdout: dict | None = None,
                          facts: dict | None = None) -> str | None:
    """从样本函数、匹配的经验条目、剪枝摘要与文献构造解释提示词；失败返回 None。

    ``background`` 是问题的领域背景（来自 config_snapshot.json 的 ``background``
    字段，即 --background / --background_file 的最终文本）。解释 LLM 只看公式与
    经验推导时，会凭先验把自变量脑补成变形/拉伸量、把材料脑补成磁流变弹性体
    （MRE）——实测 experiments/MRFCompress-Cuboid/MRFCompress-Cuboid_20260917-134427/explain.md 即
    如此，而该问题的材料是磁流变液（MRF）、自变量是颗粒轴长比。背景必须显式
    进入提示词，并声明其优先级高于文献摘要与先验直觉。

    ``pruning`` 是 ``find_best_eq.prune_and_visualize`` 的剪枝摘要；``references``
    是已检索到的文献条目（``None`` 表示本函数自行检索，便于单独调用本函数）。
    ``facts`` 是代码实测的数据事实表（``data_facts.json``，由 evaluation 层算出）；
    给定时提示词里会带上候选骨架基线与可辨识性告警，用于对质物理先验。
    """
    thinking = exp.get("thinking_content", "")
    if not thinking:
        return None
    thinking = thinking.rsplit('\n', 1)[0]
    thinking = "以下是另一个LLM给出的公式推导（思考过程）:\n" + thinking

    return_eq = exp.get("equation", "")
    eq_match = re.search(r'return\s+(.*)', return_eq)
    if not eq_match:
        return None
    eq = "以下是另一个LLM给出的含参本构公式:\n" + eq_match.group(1)

    parsed = _parse_func_header(func)
    if parsed is None:
        return None
    dependent, independent = parsed

    head = (f"你是一名力学工程师/应用力学家，对给定公式做逐项物理机理解释，以下是一个含参本构公式和这个公式的推导逻辑，"
            f"因变量是 {dependent}，自变量是 {independent}，请你据此对这个公式从力学角度进行详细的解释。"
            "具体的领域背景请参考下方提供的文献摘要。")

    # 问题背景块：材料体系与自变量语义的准绳，优先级高于 RAG 文献与先验直觉
    bg_block = ""
    if background and background.strip():
        bg_block = ("\n\n### 以下是问题的领域背景（材料体系与自变量定义的准绳） ###\n\n"
                    + background.strip()
                    + "\n\n解释必须与上述背景保持一致：材料体系是什么（例如磁流变液还是"
                      "磁流变弹性体）、自变量的物理含义（例如颗粒轴长比还是变形拉伸量），"
                      "一律以上述背景为准；若与下方文献摘要或你的先验知识冲突，以背景为准。")

    # 剪枝摘要：解释必须覆盖剪枝后的表达式与剪枝过程（用户明确要求）
    prune_block = _format_pruning_block(pruning) if pruning else (
        "\n\n### 以下是敏感度剪枝的结果 ###\n\n"
        "本次没有得到剪枝结果（剪枝未执行或失败），请只解释剪枝前的公式，"
        "并在结论里说明剪枝分析缺失。")

    # 样本外验证块：泛化性数字（机器算的），并明确"样本内 NMSE 不是泛化误差"
    holdout_block = _format_holdout_block(holdout, (pruning or {}).get("fit"))

    # 代码实测的数据事实块：候选骨架基线与可辨识性告警，用于对质物理先验
    facts_block = _format_facts_block(facts)

    # RAG 检索增强：注入相关文献背景（失败/库为空时静默跳过）。
    # references=None 表示调用方没检索过（直接调用本函数的情形），这里代劳。
    if references is None:
        references = retrieve_rag(_explain_query(independent))
    refs = merge_references(references)
    rag_block = _numbered_rag_context(refs)
    ref_list_block = _format_reference_list(refs)

    return (head + bg_block + facts_block + prune_block + holdout_block + "\n" + eq + "\n" + thinking
            + ("\n\n### 以下是相关文献背景，供力学解释参考 ###\n\n" + rag_block if rag_block else "")
            + ("\n\n" + ref_list_block if ref_list_block else "")
            + _REQUIRED_STRUCTURE + "\n"
            + "请你根据以上内容对这个公式从力学角度进行详细的解释")


def _format_holdout_block(holdout: dict | None, fit: dict | None = None) -> str:
    """渲染样本外验证块：只给数字与口径，禁止把样本内 NMSE 当泛化误差来谈。

    数字由 :mod:`drsr_420.analysis.holdout` 算出并会**另行**写成 explain.md 的
    「样本外验证」小节；这里进提示词是为了让模型在谈泛化时只能依据这些量，
    而不是拿样本内 NMSE 说事。模型自己写的小节会被 ``strip_holdout_section`` 去掉。
    """
    if not holdout:
        return ("\n\n### 样本外（held-out）验证 ###\n\n"
                "本次没有可用的 held-out 数据，因此没有任何样本外指标。"
                "正文里出现的 MSE/NMSE 一律是样本内指标（评估器在同一批点上拟合参数并打分），"
                "不得把它们说成泛化能力或预测精度。")
    in_nmse = (fit or {}).get("nmse_before")
    lines = ["\n\n### 样本外（held-out）验证 ###\n",
             f"held-out 数据：{_clip(holdout.get('path'), 300)}"
             f"（{holdout['n_points']} 个点，未参与参数拟合、打分与样本选择）",
             f"样本外 MSE={holdout['mse']:.6g}"]
    if holdout.get("nmse") is not None:
        lines.append(f"样本外 NMSE={holdout['nmse']:.6g}")
    lines.append(f"样本外最大绝对误差={holdout['max_abs_err']:.6g}，"
                 f"最大相对误差={holdout['max_rel_err']:.2%}")
    if in_nmse:
        lines.append(f"（对照）样本内 NMSE={in_nmse:.6g}")
        if holdout.get("nmse") is not None:
            lines.append(f"样本外/样本内 NMSE 之比={holdout['nmse'] / in_nmse:.3g} 倍")
    lines.append("谈泛化时只能以上述数字为依据：样本内 NMSE 不是泛化误差，"
                 "held-out 点也很少时只能说\"通过/未通过这次样本外检查\"，"
                 "不得据此声称公式已具备预测能力。")
    return "\n".join(lines)


def _load_facts(results_root: str) -> dict | None:
    """读取同一次实验目录里的 ``data_facts.json``（缺失/损坏返回 None）。

    只做 json 读盘、不 import evaluation：analysis 层的依赖白名单是
    {core, llm, knowledge}，而事实表的计算在 evaluation 层（见
    tests/test_architecture.py 的分层约束）。
    """
    try:
        path = os.path.join(results_root, "data_facts.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] 读取 data_facts.json 失败（解释将不含实测基线块）: {e}")
        return None
    return data if isinstance(data, dict) else None


def _format_facts_block(facts: dict | None) -> str:
    """渲染代码实测的数据事实块（极值点、相关结构、候选骨架基线、可辨识性）。

    数据由 evaluation 层算出并落在同一次实验目录（``data_facts.json``），本函数只读文件
    ——analysis 层不能 import evaluation（分层约束见 tests/test_architecture.py）。

    为什么要进解释提示词：实测里解释/采样模型会把物理先验讲成结论（"压缩模式应力由
    整体长细比 lambda12*lambda23 支配"），而代码实测的乘积骨架 NMSE 是分别幂律的
    8 倍；若不把实测基线摆到它面前，它没有理由放弃先验。
    """
    if not facts:
        return ("\n\n### 以下是代码实测的数据事实 ###\n\n"
                "本次实验目录里没有 data_facts.json（旧实验或计算失败），因此没有任何"
                "实测基线数字。谈\"哪个自变量主导\"\"该形状是否符合数据\"时只能引用公式"
                "自身的拟合数值，不得声称某个先验或骨架形状已被数据支持。")
    lines = ["\n\n### 以下是代码实测的数据事实（与评估器同一拟合口径，可作为唯一数值出处） ###\n"]
    ex = (facts.get("extremes") or {}).get("max")
    if ex:
        at = ", ".join(f"{k}={v}" for k, v in ex["at"].items())
        lines.append(f"- 全局 {facts.get('dependent')} 最大：{ex['value']} 在 ({at})")
    for c in facts.get("correlations") or []:
        if c.get("b") != facts.get("dependent"):
            continue
        lines.append(f"- {c['a']} vs {c['b']}：pearson={c['pearson']}、spearman={c['spearman']}、"
                     f"对数空间 pearson={c['log_pearson']}")
    if facts.get("skeletons"):
        lines.append("- 候选骨架基线（NMSE 越小越好）：")
        for s in facts["skeletons"]:
            shown = "拟合失败" if s.get("nmse") is None else f"NMSE={s['nmse']} R2={s.get('r2')}"
            lines.append(f"  - {s['expression']} -> {shown}")
    for w in facts.get("identifiability") or []:
        lines.append(f"- 可辨识性告警：{w['message']}")
    lines.append("以上数字是唯一依据：不要把物理先验、文献结论或经验说法当作已被数据证实的事实；"
                 "若它们与上述基线冲突，必须显式报告冲突。")
    return "\n".join(lines)


def _assemble_explain(answer: str | None, refs: list[dict],
                      holdout: dict | None = None, fit: dict | None = None) -> str:
    """正文 + 权威「样本外验证」小节 + 权威参考文献小节。

    正文自带的同名小节会被替换（数字一律由系统算，避免 LLM 转述出两套数字）。
    """
    body = strip_holdout_section(answer or "")
    body = _strip_reference_section(body).rstrip()
    sections = [render_holdout_section(holdout, fit), render_reference_section(refs)]
    tail = "\n\n".join(s for s in sections if s)
    return f"{body}\n\n{tail}" if body else tail


def explain_best_sample(results_root: str, func: str, sample_order: str,
                        role_clients=None, pruning: dict | None = None,
                        holdout: dict | None = None) -> None:
    """按 sample_order 匹配 Good 经验，调用 LLM 生成物理解释并落盘 explain.md。

    任意环节失败（无经验文件 / 无匹配条目 / 提示词构造失败 / LLM 初始化失败）
    均只告警并返回，不抛出，避免影响后续剪枝流程。

    Args:
        pruning: ``find_best_eq.prune_and_visualize`` 的剪枝摘要；给定时解释会覆盖
            剪枝后的表达式、被移除项与剪枝前后拟合对比。
        holdout: 同一次收尾里的样本外验证结果（``holdout.evaluate_holdout``）；
            省略时从 ``pruning["holdout"]`` 取。给定时 explain.md 会附加机器生成的
            「样本外验证」小节（样本外指标只报告，不参与任何选择）。
        role_clients: ``llm.roles.RoleClients``；取其中的 ``explain`` 角色客户端。
            省略时按 ``config/agents.config.json`` 自行解析——**不再硬编码档案
            文件名**。旧实现在这里写死了 ``deepseek_deepseek-v4-flash.config``，
            该文件在仓库中并不存在，异常被下面的 ``except`` 吞掉后静默写出空的
            ``explain.md``（物理解释长期失效且无人发现）。
    """
    exp_path = os.path.join(results_root, "experiences.json")
    try:
        with open(exp_path, "r", encoding="utf-8") as f:
            exp_data = json.load(f)
    except Exception as e:
        print(f"[WARN] 读取经验文件失败，跳过物理解释: {e}")
        return

    # 问题背景来自 config_snapshot.json（--background / --background_file 的最终
    # 文本）：解释 LLM 必须知道材料体系与自变量定义，否则会把 MRF 解释成 MRE
    background = None
    try:
        snap_path = os.path.join(results_root, "config_snapshot.json")
        if os.path.exists(snap_path):
            with open(snap_path, "r", encoding="utf-8") as f:
                background = json.load(f).get("background")
    except Exception as e:
        print(f"[WARN] 读取 config_snapshot.json 的问题背景失败（解释将不含背景块）: {e}")

    matched = None
    for exp in exp_data.get("Good", []):
        if str(exp.get("sample_order")) == sample_order:
            matched = exp
            break
    if matched is None:
        print(f"[WARN] 未找到 sample_order={sample_order} 的 Good 经验，跳过物理解释。")
        return

    # 先自己检索一次文献：既进提示词（按 [n] 编号），又是文末参考文献清单的来源。
    # 解析失败时不传 references，让 build_explain_content 内部按同样规则兜底。
    references = None
    parsed = _parse_func_header(func)
    if parsed is not None:
        references = retrieve_rag(_explain_query(parsed[1]))

    content = build_explain_content(func, matched, background=background,
                                    pruning=pruning, references=references,
                                    holdout=holdout if holdout is not None
                                    else (pruning or {}).get("holdout"),
                                    facts=_load_facts(results_root))
    if content is None:
        print("[WARN] 构造物理解释提示词失败，跳过。")
        return

    # 初始化 LLM 客户端（explain 角色；档案与参数由 config/agents.config.json 决定，
    # 未注入 role_clients 时按注册表自行解析，因此直接调用本函数也能拿到正确档案）
    client = None
    try:
        if role_clients is not None:
            client = role_clients.get('explain')
        else:
            client = llm.build_role_client('explain')
        if client is not None:
            print(f"[INFO] LLM client initialized: provider={client._provider_name()}, "
                  f"model={client.model}, kwargs={client.kwargs}")
    except Exception as e:
        print(f"[WARN] Failed to init LLM client: {e}")
        print("[WARN] 提示：运行 `python -m drsr_420.llm.roles --check` 查看角色档案解析情况")

    tool_refs: list[dict] = []
    explain = explain_re_act(client, content, tool_refs=tool_refs)

    if not (explain or "").strip():
        # 失败时**不写文件**：把既有 explain.md 覆盖成"只剩参考文献"的残件会掩盖
        # 真实故障——旧实现写出空文件，结果物理解释长期失效却没人发现。
        print("[WARN] 物理解释为空（LLM 调用失败或返回空），保留既有 explain.md 不覆盖。")
        return

    # 正文 + 权威参考文献清单（知识库检索命中 ∪ 解释过程中工具检索命中）
    #        + 权威「样本外验证」小节（数字由系统算，不经过 LLM 转述）
    refs = merge_references(references, tool_refs)
    holdout_result = holdout if holdout is not None else (pruning or {}).get("holdout")
    final_text = _assemble_explain(explain, refs, holdout=holdout_result,
                                   fit=(pruning or {}).get("fit"))
    print_block(final_text)
    print(f"[INFO] 参考文献 {len(refs)} 条"
          + ("" if refs else "（本次未检索到可引用文献）"))

    try:
        explain_out_path = os.path.join(results_root, "explain.md")
        with open(explain_out_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"[INFO] Saved explain to: {explain_out_path}")
    except Exception as e:
        print(f"[WARN] Failed to save explain: {e}")
