"""物理解释：让 LLM 对最优公式做逐项力学解释，并落盘 ``report.md``。

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
* 产物：``<results_root>/report.md`` = LLM 正文（含剪枝分析）+ **由本模块附加的
  权威参考文献清单** + 机器生成的「发布解选择」「样本外验证」「动态范围体检」
  「表达式解析自检」「训练进度」小节（除解析自检外，其余只有拿到对应数据时才出现）。

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
from drsr_420.core.sample_header import parse_dependent, parse_independents_text
import drsr_420.llm as llm
from drsr_420.analysis.data_io import read_json_file, read_snapshot
from drsr_420.analysis.md_sections import strip_section, upsert_section
from drsr_420.analysis.prune_report import format_fit_summary
from drsr_420.knowledge.tool_runner import mcp_call_tool
from drsr_420.analysis.holdout import (LOO_MAX_TRAIN, in_sample_metrics,
                                       render_holdout_section, render_loo_section,
                                       strip_holdout_section, strip_loo_section)
from drsr_420.analysis.progress_curve import render_progress_section
from drsr_420.analysis.expr_parse import audit_parse_failures
# 体检判据的参数：小节里要写明探针偏移口径（数字必须与机器判定同一来源，
# 不能在文本里另写一份——那正是"两处各判一次"的翻版）。
from drsr_420.core.range_check import RANGE_PROBE_REL

#: 单个表达式/被移除项在提示词里的最大字符数（剪枝后的表达式有时很长，
#: 无节制地塞进提示词只会挤掉真正需要模型读的推导过程）。
_EXPR_CHAR_LIMIT = 2000


#: 文末参考文献小节的标题。整份 report.md 只有这一处清单（正文若自己写了，
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


def explain_re_act(client: llm.LLMClient, content: str, tool_refs: list | None = None,
                   max_tool_rounds: int = 6) -> str | None:
    """ReAct 循环：流式对话，模型调工具就执行并回传，直到它给出最终答复。

    Args:
        tool_refs: 可选的列表；模型在本轮里通过 ``search_paper`` / ``search_kb`` /
            ``read_paper`` 检索到的文献会被追加进去，供调用方生成参考文献清单。
        max_tool_rounds: 工具轮次上限。此前是 ``while True`` 且**没有任何上限**——模型
            只要一直发起工具调用，收尾解释就永远不返回（采样侧的 ToolCallerAgent 一直有
            4 轮兜底，这里漏了）。达到上限时返回当前响应，策略与 ToolCallerAgent 一致。
    """
    if client is None:
        return None
    try:
        messages = [
            {"role": "system", "content": pc.sampling_system_prompt},
            {"role": "user", "content": content},
        ]

        tool_rounds = 0
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
                tool_rounds += 1
                print(f"调用了工具（第 {tool_rounds}/{max_tool_rounds} 轮）：", tool_calls)

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

                # 达到上限强制收尾：与 ToolCallerAgent 同策略，避免无限检索不返回
                if tool_rounds >= max_tool_rounds:
                    print(f"[explain] 达到工具轮次上限（{max_tool_rounds}），强制返回当前响应")
                    return resp.get('content', '') or resp.get('reasoning_content', '')
            # 如果未调用，则跳出循环
            else:
                return resp.get('content', '')
    except Exception as e:
        print(f"API请求发生错误: {str(e)}")
        return None


# ── 文献：检索、去重、渲染（已拆到 :mod:`drsr_420.analysis.references`）──
# 名字按旧路径保留（对象同一），既有调用方与测试不受影响。
from drsr_420.analysis.references import (  # noqa: F401
    EMPTY_REFERENCES_NOTE,
    REFERENCE_HEADING,
    collect_tool_refs as _collect_tool_refs,
    explain_query as _explain_query,
    format_reference_entry,
    format_reference_list as _format_reference_list,
    merge_references,
    numbered_rag_context as _numbered_rag_context,
    render_reference_section,
    retrieve_rag,
    strip_reference_section as _strip_reference_section,
)


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

    # 动态范围体检结果（find_best_eq 对发布式所做，与评分器同一判据）：
    # 病理时解释 LLM 必须指认器件并划定公式的可信区域，不得把它当正常物理项解释。
    # 措辞与「动态范围体检」小节共用 _range_check_lines（同一判据只在一处成文）。
    rc = pruning.get("range_check") or {}
    if rc:
        verdict = _range_check_lines(rc)
        # 剪枝摘要走纯文本：去掉 Markdown 强调标记，只留判定那一行
        lines.append("动态范围体检：" + verdict[0].replace("**", "").replace("判定：", ""))
        if _range_check_hit(rc):
            lines.append(verdict[2])      # 成因 + "不得当作正常物理项解释"的硬约束
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
    "对照时注意标记：若某条基线带 FLAGGED，它的 NMSE 是靠角点门控/尖峰类"
    "**数值器件**取得的，只能当作该形式的上限，**不得用来论证该形式有能力，"
    "也不得反过来把它当作该先验已被数据否证的证据**；"
    "若给出了可辨识性告警（自变量近似共线），必须写明哪些指数在本次数据上不可单独辨识，"
    "不得把它们的相对大小解释成独立的物理发现；\n"
    "7. 结论。\n\n"
    "关于剪枝的硬性约束：只有判定为\"本次实际剪枝\"时，才把它写成剪枝结果；"
    "判定为\"未实际剪枝/仅形式变化\"时，最终公式就是剪枝前的公式，"
    "simplify 的通分/展开形式**不是**剪枝结果，不得据此编造\"被移除的项\"；"
    "判定为\"退化剪枝被拒\"时，那次剪枝已被否决，最终公式仍是剪枝前的公式，"
    "不得把被剪掉的那几项写成\"已经简化掉的项\"。\n"
    "最终口径（用户给定）：对外发布、并被样本外验证所评估的公式就是**剪枝后表达式**"
    "（判定为未实际剪枝/退化被拒时，剪枝后等于剪枝前），第 3、7 节与全部结论都以它为基准；"
    "第 5 节的合理性论证只用来说明这次剪枝是否站得住，不改变最终表达式，"
    "也不得把样本内指标换成剪枝前的数值来配合论证。\n\n"
    "引用规范：正文引用文献处用 [n] 标注（n 为下方文献清单的编号）；只允许引用该清单"
    "内的文献，不得编造文献；文末的参考文献列表由系统自动附加，你不需要自己编写。\n"
    "引用**范围**规范（硬性）：把某个结论归给某篇文献时，不得超过该文献注入文本实际"
    "陈述的范围——不得把\"颗粒形状会影响磁流变性质\"升级成\"形状强烈改变 MR 效应\"，"
    "也不得把\"长径比增大使效应增强\"升级成\"存在非单调关系\"，除非该文献文本里确有"
    "这句话。无法确认时写成\"[n] 指出颗粒形状会改变磁流变性质；具体趋势由本数据判定\"。"
    "实测反例（务必避免）：某报告把\"形状—MR 效应的非单调关系\"归给一篇摘要只写"
    "\"纤维悬浮液 MR 效应增强\"的文献，把\"形状强烈改变 MR 效应\"归给一篇摘要实测"
    "屈服应力 1.88 kPa（球）vs 1.86 kPa（棒）、即形状无显著影响的文献。"
)


def _parse_func_header(func: str) -> tuple[str, str] | None:
    """解析样本函数头里的 (因变量, 自变量列表文本)；解析失败返回 None。

    判据统一在 :mod:`drsr_420.core.sample_header`（采样侧与收尾侧共用一份规则），
    本函数只保留"要原始文本"这一返回形态（解释提示词要把自变量列表原样写进正文）。
    """
    dependent = parse_dependent(func)
    independents = parse_independents_text(func)
    if dependent is None or independents is None:
        return None
    return dependent, independents


def build_explain_content(func: str, exp: dict, background: str | None = None,
                          pruning: dict | None = None,
                          references: list | None = None,
                          holdout: dict | None = None,
                          holdout_ood: dict | None = None,
                          facts: dict | None = None) -> str | None:
    """从样本函数、匹配的经验条目、剪枝摘要与文献构造解释提示词；失败返回 None。

    ``background`` 是问题的领域背景（来自 config_snapshot.json 的 ``background``
    字段，即 --background / --background_file 的最终文本）。解释 LLM 只看公式与
    经验推导时，会凭先验把自变量脑补成变形/拉伸量、把材料脑补成磁流变弹性体
    （MRE）——实测 experiments/MRFCompress-Cuboid/MRFCompress-Cuboid_20260917-134427/
    的收尾报告即如此，而该问题的材料是磁流变液（MRF）、自变量是颗粒轴长比。背景必须显式
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
    holdout_block = _format_holdout_block(holdout, (pruning or {}).get("fit"),
                                          ood=holdout_ood)
    # 留一交叉验证块：训练点 < 阈值时取代 held-out（数字同样是机器算的）
    loo_block = _format_loo_block((pruning or {}).get("loo"))

    # 代码实测的数据事实块：候选骨架基线与可辨识性告警，用于对质物理先验
    facts_block = _format_facts_block(facts)

    # RAG 检索增强：注入相关文献背景（失败/库为空时静默跳过）。
    # references=None 表示调用方没检索过（直接调用本函数的情形），这里代劳。
    if references is None:
        references = retrieve_rag(_explain_query(independent))
    refs = merge_references(references)
    rag_block = _numbered_rag_context(refs)
    ref_list_block = _format_reference_list(refs)

    return (head + bg_block + facts_block + prune_block + holdout_block + loo_block + "\n" + eq + "\n" + thinking
            + ("\n\n### 以下是相关文献背景，供力学解释参考 ###\n\n" + rag_block if rag_block else "")
            + ("\n\n" + ref_list_block if ref_list_block else "")
            + _REQUIRED_STRUCTURE + "\n"
            + "请你根据以上内容对这个公式从力学角度进行详细的解释")


def _format_holdout_block(holdout: dict | None, fit: dict | None = None,
                          ood: dict | None = None) -> str:
    """渲染样本外验证块：只给数字与口径，禁止把样本内 NMSE 当泛化误差来谈。

    ``holdout`` 是**同分布（ID）**、``ood`` 是**分布外（OOD）** 的指标，两者都给时
    分别列出并标明——ID/OOD 分开报是论文的硬要求，混成一个数字看不出外推是否失效。

    数字由 :mod:`drsr_420.analysis.holdout` 算出并会**另行**写成 report.md 的
    「样本外验证」小节；这里进提示词是为了让模型在谈泛化时只能依据这些量，
    而不是拿样本内 NMSE 说事。模型自己写的小节会被 ``strip_holdout_section`` 去掉。
    """
    if not holdout and not ood:
        return ("\n\n### 样本外（held-out）验证 ###\n\n"
                "本次没有可用的 held-out 数据，因此没有任何样本外指标。"
                "正文里出现的 MSE/NMSE 一律是样本内指标（评估器在同一批点上拟合参数并打分），"
                "不得把它们说成泛化能力或预测精度。")
    lines: list[str] = []
    if holdout:
        in_nmse = in_sample_metrics(fit)["nmse"]
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
    else:
        lines = ["\n\n### 样本外（held-out）验证 ###\n",
                 "本次没有同分布（ID）的 held-out 数据。"]
    if ood:
        o_in = in_sample_metrics(fit)["nmse"]
        lines += ["", "### 分布外（OOD）held-out ###",
                  f"OOD 数据：{_clip(ood.get('path'), 300)}"
                  f"（{ood['n_points']} 个点，未参与参数拟合、打分与样本选择）",
                  f"OOD MSE={ood['mse']:.6g}"]
        if ood.get("nmse") is not None:
            lines.append(f"OOD NMSE={ood['nmse']:.6g}")
        lines.append(f"OOD 最大绝对误差={ood['max_abs_err']:.6g}，"
                     f"最大相对误差={ood['max_rel_err']:.2%}")
        if o_in and ood.get("nmse") is not None:
            lines.append(f"（对照）样本内 NMSE={o_in:.6g}，"
                         f"OOD/样本内 NMSE 之比={ood['nmse'] / o_in:.3g} 倍")
        lines.append("OOD 是分布外外推：若它明显差于同分布 held-out，必须在正文里如实"
                     "说明外推失效，不得用样本内或 ID 的数字替代 OOD。")
    return "\n".join(lines)


def _format_loo_block(loo: dict | None) -> str:
    """渲染留一交叉验证块（进解释提示词）。

    训练点太少（< holdout.LOO_MAX_TRAIN）时 held-out 不可信，系统改用留一法；这里把
    机器算出的 LOO 数字交给模型，使正文谈"样本外表现"时只能依据这些量，且必须写明
    "n 小、不作泛化声明"。模型自写的小节会被 ``strip_loo_section`` 去掉。
    """
    if not loo:
        return ""
    lines = ["\n\n### 留一交叉验证（LOO） ###\n",
             f"训练数据仅 {loo['n_points']} 个点（不足 {LOO_MAX_TRAIN}）：切不出可信的"
             f"独立 held-out，故改用留一法（每次留出 1 点、用其余点重新拟合参数后预测"
             f"该点），共 {loo['n_points']} 折、成功 {loo['n_ok']} 折。",
             f"LOO MSE={loo['mse']:.6g}"]
    if loo.get("nmse") is not None:
        lines.append(f"LOO NMSE（分母=训练集方差）={loo['nmse']:.6g}")
    if loo.get("median_abs_err") is not None:
        lines.append(f"逐点绝对误差：中位数={loo['median_abs_err']:.6g}，"
                     f"95 分位={loo['p95_abs_err']:.6g}")
    if loo.get("median_rel_err") is not None:
        lines.append(f"逐点相对误差：中位数={loo['median_rel_err']:.2%}，"
                     f"95 分位={loo['p95_rel_err']:.2%}")
    lines.append("谈样本外表现时只能以上述数字为依据；样本内 NMSE 不是泛化误差。"
                 "LOO 是**插值式**泛化（不是外推），且 **n 很小，不得据此声称公式已具备"
                 "预测能力**，也不得做 OOD 结论。")
    return "\n".join(lines)


def _load_facts(results_root: str) -> dict | None:
    """读取同一次实验目录里的 ``data_facts.json``（缺失/损坏返回 None）。

    只做 json 读盘、不 import evaluation：analysis 层的依赖白名单是
    {core, llm, knowledge}，而事实表的计算在 evaluation 层（见
    tests/test_architecture.py 的分层约束）。
    """
    facts, error = read_json_file(os.path.join(results_root, "data_facts.json"))
    if error is not None:
        print(f"[WARN] 读取 data_facts.json 失败（解释将不含实测基线块）: {error}")
        return None
    return facts


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
    for m in facts.get("monotonicity") or []:
        if m.get("monotone"):
            lines.append(f"- {facts.get('dependent')} 对 {m.get('feature')} 单调{m.get('direction')}")
            continue
        rev = m.get("first_reversal") or {}
        frm, to = rev.get("from", {}), rev.get("to", {})
        lines.append(f"- {facts.get('dependent')} 对 {m.get('feature')} **不单调**"
                     f"（{m.get('reversals')} 处反转；首个反转在 {m.get('feature')}="
                     f"{frm.get(m.get('feature'))}->{to.get(m.get('feature'))}："
                     f"{frm.get('dependent')}->{to.get('dependent')}）——"
                     "不得写成单调/饱和趋势而不提这一点")
    for c in facts.get("correlations") or []:
        if c.get("b") != facts.get("dependent"):
            continue
        lines.append(f"- {c['a']} vs {c['b']}：pearson={c['pearson']}、spearman={c['spearman']}、"
                     f"对数空间 pearson={c['log_pearson']}")
    if facts.get("skeletons"):
        lines.append("- 候选骨架基线（NMSE 是拟合本身的口径，不含选择罚分；越小越好。"
                     "带 FLAGGED 标记的行其 NMSE 是靠局部化器件（角点门控/尖峰）取得的"
                     "上限，不能用来论证该形式有能力）：")
        for s in facts["skeletons"]:
            shown = "拟合失败" if s.get("nmse") is None else f"NMSE={s['nmse']} R2={s.get('r2')}"
            note = f"  [{s['note']}]" if s.get("note") else ""
            lines.append(f"  - {s['expression']} -> {shown}{note}")
    for w in facts.get("identifiability") or []:
        lines.append(f"- 可辨识性告警：{w['message']}")
    lines.append("以上数字是唯一依据：不要把物理先验、文献结论或经验说法当作已被数据证实的事实；"
                 "若它们与上述基线冲突，必须显式报告冲突。")
    return "\n".join(lines)


# ── report.md 的机器小节（已拆到 :mod:`drsr_420.analysis.report_sections`）──
# 数字一律由系统算，不经 LLM 转述；本模块只做"构造提示词 + 调 LLM + 落盘"。
from drsr_420.analysis.report_sections import (  # noqa: F401
    PARSE_AUDIT_HEADING,
    PARSE_AUDIT_MAX_LISTED,
    RANGE_HEADING,
    REPORT_FILENAME,
    SELECTION_HEADING,
    assemble_explain as _assemble_explain,   # 兼容旧私有名（测试按它引用）
    backfill_parse_audit,
    range_check_hit as _range_check_hit,     # 剪枝摘要块与权威小节共用同一套措辞
    range_check_lines as _range_check_lines,
    render_parse_audit_section,
    render_range_section,
    render_selection_section,
    upsert_parse_audit_section,
)




def explain_best_sample(results_root: str, func: str, sample_order: str,
                        role_clients=None, pruning: dict | None = None,
                        holdout: dict | None = None,
                        holdout_ood: dict | None = None,
                        progress: dict | None = None) -> None:
    """按 sample_order 匹配 Good 经验，调用 LLM 生成物理解释并落盘 report.md。

    任意环节失败（无经验文件 / 无匹配条目 / 提示词构造失败 / LLM 初始化失败）都不抛出，
    但**仍会写出 report.md**：物理解释正文缺失时在报告开头写明原因，机器小节
    （发布解选择 / 样本外验证 / 动态范围体检 / 训练进度 / 参考文献）照常输出。
    只有"连机器小节都拿不到"时才不写文件（避免产出空报告）。

    Args:
        pruning: ``find_best_eq.prune_and_visualize`` 的剪枝摘要；给定时解释会覆盖
            剪枝后的表达式、被移除项与剪枝前后拟合对比。
        holdout: 同一次收尾里的样本外验证结果（``holdout.evaluate_holdout``）；
            省略时从 ``pruning["holdout"]`` 取。给定时 report.md 会附加机器生成的
            「样本外验证」小节（样本外指标只报告，不参与任何选择）。
        progress: 训练进度摘要（``progress_curve.plot_progress_curve``）；省略时从
            ``pruning["progress"]`` 取。给定时 report.md 会附加机器生成的
            「训练进度」小节（MSE 随 sample_order 的历史最优曲线）；为 ``None``
            时该小节整节不出现。
        role_clients: ``llm.roles.RoleClients``；取其中的 ``explain`` 角色客户端。
            省略时按 ``config/agents.config.json`` 自行解析——**不再硬编码档案
            文件名**。旧实现在这里写死了 ``deepseek_deepseek-v4-flash.config``，
            该文件在仓库中并不存在，异常被下面的 ``except`` 吞掉后静默写出空的
            ``report.md``（物理解释长期失效且无人发现）。
    """
    exp_data = None
    note = None
    exp_path = os.path.join(results_root, "experiences.json")
    try:
        with open(exp_path, "r", encoding="utf-8") as f:
            exp_data = json.load(f)
    except Exception as e:
        print(f"[WARN] 读取经验文件失败，本次报告将不含物理解释: {e}")
        note = f"读取 experiences.json 失败（{e}）"

    # 问题背景来自 config_snapshot.json（--background / --background_file 的最终
    # 文本）：解释 LLM 必须知道材料体系与自变量定义，否则会把 MRF 解释成 MRE
    background = None
    snapshot, snap_error = read_snapshot(results_root)
    if snap_error is not None:
        print(f"[WARN] 读取 config_snapshot.json 的问题背景失败（解释将不含背景块）: {snap_error}")
    else:
        background = (snapshot or {}).get("background")

    matched = None
    for exp in (exp_data or {}).get("Good", []):
        if str(exp.get("sample_order")) == sample_order:
            matched = exp
            break
    if matched is None and note is None:
        note = (f"未找到 sample_order={sample_order} 的 Good 经验（该样本当时被分类为 "
                f"Bad/None，或该条经验已不在 experiences.json 中）")
        print(f"[WARN] {note}，物理解释正文未生成。")

    # 先自己检索一次文献：既进提示词（按 [n] 编号），又是文末参考文献清单的来源。
    # 解析失败时不传 references，让 build_explain_content 内部按同样规则兜底。
    references = None
    parsed = _parse_func_header(func)
    if parsed is not None:
        references = retrieve_rag(_explain_query(parsed[1]))

    tool_refs: list[dict] = []
    explain = ""
    if matched is not None:
        content = build_explain_content(func, matched, background=background,
                                        pruning=pruning, references=references,
                                        holdout=holdout if holdout is not None
                                        else (pruning or {}).get("holdout"),
                                        holdout_ood=holdout_ood if holdout_ood is not None
                                        else (pruning or {}).get("holdout_ood"),
                                        facts=_load_facts(results_root))
        if content is None:
            note = "构造物理解释提示词失败"
            print("[WARN] 构造物理解释提示词失败，物理解释正文未生成。")
        else:
            # 初始化 LLM 客户端（explain 角色；档案与参数由 config/agents.config.json
            # 决定，未注入 role_clients 时按注册表自行解析，因此直接调用本函数也能拿到
            # 正确档案）
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

            explain = explain_re_act(client, content, tool_refs=tool_refs)
            if not (explain or "").strip():
                note = "物理解释为空（LLM 调用失败或返回空）"
                print(f"[WARN] {note}。")

    # 正文 + 权威参考文献清单（知识库检索命中 ∪ 解释过程中工具检索命中）
    #        + 权威「发布解选择 / 样本外验证 / 动态范围体检 / 训练进度」小节（数字由
    #        系统直接算，不经过 LLM 转述）
    #
    # 落盘策略（2026-09-26 修正）：**无论物理解释是否生成都要写 report.md**。这些机器
    # 小节本身就是可引用的产物，而旧实现"失败就不写"，导致 490 个 run 只剩 4 份报告、
    # 失败原因只留在 run.out 里没人看；现在把原因写在报告开头，照写不会掩盖故障。
    refs = merge_references(references, tool_refs)
    holdout_result = holdout if holdout is not None else (pruning or {}).get("holdout")
    holdout_ood_result = (holdout_ood if holdout_ood is not None
                          else (pruning or {}).get("holdout_ood"))
    progress_result = progress if progress is not None else (pruning or {}).get("progress")
    # 收尾自检：这次全部已落盘样本的解析失败率（分「截断样本」/「解析器不支持的写法」）。
    # 解析失败从前只在 run.out 里留 WARN、报告不提示，读者看不到总体失败率。
    parse_audit_result = audit_parse_failures(results_root)
    final_text = _assemble_explain(explain, refs, holdout=holdout_result,
                                   fit=(pruning or {}).get("fit"),
                                   holdout_ood=holdout_ood_result,
                                   loo=(pruning or {}).get("loo"),
                                   range_check=(pruning or {}).get("range_check"),
                                   progress=progress_result,
                                   parse_audit=parse_audit_result,
                                   selection=(pruning or {}).get("selection"),
                                   note=note)
    if not final_text.strip():
        print(f"[WARN] 报告无任何可用内容（物理解释与机器小节都为空），"
              f"不写 {REPORT_FILENAME}。")
        return
    explain_out_path = os.path.join(results_root, REPORT_FILENAME)
    if note and os.path.exists(explain_out_path):
        # 物理解释缺失 + 已有报告：保留既有那份，不要把上一版好报告替换成"降级版"
        # （旧实现的口径，仍然成立）。反之若目录里根本没有 report.md，下面照写——
        # 那正是"490 个 run 只有 4 份报告"的场景，产物必须留下。
        print(f"[WARN] 物理解释未生成（{note}），保留既有 {REPORT_FILENAME} 不覆盖。")
        return
    print_block(final_text)
    print(f"[INFO] 参考文献 {len(refs)} 条"
          + ("" if refs else "（本次未检索到可引用文献）"))

    try:
        with open(explain_out_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"[INFO] Saved report to: {explain_out_path}")
    except Exception as e:
        print(f"[WARN] Failed to save explain: {e}")
