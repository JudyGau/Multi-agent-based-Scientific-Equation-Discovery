"""report.md 的**机器小节**：渲染、剥离、回填与最终装配。

角色归属
--------
物理解释（:mod:`drsr_420.analysis.explain`）的**报告装配子域**。

分工的由来
----------
report.md = LLM 正文 + 若干**机器小节**。用户明确要求"数字不能由 LLM 转述"（否则
同一份报告会出现两套数字），因此每个小节都有配套的三段代码：渲染权威版本、剥掉模型
自写的同名小节、按固定锚点幂等回填。这些小节一共 6 个（发布解选择 / 样本外验证 /
LOO / 动态范围体检 / 表达式解析自检 / 训练进度）再加参考文献——原先全部堆在 explain
里，与"调 LLM 做物理解释"混在同一个文件，改一条渲染规则要在一个只有十分之一相关的
文件里定位。

本模块**不碰 LLM**（不 import 客户端、不发起请求），因此可以脱离模型单独测试、
单独用于对历史实验目录回填小节。
"""
from __future__ import annotations

import dataclasses
import math
import os
import re

from drsr_420.analysis.expr_parse import audit_parse_failures
from drsr_420.analysis.holdout import render_holdout_section, strip_holdout_section
from drsr_420.analysis.loo import render_loo_section, strip_loo_section
from drsr_420.analysis.md_sections import strip_section, upsert_section
from drsr_420.analysis.progress_curve import render_progress_section
from drsr_420.analysis.references import (REFERENCE_HEADING, render_reference_section,
                                          strip_reference_section as _strip_reference_section)
from drsr_420.core.range_check import RANGE_PROBE_REL

#: 收尾报告的产物文件名。用户明确要求把原来的 ``explain.md`` 统一改名为
#: ``report.md``（"最终报告文件命名准确无误"），所有落盘/引用都走这个常量。
REPORT_FILENAME = "report.md"

RANGE_HEADING = "## 动态范围体检"

#: 「发布解选择」小节标题。评分是 −(拟合 MSE + 体检罚分)，罚分压不彻底时病理解仍可能是
#: 最高分，所以 find_best_eq 会优先发布**无病理**的最高分样本；本小节把这次选择连同
#: 被跳过的病理解候选一起写进报告，让"为什么发布的不是最高分"可追溯。
SELECTION_HEADING = "## 发布解选择"


def render_selection_section(selection: dict | None) -> str:
    """渲染 report.md 的「发布解选择」小节（机器生成）。

    只有 `find_best_eq` 走完病理门禁才会带上 ``selection``；旧产物或单独重跑 explain
    时没有这一段。

    **没有 selection 时也渲染**——渲染标题加一行原因，而不是让整节消失（缺陷 ②）：
    实测 ``ab-fix6-control/MRFCompress-Cuboid_20260928-154613`` 与
    ``ab-iso6-no6/MRFCompress-Cuboid_20260929-091844`` 两份报告的整节不见，读者与收尾
    脚本会把「没这一节」读成「这次没做体检门禁」。真实的成因是**选解所用的那条 best
    样本的 return 表达式无法解析**（同一目录的 ``run.out`` 里同时有
    ``[WARN] return 表达式解析失败`` 或 ``[WARN] 表达式含未定义符号``），所以缺失态
    必须显式点名去查 WARN，并说明「本节缺失」不等于「没做门禁」。
    """
    if not selection or not selection.get("chosen"):
        return "\n".join([
            SELECTION_HEADING, "",
            "**本节没有选解信息**：收尾未取得「发布解 vs 病理解候选」的对照，因此"
            "**无法说明本次发布的是不是最高分、是否发生过降级**。"
            "这**不等于**没有做体检门禁——门禁跑了，但没能给出对照。"
            "最常见的原因是选解所用样本的 return 表达式无法解析："
            "请查同目录 `run.out` 里的 `[WARN] return 表达式解析失败` 或 "
            "`[WARN] 表达式含未定义符号`，那里会写明是哪一个符号、哪一行。",
        ])
    chosen = selection["chosen"]
    best = selection.get("best") or chosen
    lines = [SELECTION_HEADING, ""]

    def _fmt(entry: dict) -> str:
        penalty = entry.get("penalty")
        penalty_txt = "未知" if penalty is None else str(penalty)
        return (f"sample_order={entry.get('sample_order')}、score={entry.get('score')}、"
                f"拟合 MSE={entry.get('mse')}、体检罚分={penalty_txt}")

    lines.append(f"发布解：{_fmt(chosen)}。")
    lines.append(f"候选共 {selection.get('n_candidates', '?')} 个，"
                 f"其中体检罚分 == 0（无病理）的有 {selection.get('n_clean', '?')} 个。")
    if selection.get("degraded"):
        lines.append("")
        lines.append(f"**本次发生了降级**：分数最高的样本（{_fmt(best)}）携带**数值病理**"
                     f"（罚分 > 0），按「优先发布无病理解」的口径改为发布上面那个"
                     f"无病理样本。被跳过的最高分样本**不是被丢弃**：它的分数与罚分"
                     f"一并记录在此，供人工判断。")
        for entry in selection.get("rejected") or []:
            lines.append(f"- 因病理被跳过（分数更高）：{_fmt(entry)}")
        if selection.get("n_rejected", 0) > len(selection.get("rejected") or []):
            lines.append(f"- 另有 {selection['n_rejected'] - len(selection['rejected'])} 个"
                         f"分数更高的病理性候选未列出")
        if selection.get("n_unknown_skipped"):
            lines.append(f"- 另有 {selection['n_unknown_skipped']} 个分数更高的候选"
                         f"体检罚分**未知**（旧产物缺少拟合 MSE），既不能判为干净也不能"
                         f"判为病理，故未采用")
    elif not selection.get("n_clean"):
        lines.append("")
        lines.append("**本次没有任何无病理候选**（全部带罚分，或罚分未知）：上面的发布解仅作记录，"
                     "不应作为可用形式引用。罚分为「未知」时说明该产物缺少拟合 MSE，"
                     "既不能判为干净也不能判为病理。")
    else:
        lines.append("")
        lines.append("本次发布的样本本身就是无病理候选中的最高分，未发生降级。")
    return "\n".join(lines)


def _strip_range_section(text: str) -> str:
    """去掉正文里自带的「动态范围体检」小节（清单/数字一律由系统生成，避免两套数字）。"""
    return strip_section(text, RANGE_HEADING)


def range_check_hit(rc: dict) -> bool:
    """体检是否命中任一判据。

    ``limit`` 缺失（异常/旧格式输入）按命中处理：不能在未验证的情况下宣称通过。
    缺失 ``slope_max`` / ``coef_ratio`` 则只按现有的判据判（旧摘要没有这两项，
    不能因此改判病理）。
    """
    ratio, limit = rc.get("span_ratio"), rc.get("limit")
    slope, slimit = rc.get("slope_max"), rc.get("slope_limit")
    coef, climit = rc.get("coef_ratio"), rc.get("coef_limit")
    if limit is None:
        return True
    if ratio is not None and ratio > limit:
        return True
    if slimit is not None and slope is not None and slope > slimit:
        return True
    return climit is not None and coef is not None and coef > climit


def range_check_lines(rc: dict) -> list[str]:
    """把体检结果渲染成结论行（剪枝摘要与权威小节共用同一套措辞与数字）。

    三条判据（输出跨度 / 局部斜率 / 系数抵消）分别报告，命中时点名是哪一条；通过时
    只声明**"未检出"**并列出所检范围——体检是有限网格上的有限判据，不能写成
    "无角点钉扎/溢出类病理"（那是对未检内容的断言）。实测反例：
    MRFCompress-Cuboid_20260925-112514 的发布解核心器件是 λ12^(−40.153) 门控
    （自身动态范围 1.16e28），输出跨度只有 2.15 倍（判据一判"正常"），
    只有局部斜率能认出它；20260925-134149 的最优族则是"两个 ~7e3 系数相减出 ~300"
    （跨度/斜率都正常），只有系数抵消判据能认出。
    """
    ratio = rc.get("span_ratio")
    limit = rc.get("limit")
    slope = rc.get("slope_max")
    slimit = rc.get("slope_limit")
    coef = rc.get("coef_ratio")
    climit = rc.get("coef_limit")
    span_txt = (f"输出跨度 {ratio:.4g} 倍（阈值 {limit}）" if ratio is not None
                else "输出跨度 无有效结果")
    slope_txt = (f"局部斜率 {slope:.4g}（阈值 {slimit}）" if slope is not None
                 else "局部斜率 无有效结果")
    coef_txt = (f"系数抵消 {coef:.4g} 倍（阈值 {climit}）" if coef is not None
                else "系数抵消 未评估")
    if not range_check_hit(rc):
        return [
            f"**判定：未检出病理**——三条判据均未超阈值：{span_txt}、{slope_txt}、"
            f"{coef_txt}。",
            "",
            "该体检只覆盖**有限网格与参数上的三类数值病理**：包围盒内输出跨度异常"
            "（尖峰/深谷/溢出）、局部斜率异常（门控式局部化器件，即在角点邻域外"
            "下溢消失的项）、以及系数数量级远超数据跨度的“大系数抵消”型参数化"
            "（系数无物理读数、解会被参数边界截断）。它不证明公式在物理上正确，"
            "也不排除网格未采到的行为——物理先验、可辨识性与泛化能力只能由正文的"
            "基线与样本外对照回答。",
        ]
    hits = []
    if limit is None or (ratio is not None and ratio > limit):
        hits.append(f"输出跨度 {ratio:.4g} 倍（阈值 {limit}）" if ratio is not None
                    else "输出跨度：无有效网格点")
    if slimit is not None and slope is not None and slope > slimit:
        hits.append(f"局部斜率 {slope:.4g}（阈值 {slimit}）")
    if climit is not None and coef is not None and coef > climit:
        hits.append(f"系数抵消 {coef:.4g} 倍（阈值 {climit}）")
    gmin, gmax = rc.get("grid_min"), rc.get("grid_max")
    if gmin is not None and gmax is not None:
        hits.append(f"网格极值 [{gmin:.4g}, {gmax:.4g}]")
    return [
        f"**判定：病理性**——{'；'.join(hits)}。",
        "",
        "典型成因有两类：**角点钉扎/下溢尖峰类局部化器件**（某个项只在个别数据点"
        "——常见于自变量取值下限角点——非零、在其余区域数值下溢/上溢到无意义，"
        "用于把该点的残差单独清零）；以及**大系数抵消型参数化**（输出由几个远大于"
        "输出的系数相减而来，跨度与斜率都正常，但各系数没有物理读数、且解会在近乎"
        "平坦的方向上漂到参数边界）。这类公式在训练点上的 MSE 很好看，但点与点"
        "之间的行为是数值病理——正文如把它当作正常物理项解释，以本节为准。",
        "",
        "> 可信范围声明：该公式仅在训练数据点附近可靠；跨过器件起作用的狭窄"
        "邻域后（如角点与山脊主体之间）外推无意义。评分已按超出阈值的幅度罚分，"
        "采样阶段会因此更偏好无病理的结构。",
    ]


def render_range_section(range_check: dict | None) -> str:
    """渲染 report.md 的「动态范围体检」小节（机器生成，数字不由 LLM 转述）。

    报告最终发布公式在训练数据包围盒网格（含角点对数壳层）上的**输出跨度**与
    **局部斜率**两条判据，判定其是否携带角点钉扎/下溢尖峰类局部化器件
    （判定与评分罚分同一判据，见 ``core.range_check.dynamic_range_check``）。
    """
    lines = [RANGE_HEADING, ""]
    if not range_check:
        lines.append("本次没有可用的体检结果（体检未执行或失败）。")
        return "\n".join(lines)
    lines.append(f"评估网格：训练数据包围盒均匀网格 + 各角点向域内的对数壳层，"
                 f"共 {range_check.get('n_points', '?')} 个点；"
                 f"局部斜率按 h = {RANGE_PROBE_REL:g} × 各维 range 朝盒内偏移取差商。")
    ratio = range_check.get("span_ratio")
    if ratio is None and range_check.get("slope_max") is None:
        lines.append("体检无有效结果。")
        return "\n".join(lines)
    if ratio is not None and not math.isfinite(float(ratio)):
        gmin, gmax = range_check.get("grid_min"), range_check.get("grid_max")
        lines.append("**判定：病理性**——网格点上求值全部非有限"
                     "（方程在包围盒内处处溢出/NaN）。")
        if gmin is not None:
            lines.append(f"网格极值 [{gmin:.6g}, {gmax:.6g}]。")
        return "\n".join(lines)
    lines.extend(range_check_lines(range_check))
    return "\n".join(lines)


#: 「表达式解析自检」小节标题（机器生成）。
PARSE_AUDIT_HEADING = "## 表达式解析自检"

#: 每类失败在小节里最多列出的明细条数（其余只报数量，避免报告被几十行 WARN 淹没）。
PARSE_AUDIT_MAX_LISTED = 20


def render_parse_audit_section(audit: dict | None) -> str:
    """渲染 report.md 的「表达式解析自检」小节（机器生成，数字不由 LLM 转述）。

    收尾要把选出的样本公式解析成 SymPy 表达式；解析失败时只在同目录 ``run.out`` 留一行
    ``[WARN]``，报告里从前的完全不提示（读者看不到"这次有多少样本解释不了"，也看不出
    该修采样侧还是解析器侧）。本小节把全部已落盘样本的解析结果摊开，并**分两类**计数：
    "样本本身不完整（无 ``return``，多为截断）"与"解析器不支持的写法"。

    ``audit`` 为 ``None``（未运行自检）时返回空串，不插入小节；只要拿到结果（哪怕样本
    数为 0、或失败数为 0）都渲染，保证"这一节为什么长这样"可自解释——与
    :func:`render_selection_section` 的接法一致。
    """
    if audit is None:
        return ""
    n_total = audit.get("n_total", 0)
    lines = [PARSE_AUDIT_HEADING, ""]
    if not n_total:
        lines.append("**本次没有可自检的样本**：实验目录的 `samples/` 下没有已落盘的 "
                     "`*.json`。这**不等于**解析全部成功——只是没有样本可判。")
        return "\n".join(lines)

    n_failed = audit.get("n_failed", 0)
    rate = audit.get("failure_rate")
    rate_txt = "—" if rate is None else f"{rate:.1%}"
    lines.append(f"全部已落盘样本 **{n_total}** 个：可解析 **{audit.get('n_ok', 0)}** 个，"
                 f"解析失败 **{n_failed}** 个（失败率 {rate_txt}）。")
    if not n_failed:
        lines.append("")
        lines.append("本次**全部样本均可解析**，收尾的物理解释 / 剪枝 / 预览图不会因解析"
                     "失败被跳过。")
        return "\n".join(lines)

    lines += [
        "",
        "失败**分类**计数——这是本小节存在的意义：一个总百分比读不出该修采样侧还是"
        "解析器侧（**只有最后一类**需要在解析器侧动手）：",
        "",
        f"- **样本本身不完整（无 `return`，多为 `max_tokens` 截断）**："
        f"**{audit.get('n_truncated', 0)}** 个。属于**采样侧**问题（这些样本未被评估、"
        f"`score` 为 None），**不是**解析器缺陷。",
        f"- **样本无参数（有 `return` 但 `params` 为空，评估器从未拟合）**："
        f"**{audit.get('n_no_params', 0)}** 个。属于**样本状态**问题（`score` 为 None），"
        f"**不是**解析器缺陷——`params[k]` 无从代换，任何解析器都解不了。",
        f"- **解析器不支持的写法**：**{audit.get('n_unsupported', 0)}** 个。有 `return` "
        f"且有 `params` 但解析不出，这才是解析器需要补的写法。",
    ]

    truncated = audit.get("truncated") or []
    if truncated:
        lines += ["", f"截断样本（最多列 {PARSE_AUDIT_MAX_LISTED} 个）："]
        for rec in truncated[:PARSE_AUDIT_MAX_LISTED]:
            lines.append(f"- `{rec.get('file')}`（sample_order={rec.get('sample_order')}，"
                         f"score={rec.get('score')}）")
        if len(truncated) > PARSE_AUDIT_MAX_LISTED:
            lines.append(f"- 另有 {len(truncated) - PARSE_AUDIT_MAX_LISTED} 个未列出")

    no_params = audit.get("no_params") or []
    if no_params:
        lines += ["", f"无参数样本（最多列 {PARSE_AUDIT_MAX_LISTED} 个）："]
        for rec in no_params[:PARSE_AUDIT_MAX_LISTED]:
            lines.append(f"- `{rec.get('file')}`（sample_order={rec.get('sample_order')}，"
                         f"score={rec.get('score')}）")
        if len(no_params) > PARSE_AUDIT_MAX_LISTED:
            lines.append(f"- 另有 {len(no_params) - PARSE_AUDIT_MAX_LISTED} 个未列出")

    unsupported = audit.get("unsupported") or []
    if unsupported:
        lines += ["", f"解析器不支持的写法（最多列 {PARSE_AUDIT_MAX_LISTED} 个）："]
        for rec in unsupported[:PARSE_AUDIT_MAX_LISTED]:
            lines.append(f"- `{rec.get('file')}`（sample_order={rec.get('sample_order')}，"
                         f"score={rec.get('score')}）：{rec.get('warn') or '解析返回 None'}")
        if len(unsupported) > PARSE_AUDIT_MAX_LISTED:
            lines.append(f"- 另有 {len(unsupported) - PARSE_AUDIT_MAX_LISTED} 个未列出")

    lines += [
        "",
        "> 口径：本自检只读 `samples/*.json`，对每个样本调用收尾所用的 "
        "`expr_substitution` 判定能否解析——判定与收尾实际使用的解析器是**同一份代码**，"
        "故这里的失败率即收尾会遇到的失败率。",
    ]
    return "\n".join(lines)


def _strip_parse_audit_section(text: str) -> str:
    """去掉正文里自带的「表达式解析自检」小节（与 range/holdout 的 strip 同形）。"""
    return strip_section(text, PARSE_AUDIT_HEADING)


def upsert_parse_audit_section(text: str, section: str) -> str:
    """把解析自检小节写进报告正文：已有同名小节则**整节替换**，否则插到参考文献之前。

    幂等：先整节剥掉旧小节再按固定锚点插回，最后把连续空行收敛成一行——重复回填得到
    逐字节相同的结果（同一个报告不会出现两节，也不会每次多一个空行）。锚点缺失
    （正文没有参考文献小节）时追加到末尾，保证小节不丢。与
    :func:`drsr_420.analysis.progress_curve.upsert_progress_section` 同形
    （两者共用 :func:`drsr_420.analysis.md_sections.upsert_section`）。
    """
    return upsert_section(text, section, heading=PARSE_AUDIT_HEADING,
                          anchor=REFERENCE_HEADING)


def backfill_parse_audit(results_root: str, report_name: str = REPORT_FILENAME) -> dict:
    """对已有实验目录补出「表达式解析自检」小节（只读样本 + 改写报告，不触发 LLM）。

    报告不存在时只打印小节文本、不创建文件。返回 :func:`audit_parse_failures` 的结果。
    """
    audit = audit_parse_failures(results_root)
    path = os.path.join(results_root, report_name)
    if not os.path.exists(path):
        print(f"[INFO] 未找到 {report_name}，只输出小节文本")
        print(render_parse_audit_section(audit))
        return audit
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        updated = upsert_parse_audit_section(text, render_parse_audit_section(audit))
        if updated != text:
            with open(path, "w", encoding="utf-8") as f:
                f.write(updated)
            print(f"[INFO] 已更新 {path}")
        else:
            print(f"[INFO] {path} 无需改动（小节已是最新）")
    except Exception as e:
        print(f"[WARN] 更新 {report_name} 失败: {e}")
    return audit


@dataclasses.dataclass(frozen=True)
class ReportData:
    """report.md 各机器小节的数据（``None`` = 该小节整节不出现）。

    为什么要有这个类型
    -----------------
    这些值原先在两处**各解析一次**——提示词块一处、报告装配一处——形式都是
    ``x if x is not None else (pruning or {}).get("x")``（``holdout`` / ``holdout_ood`` /
    ``progress`` 三条回退链各写两遍）。同一件事写两遍的代价不是行数，而是**两处可能给出
    不同数字**：提示词告诉模型"样本外 MSE=…"，报告小节却渲染另一个值，而这类"两处各判
    一次"正是本仓库反复记录过的缺陷成因（见 ``core/range_check`` 的判据共用说明）。

    现在由 :meth:`from_pruning` **一次解析**，提示词与报告小节共用同一份；装配侧也顺带
    从 11 个位置参数收到 1 个对象——位置参数一多，调用点就变成"阅读时数不清哪个是哪个"
    的对照表。

    ``note`` 是**物理解释未能生成**的原因（无匹配经验 / LLM 返回空…）。旧实现在这些
    情况下直接不写文件，结果 490 个 run 只有 4 份 report.md，而失败原因只留在 run.out
    里没人看；现在改成"照写报告 + 在开头显式写明原因"，既保证产物齐全，也不掩盖故障。
    """
    refs: list = dataclasses.field(default_factory=list)
    holdout: dict | None = None
    holdout_ood: dict | None = None
    loo: dict | None = None
    fit: dict | None = None
    range_check: dict | None = None
    progress: dict | None = None
    parse_audit: dict | None = None
    selection: dict | None = None
    note: str | None = None

    @classmethod
    def from_pruning(cls, pruning: dict | None, refs=None, *,
                     parse_audit: dict | None = None,
                     note: str | None = None) -> "ReportData":
        """从 ``find_best_eq.prune_and_visualize`` 的剪枝摘要取出全部小节的机器数据。

        ``pruning`` 为 ``None``（本次没有剪枝结果）时各字段取 ``None``——机器小节里
        只有参考文献会照常出现，与旧行为一致。
        """
        P = pruning or {}
        return cls(refs=list(refs or []),
                   holdout=P.get("holdout"), holdout_ood=P.get("holdout_ood"),
                   loo=P.get("loo"), fit=P.get("fit"), range_check=P.get("range_check"),
                   progress=P.get("progress"), parse_audit=parse_audit,
                   selection=P.get("selection"), note=note)


def assemble_explain(answer: str | None, data: ReportData) -> str:
    """正文 + 权威「发布解选择」「样本外验证」「动态范围体检」「表达式解析自检」「训练进度」
    小节 + 参考文献。

    正文自带的同名小节会被替换（数字一律由系统算，避免 LLM 转述出两套数字）。
    没有训练进度记录（``best_history`` 为空）时该小节整节不出现，而不是写一句
    "本次无数据"。
    """
    refs = data.refs
    body = strip_holdout_section(answer or "")
    body = strip_loo_section(body)
    body = _strip_range_section(body)
    body = _strip_parse_audit_section(body)
    body = _strip_reference_section(body).rstrip()
    # LOO 生效时不再渲染"本次没有 held-out 数据"的空小节，避免与 LOO 小节自相矛盾
    holdout_section = ("" if (data.loo and not data.holdout and not data.holdout_ood)
                       else render_holdout_section(data.holdout, data.fit,
                                                   ood=data.holdout_ood))
    sections = [render_selection_section(data.selection),
                holdout_section,
                render_loo_section(data.loo) if data.loo else "",  # 未启用时整节不出现
                render_range_section(data.range_check),
                render_parse_audit_section(data.parse_audit),
                render_progress_section(data.progress),
                render_reference_section(refs)]
    tail = "\n\n".join(s for s in sections if s)
    parts = []
    if data.note:
        parts.append(f"> ⚠️ 物理解释未生成：{data.note}。"
                     f"本节以下的机器小节仍由系统直接计算，不依赖 LLM。")
    if body:
        parts.append(body)
    if tail:
        parts.append(tail)
    return "\n\n".join(parts)
