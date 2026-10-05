"""训练进度曲线：把"逐样本轨迹"与"历史最优刷新点"各自画成随 sample_order 变化的
三条曲线（拟合 MSE / 动态范围体检罚分 / 评分），并渲染 report.md 的机器生成小节。

用途
----
收尾时回答"这一次搜索是怎么收敛的"：最优解在第几个样本出现、中途刷新过几次、
末段的改善是台阶还是抖动；以及——只看 MSE 会漏掉的那件事——**罚分是不是在同步上涨**
（实测"低 MSE 高罚分 ↔ 高 MSE 零罚分"两模态会让 MSE 曲线看起来在改善而评分没动）。
两套曲线各三张图：

* **逐样本**（``samples/*.json``，每个已落盘样本一个点、折线连接）：
  ``mse_per_sample_vs_sample_order.png`` / ``penalty_per_sample_vs_sample_order.png`` /
  ``score_per_sample_vs_sample_order.png``；
* **刷新点**（``best_history/best_sample_<sample_order>.json``，一次全局最优刷新一个点、
  阶梯线）::

      mse_vs_sample_order.png / penalty_vs_sample_order.png / score_vs_sample_order.png

小节由系统生成（数字不经 LLM 转述），排在「动态范围体检」之后、「参考文献」之前。

口径
----
* **只读实验目录内的机器产物**：``samples/*.json``（逐样本）与
  ``best_history/best_sample_<sample_order>.json``（刷新点）。``mse`` 是**拟合本身**的
  均方误差，体检罚分记在 ``penalty`` 里（评分 = −(mse+penalty)）；**旧目录没有
  ``penalty`` 字段**，其 ``mse`` 内含罚分，本模块会据此改写纵轴标注、跳过罚分曲线
  并在小节里告警。逐样本曲线还可能只含 top-K（``persist_all_samples`` 之前）——
  小节里必须写明，否则会被读成"只评估了这么多次"。
* 刷新点曲线的三条线共用同一批刷新点（刷新 = 评分改善），横轴逐点对齐；每个点上恒有
  ``|score| = mse + penalty``，可逐点自检。
* **不解析 run.out 的逐样本分数**：``MRFCompress-Cuboid.bat`` 等启动入口并不重定向
  stdout（实测其调用是裸的 ``python -m drsr_420.cli.main …``），``run.out`` 不是每条
  启动路径都存在的产物，把它当数据源会让报告在别的启动方式下缺图。
* 曲线**全程是样本内指标**（评估器在同一批训练点上拟合参数并打分），低 MSE 不代表
  泛化——小节末尾写明这一点，泛化对照归「样本外验证」小节。

用法
----
::

    python -m drsr_420.analysis.progress_curve <results_root>

回填：对已有实验目录重跑本模块即可补出六张图与小节（不触发任何 LLM 调用）；
小节按 ``## 训练进度`` 前缀整节替换，故对只含旧 MSE 标题的历史报告同样幂等。
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

from drsr_420.analysis.md_sections import (
    only_h1_or_h2_ends_section,
    strip_section,
    upsert_section,
)
from drsr_420.core.sample_records import load_sample_records

#: 三个量各自的进度图文件名（报告小节按相对路径引用）。MSE 沿用旧名——历史报告与
#: 文档都引用过它，改名只会制造孤儿文件。
PROGRESS_PNG_NAME = "mse_vs_sample_order.png"
PENALTY_PNG_NAME = "penalty_vs_sample_order.png"
SCORE_PNG_NAME = "score_vs_sample_order.png"

#: 逐样本（每个已落盘样本一个点、折线连接）三张图。与上面三张的区别是**数据源**：
#: 上面只取全局最优刷新点（``best_history``），这里取 ``samples/*.json`` 的全部样本。
#: 文件名刻意不与上面互为子串，便于在任何文本里唯一定位。
PER_SAMPLE_MSE_PNG_NAME = "mse_per_sample_vs_sample_order.png"
PER_SAMPLE_PENALTY_PNG_NAME = "penalty_per_sample_vs_sample_order.png"
PER_SAMPLE_SCORE_PNG_NAME = "score_per_sample_vs_sample_order.png"

#: report.md 里训练进度小节的标题（机器生成）。现在覆盖三个量，故标题改为此；
#: **剥离旧小节用前缀** :data:`_SECTION_PREFIX`，这样老报告里只含 MSE 的旧标题
#: （``## 训练进度：MSE …``）也能被整节替换掉，不会回填出两节。
PROGRESS_HEADING = "## 训练进度：MSE / 罚分 / score 随 sample_order 的变化"
_SECTION_PREFIX = "## 训练进度"

#: 评分曲线的纵轴用 symlog：评分 ≤ 0 且跨数量级（实测 −102.6 → −0.27），
#: 线性轴会把后段的改善压成一条线、对数轴又表示不了负数。阈值取 1e-3，使本问题的
#: 全部评分都落在对数段（symlog 的线性段只服务于 |score| < 1e-3 的"近平完美"解）。
SCORE_SYMLOG_LINTHRESH = 1e-3

#: 插入位置锚点：机器小节排在参考文献之前。
#: 这里刻意**不** import ``explain.REFERENCE_HEADING``——explain 会 import 本模块来
#: 装配小节，反向在模块级 import 会成环；两者的取值一致性由
#: ``tests/test_progress_curve.py`` 断言守住（单一来源靠测试而非重复 import）。
_REFERENCE_ANCHOR = "## 参考文献"

_ORDER_RE = re.compile(r"best_sample_(\d+)\.json$")


def load_best_history(results_root: str) -> list[dict]:
    """读 ``best_history/best_sample_*.json``，按 ``sample_order`` 升序返回刷新点。

    单个文件读坏/字段不全/``mse`` 非有限时只告警跳过（不因为一个坏文件丢掉整条曲线）。
    返回项：``{"sample_order", "iteration", "mse", "nmse", "penalty", "score"}``
    （``iteration`` / ``nmse`` / ``penalty`` 可能为 ``None``，历史目录缺字段时按缺失处理）。

    ``score`` 优先取记录里的字段；缺失或非有限时按同一口径推出来（有 ``penalty`` →
    ``-(mse+penalty)``；旧目录无 ``penalty`` → ``-mse``，那里的 ``mse`` 本就内含罚分）。
    这样三条曲线在**每个刷新点**上都满足 ``|score| = mse + penalty``，可自检。
    """
    pattern = os.path.join(results_root, "best_history", "best_sample_*.json")
    rows: list[dict] = []
    for path in glob.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception as e:                          # 单文件坏不影响整条曲线
            print(f"[WARN] 读取历史最优失败，跳过 {os.path.basename(path)}: {e}")
            continue
        order = rec.get("sample_order")
        mse = rec.get("mse")
        if order is None or mse is None:
            print(f"[WARN] 历史最优缺 sample_order/mse，跳过 {os.path.basename(path)}")
            continue
        try:
            mse = float(mse)
            order = int(order)
        except (TypeError, ValueError):
            print(f"[WARN] 历史最优字段非数值，跳过 {os.path.basename(path)}")
            continue
        if not np.isfinite(mse):
            print(f"[WARN] 历史最优 mse 非有限，跳过 {os.path.basename(path)}")
            continue
        penalty = _finite_or_none(rec.get("penalty"))
        score = _finite_or_none(rec.get("score"))
        if score is None:
            score = -(mse + penalty) if penalty is not None else -mse
        rows.append({"sample_order": order, "iteration": rec.get("iteration"),
                     "mse": mse, "nmse": rec.get("nmse"),
                     "penalty": penalty, "score": score,
                     # 口径拆分（20260926 之后）才写 penalty 字段；旧目录的 mse 里
                     # 混着动态范围体检罚分（= −score），不能当拟合质量读。
                     "mse_includes_penalty": "penalty" not in rec})
    # 兜底：文件名里的 order 与内容不一致时以内容为准（内容才是评估器写的）
    rows.sort(key=lambda r: r["sample_order"])
    return rows


def _finite_or_none(value):
    """有限实数（不是 bool）→ float；缺失 / 非数值 / NaN / inf → ``None``。"""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def load_sample_points(results_root: str) -> list[dict]:
    """读 ``samples/*.json`` 的**每个已落盘样本**，按 ``sample_order`` 升序返回。

    与 :func:`load_best_history`（只有全局最优刷新点）相对：这里是**原始轨迹**——
    每个 ``sample_order`` 一个点，于是"低 MSE 高罚分 ↔ 高 MSE 零罚分"在两模态之间的
    来回摆动直接可见；而刷新点曲线按定义只保留"评分改善"的那几个点，看不见摆动。

    返回项与 :func:`load_best_history` 同形（``mse`` / ``nmse`` / ``penalty`` / ``score``），
    以便复用同一套画法与刻度规则。

    只包含**已落盘**的样本：``persist_all_samples`` 之前的实验只留 top-K，此时点数
    远小于实际评估数——渲染方必须把这一点写进小节（否则会被读成"只评估了这么多次"）。
    """
    rows: list[dict] = []
    for record in load_sample_records(results_root or "."):
        order = record.get("sample_order")
        mse = _finite_or_none(record.get("mse"))
        if order is None or mse is None:
            continue
        rows.append({"sample_order": int(order), "iteration": None, "mse": mse,
                     "nmse": record.get("nmse"),
                     "penalty": _finite_or_none(record.get("penalty")),
                     # load_sample_records 只收带数字 score 的记录，故这里必有值
                     "score": float(record["score"])})
    rows.sort(key=lambda r: r["sample_order"])
    return rows


def _plot_quantity(plt, rows: list[dict], results_root: str, *, key: str, png_name: str,
                   label: str, ylabel: str, title: str, color: str,
                   symlog: bool = False, step: bool = True) -> dict | None:
    """把单个量画成一条曲线，返回 ``{path, n_points, scale}``。

    ``step=True``（默认，刷新点曲线）：阶梯线是"当时的全局最优、保持到下一次刷新"，
    菱形标出刷新点。``step=False``（逐样本曲线）：把每个 ``sample_order`` 的点用
    **折线**连起来——它表示"每个样本各自的值"，没有"保持"的语义，故不能画成阶梯。

    没有可画的数据（该字段全缺）时返回 ``None`` **且不落盘**——报告侧据此省略这一条
    曲线，而不是贴一张空图。
    """
    values = np.asarray([row[key] for row in rows], dtype=float)
    xs = np.asarray([row["sample_order"] for row in rows], dtype=float)
    # 罚分可以恰好为 0（干净解），对数轴表示不了 → 退回线性（与 MSE 的既有规则一致）
    log_scale = bool(not symlog and np.all(values > 0))

    fig, ax = plt.subplots(figsize=(7, 5))
    if step:
        ax.step(xs, values, where="post", color=color, lw=2,
                label=f"{label} (held to next refresh)")
        ax.plot(xs, values, linestyle="none", marker="D", ms=6, color="tab:red",
                label=f"refresh points ({len(rows)})")
    else:
        ax.plot(xs, values, marker="o", ms=3.5, lw=1.2, color=color,
                label=f"{label} ({len(rows)} samples)")
    scale_text = "linear"
    if symlog:
        ax.set_yscale("symlog", linthresh=SCORE_SYMLOG_LINTHRESH)
        scale_text = f"symlog (linthresh={SCORE_SYMLOG_LINTHRESH:g})"
    elif log_scale:
        ax.set_yscale("log")
        scale_text = "log"
    ax.set_xlabel("sample_order")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n({len(rows)} global-best refresh points, in-sample, "
                 f"y axis: {scale_text})")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend()
    fig.tight_layout()

    out = os.path.join(results_root, png_name)
    try:
        fig.savefig(out, dpi=150)
    except Exception as e:
        print(f"[WARN] 保存 {png_name} 失败（跳过该曲线）: {e}")
        return None
    finally:
        plt.close(fig)
    print(f"[INFO] Saved {out}")
    return {"path": out, "n_points": len(rows), "log_scale": log_scale,
            "symlog": symlog, "scale": scale_text}


def plot_progress_curve(results_root: str) -> dict | None:
    """画两套各三条曲线（逐样本轨迹 / 全局最优刷新点），返回报告小节所需的数字摘要。

    **逐样本**（``samples/*.json``，每个 ``sample_order`` 一个点、折线连接）::

        mse_per_sample_vs_sample_order.png / penalty_per_sample_vs_sample_order.png /
        score_per_sample_vs_sample_order.png

    **刷新点**（``best_history/best_sample_<sample_order>.json``，阶梯线 + 刷新点标记，
    每个文件 = 一次全局最优刷新）::

        mse_vs_sample_order.png / penalty_vs_sample_order.png / score_vs_sample_order.png

    为什么要两套：刷新点曲线只看得到"评分改善"，而实测存在"低 MSE 高罚分 ↔ 高 MSE
    零罚分"两模态——两个模式各自的评分都可能"看起来在改善/停滞"，摆动本身只在
    **逐样本**轨迹上可见（20260926-151008 的 order 53 拟合 MSE 0.1747 却带 7.0e8 罚分）。
    两套共用同一套纵轴规则：MSE 与罚分全为正时取对数（含 0 退回线性），评分恒 symlog。

    没有可用记录时返回 ``None`` 且不生成任何文件，报告侧据此跳过整个小节。返回值除
    六个子摘要（``mse`` / ``penalty`` / ``score`` / ``per_sample``）外，仍保留描述
    **刷新点 MSE 曲线**的顶层键（``path`` / ``n_points`` / ``log_scale`` / ``first`` /
    ``best`` / ``points``），供既有调用方与测试使用。
    """
    points = load_best_history(results_root)
    if not points:
        print("[INFO] 没有可用的 best_history 记录，跳过训练进度图")
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")            # 无头环境；必须在 pyplot 之前
        import matplotlib.pyplot as plt
    except Exception as e:               # 缺 matplotlib 时只告警跳过，不影响实验
        print(f"[WARN] 无法导入 matplotlib，跳过训练进度图: {e}")
        return None

    # 旧目录没有 penalty 字段：那里的 mse 是"拟合 MSE + 体检罚分"，纵轴必须写明，
    # 否则读者会把罚分当拟合质量（实测 20260926-094330：11.999 里 98% 是罚分）。
    legacy = any(p["mse_includes_penalty"] for p in points)
    with_penalty = [p for p in points if p["penalty"] is not None]

    mse = _plot_quantity(
        plt, points, results_root, key="mse", png_name=PROGRESS_PNG_NAME,
        label="best-so-far MSE", ylabel="MSE" if not legacy else "MSE (+ pathology penalty)",
        title="MSE vs sample_order" + (" (legacy records: MSE incl. penalty)" if legacy else ""),
        color="tab:blue")
    if mse is None:                       # MSE 画不出来就不该有小节（其余各条也没意义）
        return None
    penalty = _plot_quantity(
        plt, with_penalty, results_root, key="penalty", png_name=PENALTY_PNG_NAME,
        label="pathology penalty at each refresh", ylabel="dynamic-range penalty",
        title="Dynamic-range penalty vs sample_order", color="tab:orange") \
        if with_penalty else None
    score = _plot_quantity(
        plt, points, results_root, key="score", png_name=SCORE_PNG_NAME,
        label="best-so-far score", ylabel="score = -(fit MSE + penalty)",
        title="Score vs sample_order", color="tab:green", symlog=True)

    # 逐样本轨迹（第二套）：同一套纵轴规则，但画折线、点全部来自 samples/
    samples = load_sample_points(results_root)
    per_sample = None
    if samples:
        sample_penalty = [p for p in samples if p["penalty"] is not None]
        per_sample = {
            "n_points": len(samples),
            "n_clean": sum(1 for p in samples if p["penalty"] == 0),
            "best_score": max(samples, key=lambda p: p["score"]),
            "worst_score": min(samples, key=lambda p: p["score"]),
            "max_penalty": max(sample_penalty, key=lambda p: p["penalty"])
            if sample_penalty else None,
            "mse": _plot_quantity(
                plt, samples, results_root, key="mse", png_name=PER_SAMPLE_MSE_PNG_NAME,
                label="fit MSE per sample", ylabel="MSE",
                title="Fit MSE vs sample_order (every persisted sample)",
                color="tab:blue", step=False),
            "penalty": _plot_quantity(
                plt, sample_penalty, results_root, key="penalty",
                png_name=PER_SAMPLE_PENALTY_PNG_NAME,
                label="pathology penalty per sample", ylabel="dynamic-range penalty",
                title="Dynamic-range penalty vs sample_order (every persisted sample)",
                color="tab:orange", step=False) if sample_penalty else None,
            "score": _plot_quantity(
                plt, samples, results_root, key="score", png_name=PER_SAMPLE_SCORE_PNG_NAME,
                label="score per sample", ylabel="score = -(fit MSE + penalty)",
                title="Score vs sample_order (every persisted sample)",
                color="tab:green", symlog=True, step=False),
        }

    best = min(points, key=lambda p: p["mse"])
    return {"path": mse["path"], "n_points": len(points), "log_scale": mse["log_scale"],
            "legacy_records": legacy, "first": points[0], "best": best,
            "points": points,
            "mse": mse, "penalty": penalty, "score": score, "per_sample": per_sample}


def _fmt_point(point: dict) -> str:
    """把刷新点渲染成 ``sample_order=N，MSE=…，NMSE=…（含体检罚分 X）``。

    ``MSE`` 在口径拆分后是**拟合本身**的 MSE；`penalty` 字段存在时把罚分一并写出——
    罚分可以远大于 MSE 本身（实测 11.755 vs 0.244），不写出来读者会以为拟合很差。
    """
    text = f"sample_order={point['sample_order']}，MSE={point['mse']:.6g}"
    nmse = point.get("nmse")
    try:
        if nmse is not None and np.isfinite(float(nmse)):
            text += f"，NMSE={float(nmse):.4g}"
    except (TypeError, ValueError):
        pass
    penalty = point.get("penalty")
    try:
        if penalty is not None and np.isfinite(float(penalty)) and float(penalty) > 0:
            text += f"（其中动态范围体检罚分 {float(penalty):.6g}）"
    except (TypeError, ValueError):
        pass
    return text


def _fmt_penalty_point(point: dict) -> str:
    """刷新点的罚分（``None`` 时写 ``n/a``——旧目录没有该字段）。"""
    value = _finite_or_none(point.get("penalty"))
    text = "n/a" if value is None else f"{value:.6g}"
    return f"sample_order={point['sample_order']}，罚分={text}"


def _fmt_score_point(point: dict) -> str:
    """刷新点的评分（缺失时写 ``n/a``，绝不把 ``None`` 写进报告）。"""
    value = _finite_or_none(point.get("score"))
    text = "n/a" if value is None else f"{value:.6g}"
    return f"sample_order={point['sample_order']}，评分={text}"


def render_progress_section(progress: dict | None) -> str:
    """渲染 report.md 的「训练进度」小节（机器生成，数字不由 LLM 转述）。

    三条曲线（MSE / 体检罚分 / 评分）共用同一批刷新点，故放在同一节里逐条给出，
    并交代它们之间的构造关系（评分 = −(MSE + 罚分)、刷新点由评分改善定义）。

    ``progress`` 为 ``None``（没有 best_history）时返回空串——调用方据此不插入小节，
    而不是写一节"本次无数据"（那不是实验结果）。
    """
    if not progress:
        return ""
    points = progress["points"]
    first, best = progress["first"], progress["best"]
    scale = "对数刻度" if progress.get("log_scale") else "线性刻度"
    lines = [
        PROGRESS_HEADING,
        "",
        f"本节有两套曲线，回答不同的问题：**逐样本轨迹**（`samples/`，每个已落盘样本"
        f"一个点、折线连接——摆动与个别离群都看得见）与**全局最优刷新点**"
        f"（`best_history/`，只保留评分改善的那几个点、阶梯线——收敛过程看得见）。"
        f"共 **{progress['n_points']}** 个刷新点。",
    ]

    if progress.get("per_sample"):
        per = progress["per_sample"]
        lines += [
            "",
            "### 逐样本轨迹（每个已落盘样本一个点，折线连接）",
            "",
            f"![逐样本拟合 MSE]({PER_SAMPLE_MSE_PNG_NAME})",
            "",
            f"![逐样本体检罚分]({PER_SAMPLE_PENALTY_PNG_NAME})",
            "",
            f"![逐样本评分]({PER_SAMPLE_SCORE_PNG_NAME})",
            "",
            f"- 共 **{per['n_points']}** 个样本点，其中体检罚分 == 0 的有 **{per['n_clean']}** 个。",
            f"- 评分最好：{_fmt_score_point(per['best_score'])}；最差："
            f"{_fmt_score_point(per['worst_score'])}。",
        ]
        if per.get("max_penalty"):
            lines.append(f"- 该批样本里最大罚分：{_fmt_penalty_point(per['max_penalty'])}。")
        lines += [
            "- 与「全局最优刷新点」的区别：刷新点只留评分改善的那几个点，看不见"
            "「低 MSE 高罚分 ↔ 高 MSE 零罚分」的来回摆动——摆动只在**本图**上可见。",
            "- 只含**已落盘**的样本：`persist_all_samples` 之前的实验只留 top-K，"
            "此时点数远小于实际评估数（图上这些「最好/最差」也只覆盖已落盘部分）。",
        ]

    lines += [
        "",
        "### 全局最优刷新点：拟合 MSE",
        "",
        f"![MSE 随 sample_order 的变化]({PROGRESS_PNG_NAME})",
        "",
        f"- 纵轴 {scale}；阶梯线是当时的**历史最优**（best-so-far，保持到下一个刷新点），"
        f"菱形为刷新点。阶梯线覆盖首个刷新点到最后一次刷新点（各点之间的「未改善」区段"
        f"没有额外采样点可画）。",
        f"- 首个刷新点：{_fmt_point(first)}",
        f"- 最终最优：{_fmt_point(best)}",
    ]

    if progress.get("penalty"):
        penalty_rows = [p for p in points if _finite_or_none(p.get("penalty")) is not None]
    else:
        penalty_rows = []
    if penalty_rows:
        worst = max(penalty_rows, key=lambda p: p["penalty"])
        penalty_scale = "对数刻度" if progress["penalty"].get("log_scale") else "线性刻度"
        lines += [
            "",
            "### 全局最优刷新点：体检罚分",
            "",
            f"![体检罚分随 sample_order 的变化]({PENALTY_PNG_NAME})",
            "",
            f"- 纵轴 {penalty_scale}；每个点是**该刷新点当时的全局最优样本**的体检罚分。"
            f"罚分是选择用的偏好、不是拟合质量，故与上一节的 MSE 必须分开读。",
            f"- 首个刷新点：{_fmt_penalty_point(first)}",
            f"- 该曲线上最大罚分：{_fmt_penalty_point(worst)}",
            f"- 末次刷新点：{_fmt_penalty_point(points[-1])}",
        ]
    else:
        lines += [
            "",
            "### 全局最优刷新点：体检罚分",
            "",
            "> 本实验目录的记录里**没有 `penalty` 字段**（口径拆分之前），不画罚分曲线："
            "那里的 `mse` = 拟合 MSE + 体检罚分，两者无法分离。",
        ]

    if progress.get("score"):
        lines += [
            "",
            "### 全局最优刷新点：评分（−(拟合 MSE + 罚分)）",
            "",
            f"![评分随 sample_order 的变化]({SCORE_PNG_NAME})",
            "",
            f"- 纵轴 symlog（评分 ≤ 0 且跨数量级）；阶梯线**单调上升是构造性的**"
            f"——刷新点的定义就是「评分改善」，所以这条线只能看到改善、看不到抖动。",
            f"- 首个刷新点：{_fmt_score_point(first)}",
            f"- 最终最优（最后一次刷新）：{_fmt_score_point(best)}",
        ]

    if progress.get("legacy_records"):
        lines += [
            "",
            "> **口径注意**：本实验是口径拆分之前的目录，记录里**没有 `penalty` 字段**，"
            "其 `mse` = 拟合 MSE + 体检罚分（也就是 −评分），**不能当作拟合质量**读——"
            "纵轴已标注为 `MSE (+ pathology penalty)`。拆分后同一组数字会写成"
            "「mse = 拟合值」+「penalty = 罚分」两个字段，评分不变。",
        ]
    else:
        lines += [
            "",
            "> 口径：MSE 曲线的纵轴是**拟合本身**的均方误差，与「动态范围体检」的罚分分开记账"
            "（评分 = −(拟合 MSE + 罚分)）。罚分可以远大于拟合 MSE——实测 20260926-094330 的"
            "最优解拟合 MSE 仅 0.2438，而罚分 11.755，两者之和才是当时记录的 11.999。",
            "",
            "> 三条曲线在每个刷新点上都满足 |评分| = MSE + 罚分（可逐点自检）。MSE 与罚分的"
            "阶梯线**不是单调的**：刷新由评分定义，所以「MSE 降而罚分升」「MSE 升而罚分降」"
            "都会出现——这正是「低 MSE 高罚分 ↔ 高 MSE 零罚分」两模态（实测 20260926-151008 "
            "的 order 53 拟合 MSE 0.1747 却带 7.0e8 罚分，只看 MSE 一条线会读成「在改善」）。",
        ]
    lines += [
        "",
        "> 该曲线**全程是样本内指标**（评估器在同一批训练点上拟合参数并打分），"
        "低 MSE 不代表泛化能力——泛化对照见上面的「样本外验证」小节。",
    ]
    return "\n".join(lines)


def _strip_progress_section(text: str) -> str:
    """去掉正文里已有的训练进度小节（回填幂等的第一步，与 holdout/range 的 strip 同形）。

    两处与"同形"实现不同的细节，都来自本小节比其他机器小节多出来的结构：

    * 按**前缀** ``## 训练进度`` 匹配，而不是当前的完整标题——本小节从"只有 MSE"
      扩成三条曲线时标题改过一次，用完整标题匹配会让老报告里的旧标题小节**留在原地**、
      再插一节新的 → 同一份报告出现两节；
    * 只把 **h1/h2** 当作"小节结束"，``###`` 子标题（拟合 MSE / 罚分 / 评分）属于本节
      内容——否则剥离会在第一个子标题处停下，留下半节旧内容。
    """
    return strip_section(text, _SECTION_PREFIX,
                         end_check=only_h1_or_h2_ends_section, rstrip=False)


def upsert_progress_section(text: str, section: str) -> str:
    """把训练进度小节写进报告正文：已有同名小节则**整节替换**，否则插到参考文献之前。

    幂等：先整节剥掉旧小节再按固定锚点插回，最后把连续空行收敛成一行——重复回填
    得到逐字节相同的结果（同一个报告不会出现两节，也不会每次多一个空行）。
    锚点缺失（正文没有参考文献小节）时追加到末尾，保证小节不丢。
    """
    return upsert_section(text, section, heading=_SECTION_PREFIX,
                          anchor=_REFERENCE_ANCHOR,
                          end_check=only_h1_or_h2_ends_section)


def backfill(results_root: str, report_name: str = "report.md",
             update_report: bool = True) -> dict | None:
    """对已有实验目录补出进度图（可选同步更新报告小节），返回 ``plot_progress_curve`` 的结果。

    ``update_report=False`` 时只画图，不动报告。报告不存在时只画图。
    """
    progress = plot_progress_curve(results_root)
    if not update_report or progress is None:
        return progress
    path = os.path.join(results_root, report_name)
    if not os.path.exists(path):
        print(f"[INFO] 未找到 {report_name}，只生成图与下面的小节文本")
        print(render_progress_section(progress))
        return progress
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        updated = upsert_progress_section(text, render_progress_section(progress))
        if updated != text:
            with open(path, "w", encoding="utf-8") as f:
                f.write(updated)
            print(f"[INFO] 已更新 {path}")
        else:
            print(f"[INFO] {path} 无需改动（小节已是最新）")
    except Exception as e:
        print(f"[WARN] 更新 {report_name} 失败: {e}")
    return progress


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="训练进度曲线（逐样本 + 全局最优刷新点；拟合 MSE / 体检罚分 / 评分）"
                    " + 报告小节回填")
    parser.add_argument("results_root", help="实验目录")
    parser.add_argument("--report", default="report.md",
                        help="要更新的报告文件名（缺省 report.md）")
    parser.add_argument("--no-report", action="store_true",
                        help="只画图，不更新报告")
    args = parser.parse_args()
    if not os.path.isdir(args.results_root):
        print(f"[WARN] 实验目录不存在: {args.results_root}")
        sys.exit(1)
    backfill(args.results_root, report_name=args.report,
             update_report=not args.no_report)