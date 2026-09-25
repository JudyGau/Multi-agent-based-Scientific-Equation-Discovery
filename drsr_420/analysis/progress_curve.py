"""训练进度曲线：把"历史最优刷新点"画成 MSE 随 sample_order 的阶梯图，并渲染
report.md 的机器生成小节。

用途
----
收尾时回答"这一次搜索是怎么收敛的"：最优解在第几个样本出现、中途刷新过几次、
末段的改善是台阶还是抖动。曲线画在 ``<results_root>/mse_vs_sample_order.png``，
小节由系统生成（数字不经 LLM 转述），排在「动态范围体检」之后、「参考文献」之前。

口径
----
* **只读实验目录内的机器产物** ``best_history/best_sample_<sample_order>.json``
  ——评估器每刷新一次全局最优就写一个文件，含 ``sample_order`` / ``iteration`` /
  ``mse`` / ``nmse``。这些文件本身就是"刷新点"，故图上的阶梯用 ``step(where="post")``
  表示"该最优值保持到下一个刷新点"。
* **不解析 run.out 的逐样本分数**：``MRFCompress-Cuboid.bat`` 等启动入口并不重定向
  stdout（实测其调用是裸的 ``python -m drsr_420.cli.main …``），``run.out`` 不是每条
  启动路径都存在的产物，把它当数据源会让报告在别的启动方式下缺图。
* 曲线**全程是样本内指标**（评估器在同一批训练点上拟合参数并打分），低 MSE 不代表
  泛化——小节末尾写明这一点，泛化对照归「样本外验证」小节。

用法
----
::

    python -m drsr_420.analysis.progress_curve <results_root>

回填：对已有实验目录重跑本模块即可补出图与小节（不触发任何 LLM 调用）。
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

#: 进度图文件名（报告小节按相对路径引用它）。
PROGRESS_PNG_NAME = "mse_vs_sample_order.png"
#: report.md 里训练进度小节的标题（机器生成）。
PROGRESS_HEADING = "## 训练进度：MSE 随 sample_order 的变化"

#: 插入位置锚点：机器小节排在参考文献之前。
#: 这里刻意**不** import ``explain.REFERENCE_HEADING``——explain 会 import 本模块来
#: 装配小节，反向在模块级 import 会成环；两者的取值一致性由
#: ``tests/test_progress_curve.py`` 断言守住（单一来源靠测试而非重复 import）。
_REFERENCE_ANCHOR = "## 参考文献"

_ORDER_RE = re.compile(r"best_sample_(\d+)\.json$")


def load_best_history(results_root: str) -> list[dict]:
    """读 ``best_history/best_sample_*.json``，按 ``sample_order`` 升序返回刷新点。

    单个文件读坏/字段不全/``mse`` 非有限时只告警跳过（不因为一个坏文件丢掉整条曲线）。
    返回项：``{"sample_order", "iteration", "mse", "nmse"}``（``iteration`` / ``nmse``
    可能为 ``None``，历史目录缺字段时按缺失处理）。
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
        rows.append({"sample_order": order, "iteration": rec.get("iteration"),
                     "mse": mse, "nmse": rec.get("nmse")})
    # 兜底：文件名里的 order 与内容不一致时以内容为准（内容才是评估器写的）
    rows.sort(key=lambda r: r["sample_order"])
    return rows


def plot_progress_curve(results_root: str) -> dict | None:
    """画 MSE 随 sample_order 的变化（历史最优阶梯 + 刷新点标记）。

    没有可用记录（目录为空 / 全部读坏）时返回 ``None`` 且**不生成**任何文件，
    报告侧据此跳过该小节。返回值（同时作为渲染小节的唯一数字来源）::

        {"path", "n_points", "log_scale", "first", "best", "points"}

    纵轴取对数：同一次实验内 MSE 可跨数个数量级（实测本问题 1.79 ~ 4.4e4）；
    任一 MSE ≤ 0 时退回线性刻度（对数下无法表示）。
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

    xs = np.asarray([p["sample_order"] for p in points], dtype=float)
    ys = np.asarray([p["mse"] for p in points], dtype=float)
    log_scale = bool(np.all(ys > 0))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(xs, ys, where="post", color="tab:blue", lw=2,
            label="best-so-far MSE (held to next refresh)")
    ax.plot(xs, ys, linestyle="none", marker="D", ms=6, color="tab:red",
            label=f"best-so-far refresh points ({len(points)})")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("sample_order")
    ax.set_ylabel("MSE")
    ax.set_title(f"MSE vs sample_order"
                 f"\n(best-so-far; {len(points)} refresh points, "
                 f"in-sample on the training points)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend()
    fig.tight_layout()

    out = os.path.join(results_root, PROGRESS_PNG_NAME)
    try:
        fig.savefig(out, dpi=150)
    except Exception as e:
        print(f"[WARN] 保存训练进度图失败（跳过该小节）: {e}")
        return None
    finally:
        plt.close(fig)
    print(f"[INFO] Saved {out}")

    best = min(points, key=lambda p: p["mse"])
    return {"path": out, "n_points": len(points), "log_scale": log_scale,
            "first": points[0], "best": best, "points": points}


def _fmt_point(point: dict) -> str:
    """把一个刷新点渲染成 ``sample_order=N，MSE=…，NMSE=…``（缺字段就不写该字段）。"""
    text = f"sample_order={point['sample_order']}，MSE={point['mse']:.6g}"
    nmse = point.get("nmse")
    try:
        if nmse is not None and np.isfinite(float(nmse)):
            text += f"，NMSE={float(nmse):.4g}"
    except (TypeError, ValueError):
        pass
    return text


def render_progress_section(progress: dict | None) -> str:
    """渲染 report.md 的「训练进度」小节（机器生成，数字不由 LLM 转述）。

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
        f"![MSE 随 sample_order 的变化]({PROGRESS_PNG_NAME})",
        "",
        f"数据源：`best_history/best_sample_<sample_order>.json`——评估器每刷新一次"
        f"全局最优写一个文件，共 **{progress['n_points']}** 个刷新点。横轴为全局样本"
        f"序号，纵轴为 MSE（{scale}）；阶梯线是当时的**历史最优**（best-so-far，"
        f"保持到下一个刷新点），菱形为刷新点。阶梯线覆盖首个刷新点到最后一次刷新点"
        f"（各点之间的「未改善」区段没有额外采样点可画）。",
        "",
        f"- 首个刷新点：{_fmt_point(first)}",
        f"- 最终最优：{_fmt_point(best)}",
        "",
        "> 口径：该曲线**全程是样本内指标**（评估器在同一批训练点上拟合参数并打分），"
        "低 MSE 不代表泛化能力——泛化对照见上面的「样本外验证」小节。",
    ]
    return "\n".join(lines)


def _strip_progress_section(text: str) -> str:
    """去掉正文里已有的训练进度小节（回填幂等的第一步，与 holdout/range 的 strip 同形）。"""
    if not text or PROGRESS_HEADING not in text:
        return text
    kept: list[str] = []
    skipping = False
    for line in text.splitlines():
        if line.strip().startswith(PROGRESS_HEADING):
            skipping = True
            continue
        if skipping and line.startswith("#"):
            skipping = False
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


def upsert_progress_section(text: str, section: str) -> str:
    """把训练进度小节写进报告正文：已有同名小节则**整节替换**，否则插到参考文献之前。

    幂等：先整节剥掉旧小节再按固定锚点插回，最后把连续空行收敛成一行——重复回填
    得到逐字节相同的结果（同一个报告不会出现两节，也不会每次多一个空行）。
    锚点缺失（正文没有参考文献小节）时追加到末尾，保证小节不丢。
    """
    if not section:
        return text
    lines = _strip_progress_section(text).splitlines()
    anchor_at = next((i for i, line in enumerate(lines)
                      if line.strip().startswith(_REFERENCE_ANCHOR)), None)
    if anchor_at is None:
        head, tail = "\n".join(lines).rstrip(), ""
    else:
        head = "\n".join(lines[:anchor_at]).rstrip()
        tail = "\n".join(lines[anchor_at:]).rstrip()
    merged = f"{head}\n\n{section}" if head else section
    if tail:
        merged += f"\n\n{tail}"
    return re.sub(r"\n{3,}", "\n\n", merged).rstrip() + "\n"


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
        description="训练进度曲线（MSE 随 sample_order）+ 报告小节回填")
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