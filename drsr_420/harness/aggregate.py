"""跨 run 汇总：把 ``experiments/`` 下的多次运行变成一张可复现的指标表。

为什么需要它
------------
``reporting`` 面向**单次** run 产出 ``report.md``；论文的 E1/E2 要的是**跨 run**
的统计（多种子协议下的中位数与 ``Acc@阈值``）。本模块只做三步——读盘、取每个 run
的代表指标、按问题分组汇总——**不重新拟合、不调用 LLM**，因此可以反复跑、结果可复现。

两个口径决定（与单 run 报告刻意不同）
------------------------------------
* **代表指标取"分数最高样本"的样本内 NMSE**（``load_sample_records`` 已按分数降序）。
  发布解门禁（优先发无病理的最高分）是 ``reporting.select_published_sample`` 的
  **单 run 发布决策**；跨 run 统计若也套门禁，就变成"报告口径不一致"
  （同一次 run 在表里与报告里是两个数），故这里只取"该 run 达到的最好样本内 NMSE"。
* **用中位数而不是均值**：NMSE 跨 run 是重尾的（实测样本外/内比值 7.9e3 ~ 8.8e9，
  见 ``docs/RESEARCH_PLAN.md`` §2.5），均值会被离群 run 主导。重尾分布下中位数才是
  稳健摘要；均值±标准差留待样本外（OOD）评估接入后再给（那需要逐 run 重算 test 集，
  不在本模块范围）。

样本外（ID/OOD）列暂缺：要按 run 重算 ``test.csv`` / ``test_ood.csv`` 上的 NMSE，
属 ``reporting.generalization`` 的逐 run 口径，尚未接进本汇总链。
"""
from __future__ import annotations

import dataclasses
import glob
import json
import math
import os
from typing import Sequence

import numpy as np

from drsr_420.equations.records import load_sample_records
from drsr_420.harness.metrics import DEFAULT_ACC_TOL, acc_at

#: 默认扫描的产物根目录（与 ``cli`` 的 ``--results_root`` 默认值一致）。
DEFAULT_ROOT = "experiments"


@dataclasses.dataclass(frozen=True)
class RunMetrics:
    """单次 run 的代表指标（跨 run 汇总的最小输入）。"""
    problem: str
    run_id: str
    seed: int | None
    best_score: float
    best_nmse: float | None
    n_scored: int


@dataclasses.dataclass(frozen=True)
class GroupSummary:
    """按问题分组后的汇总（一行 = 论文表的一行）。"""
    problem: str
    n_runs: int
    n_with_nmse: int
    nmse_median: float | None
    nmse_min: float | None
    acc_at: float | None


def _read_snapshot(run_dir: str) -> dict:
    """读 ``config_snapshot.json``；缺失/损坏时返回空 dict（不阻断汇总）。"""
    path = os.path.join(run_dir, "config_snapshot.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _problem_from_dirname(run_dir: str) -> str:
    """目录名 ``<问题>_<时间戳>`` → 问题名；没有时间戳后缀时原样返回。"""
    name = os.path.basename(os.path.normpath(run_dir))
    head, sep, tail = name.rpartition("_")
    # 时间戳形如 20261004-162656：只有右段像时间戳时才剥掉，避免误伤 a_b 这类名字
    if sep and len(tail) == 15 and tail[8] == "-" and tail.replace("-", "").isdigit():
        return head
    return name


def load_run_metrics(run_dir: str) -> RunMetrics | None:
    """从单个 run 目录取代表指标；无有效样本返回 ``None``。"""
    records = load_sample_records(run_dir)
    if not records:
        return None
    best = records[0]                      # load_sample_records 已按 score 降序
    snapshot = _read_snapshot(run_dir)
    problem = str(snapshot.get("problem_name") or _problem_from_dirname(run_dir))

    raw_seed = snapshot.get("seed")
    seed = int(raw_seed) if isinstance(raw_seed, (int, float)) and not isinstance(raw_seed, bool) else None

    nmse_value = best.get("nmse")
    best_nmse = None
    if isinstance(nmse_value, (int, float)) and not isinstance(nmse_value, bool):
        candidate = float(nmse_value)
        best_nmse = candidate if math.isfinite(candidate) else None

    return RunMetrics(
        problem=problem,
        run_id=os.path.basename(os.path.normpath(run_dir)),
        seed=seed,
        best_score=float(best["score"]),
        best_nmse=best_nmse,
        n_scored=len(records),
    )


def _is_run_dir(path: str) -> bool:
    return (os.path.isfile(os.path.join(path, "config_snapshot.json"))
            or os.path.isdir(os.path.join(path, "samples")))


def collect_runs(root: str = DEFAULT_ROOT) -> list[RunMetrics]:
    """扫描 ``root`` 下**两种**目录布局中的全部 run，返回代表指标列表。

    两种布局并存必须都读（与 ``reporting.find_best_eq._latest_run_dir`` 同理）：
    ``root/<问题>/<问题>_<时间戳>/``（新）与 ``root/<问题>_<时间戳>/``（旧）。
    问题分组目录本身不含样本，会被 :func:`_is_run_dir` 过滤掉。
    """
    candidates = sorted(glob.glob(os.path.join(root, "*", "*")))
    candidates += sorted(glob.glob(os.path.join(root, "*")))
    runs: list[RunMetrics] = []
    seen: set[str] = set()
    for path in candidates:
        key = os.path.normcase(os.path.abspath(path))
        if key in seen or not _is_run_dir(path):
            continue
        seen.add(key)
        metrics = load_run_metrics(path)
        if metrics is not None:
            runs.append(metrics)
    return runs


def summarize(runs: Sequence[RunMetrics], tol: float = DEFAULT_ACC_TOL) -> list[GroupSummary]:
    """按问题分组汇总；每个问题一行。"""
    by_problem: dict[str, list[RunMetrics]] = {}
    for run in runs:
        by_problem.setdefault(run.problem, []).append(run)

    summaries: list[GroupSummary] = []
    for problem in sorted(by_problem):
        items = by_problem[problem]
        nmses = [run.best_nmse for run in items if run.best_nmse is not None]
        summaries.append(GroupSummary(
            problem=problem,
            n_runs=len(items),
            n_with_nmse=len(nmses),
            nmse_median=float(np.median(nmses)) if nmses else None,
            nmse_min=float(np.min(nmses)) if nmses else None,
            acc_at=acc_at(nmses, tol) if nmses else None,
        ))
    return summaries


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3g}"


def format_table(summaries: Sequence[GroupSummary], tol: float = DEFAULT_ACC_TOL) -> str:
    """渲染成 markdown 表（论文 E1 的行骨架）。"""
    lines = [
        f"| 问题 | runs | 有 NMSE | NMSE 中位数 | NMSE 最小 | Acc@{tol:g} |",
        "|---|---|---|---|---|---|",
    ]
    for item in summaries:
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            item.problem, item.n_runs, item.n_with_nmse,
            _fmt(item.nmse_median), _fmt(item.nmse_min),
            "—" if item.acc_at is None else f"{item.acc_at:.3f}"))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m drsr_420.harness.aggregate [root]``：扫描并打印汇总表。"""
    args = list(argv if argv is not None else [])
    root = args[0] if args else DEFAULT_ROOT
    runs = collect_runs(root)
    summaries = summarize(runs)
    print(f"# 跨 run 汇总：{root}（{len(runs)} 个 run）\n")
    print(format_table(summaries))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))