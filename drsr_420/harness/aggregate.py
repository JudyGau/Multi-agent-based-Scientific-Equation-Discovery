"""跨 run 汇总：把 ``experiments/`` 下的多次运行变成一张可复现的指标表。

为什么需要它
------------
``reporting`` 面向**单次** run 产出 ``report.md``；论文的 E1/E2 要的是**跨 run**
的统计（多种子协议下的中位数与 ``Acc@阈值``）。本模块只做三步——读盘、取每个 run
的代表指标、按问题分组汇总——**不重新拟合、不调用 LLM**，因此可以反复跑、结果可复现。

样本内 vs 样本外（ID / OOD）两条口径
------------------------------------
* **样本内**：取"分数最高样本"的 NMSE（``load_sample_records`` 已按分数降序）。
  发布解门禁（优先发无病理的最高分）是 ``reporting.select_published_sample`` 的
  **单 run 发布决策**；跨 run 统计若也套门禁，就变成"同一次 run 在表里与报告里是两个数"，
  故这里只取"该 run 达到的最好样本内 NMSE"。
* **样本外（ID / OOD）**：默认**不算**（要读测试 CSV 并重新求值）。传
  ``holdout=True``（或 CLI ``--holdout``）时，对每个 run 的最佳样本表达式在
  ``test`` / ``ood`` 上求值——**纯本地计算，不调 LLM**。求值口径完全复用
  ``reporting.generalization.holdout``（同一个 :func:`evaluate_holdout`），
  NMSE 分母取**训练集方差**，与样本内 NMSE 可直接比大小。

  实测提示：MRF 各系的 ``test.csv`` 只有 2 行且落在训练点之间（插值），
  ``data/README.md`` 已声明"不能拿它论证泛化能力"；表里照数报出，引用时须一并看
  ``n``。LSR-Synth 四域与 LLM-SR 真实任务才有统计意义上的 ID/OOD 划分。

用中位数而不是均值：NMSE 跨 run 是重尾的（实测样本外/内比值 7.9e3 ~ 8.8e9，见
``docs/RESEARCH_PLAN.md`` §2.5），均值会被离群 run 主导。
"""
from __future__ import annotations

import contextlib
import dataclasses
import glob
import io
import json
import math
import os
from typing import Sequence

import numpy as np

from drsr_420.equations.records import load_sample_records
from drsr_420.harness.metrics import DEFAULT_ACC_TOL, acc_at

#: 默认扫描的产物根目录（与 ``cli`` 的 ``--results_root`` 默认值一致）。
DEFAULT_ROOT = "experiments"

#: 批量求值时被折叠的单 run 日志行数（见 :func:`_holdout_nmse`）。
_suppressed_lines = 0


@dataclasses.dataclass(frozen=True)
class RunMetrics:
    """单次 run 的代表指标（跨 run 汇总的最小输入）。

    ``nmse_id`` / ``nmse_ood`` 只在 ``holdout=True`` 时才尝试填充，取不到记 ``None``
    （"测不出来"不是"没命中"）。
    """
    problem: str
    run_id: str
    seed: int | None
    best_score: float
    best_nmse: float | None
    n_scored: int
    nmse_id: float | None = None
    nmse_ood: float | None = None


@dataclasses.dataclass(frozen=True)
class GroupSummary:
    """按问题分组后的汇总（一行 = 论文表的一行）。"""
    problem: str
    n_runs: int
    n_with_nmse: int
    nmse_median: float | None
    nmse_min: float | None
    acc_at: float | None
    nmse_id_median: float | None = None
    nmse_ood_median: float | None = None
    acc_id: float | None = None
    acc_ood: float | None = None


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


def _finite_or_none(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _train_variance(run_dir: str) -> float | None:
    """本次运行训练集因变量的**总体方差**（样本外 NMSE 的分母，与评估器同口径）。"""
    from drsr_420.reporting.data_io import load_training_data

    data = load_training_data(run_dir)
    if data is None or data.dtype.names is None:
        return None
    try:
        y = np.asarray(data[data.dtype.names[-1]], dtype=float)
    except (ValueError, TypeError):
        return None
    if y.size == 0:
        return None
    variance = float(np.var(y))
    return variance if math.isfinite(variance) and variance > 0 else None


def _holdout_nmse(run_dir: str, function: str, params) -> tuple[float | None, float | None]:
    """在 ID / OOD 测试集上求值最佳样本表达式，返回两者的 NMSE（取不到记 ``None``）。

    口径与单 run 报告**同源**：复用 ``reporting.generalization.holdout`` 的
    ``evaluate_holdout`` 与路径自动探测（``test.csv``/``test_id.csv``、
    ``ood_test.csv``/``test_ood.csv``），因此表里的数字与 ``report.md`` 一致。
    sympy 相关导入放在函数内：默认（不算样本外）路径不必拖起 sympy。
    """
    from drsr_420.equations.header import parse_symbols
    from drsr_420.equations.parse import expr_substitution
    from drsr_420.reporting.generalization.holdout import (
        evaluate_holdout, load_ood_data, load_test_data)

    # 折叠单 run 求值日志：`expr_substitution`（会打印代入后的表达式）、路径探测与
    # 训练数据读取都会对每个 run 各打印若干行，77~500 个 run 会把表格彻底淹没。
    # 异常照常抛出，只折叠常规输出。
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        parsed = parse_symbols(function or "")
        result: tuple[float | None, float | None] = (None, None)
        if parsed is not None:
            dependent, sym_names = parsed
            expr = expr_substitution(function, list(params or []))
            if expr is not None:
                train_var = _train_variance(run_dir)
                values: list[float | None] = []
                for loader in (load_test_data, load_ood_data):
                    data = loader(run_dir)
                    metrics = (evaluate_holdout(dependent, sym_names, expr, data,
                                                train_var=train_var, path="")
                               if data is not None else None)
                    values.append(_finite_or_none(metrics.get("nmse")) if metrics else None)
                result = (values[0], values[1])
    global _suppressed_lines
    _suppressed_lines += buffer.getvalue().count("\n")
    return result


def load_run_metrics(run_dir: str, *, holdout: bool = False) -> RunMetrics | None:
    """从单个 run 目录取代表指标；无有效样本返回 ``None``。

    ``holdout=True`` 时额外在 ID/OOD 测试集上求值最佳样本表达式（纯本地计算）。
    """
    records = load_sample_records(run_dir)
    if not records:
        return None
    best = records[0]                      # load_sample_records 已按 score 降序
    snapshot = _read_snapshot(run_dir)
    problem = str(snapshot.get("problem_name") or _problem_from_dirname(run_dir))

    raw_seed = snapshot.get("seed")
    seed = int(raw_seed) if isinstance(raw_seed, (int, float)) and not isinstance(raw_seed, bool) else None

    nmse_id = nmse_ood = None
    if holdout:
        nmse_id, nmse_ood = _holdout_nmse(run_dir, best.get("function", ""), best.get("params"))

    return RunMetrics(
        problem=problem,
        run_id=os.path.basename(os.path.normpath(run_dir)),
        seed=seed,
        best_score=float(best["score"]),
        best_nmse=_finite_or_none(best.get("nmse")),
        n_scored=len(records),
        nmse_id=nmse_id,
        nmse_ood=nmse_ood,
    )


def _is_run_dir(path: str) -> bool:
    return (os.path.isfile(os.path.join(path, "config_snapshot.json"))
            or os.path.isdir(os.path.join(path, "samples")))


def collect_runs(root: str = DEFAULT_ROOT, *, holdout: bool = False) -> list[RunMetrics]:
    """扫描 ``root`` 下**两种**目录布局中的全部 run，返回代表指标列表。

    两种布局并存必须都读（与 ``reporting.find_best_eq._latest_run_dir`` 同理）：
    ``root/<问题>/<问题>_<时间戳>/``（新）与 ``root/<问题>_<时间戳>/``（旧）。
    问题分组目录本身不含样本，会被 :func:`_is_run_dir` 过滤掉。

    ``holdout=True`` 时对每个 run 额外算 ID/OOD（见 :func:`_holdout_nmse`）。
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
        metrics = load_run_metrics(path, holdout=holdout)
        if metrics is not None:
            runs.append(metrics)
    return runs


def _median(values: Sequence[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return float(np.median(known)) if known else None


def summarize(runs: Sequence[RunMetrics], tol: float = DEFAULT_ACC_TOL) -> list[GroupSummary]:
    """按问题分组汇总；每个问题一行。"""
    by_problem: dict[str, list[RunMetrics]] = {}
    for run in runs:
        by_problem.setdefault(run.problem, []).append(run)

    summaries: list[GroupSummary] = []
    for problem in sorted(by_problem):
        items = by_problem[problem]
        nmses = [run.best_nmse for run in items if run.best_nmse is not None]
        ids = [run.nmse_id for run in items if run.nmse_id is not None]
        oods = [run.nmse_ood for run in items if run.nmse_ood is not None]
        summaries.append(GroupSummary(
            problem=problem,
            n_runs=len(items),
            n_with_nmse=len(nmses),
            nmse_median=_median(nmses),
            nmse_min=float(np.min(nmses)) if nmses else None,
            acc_at=acc_at(nmses, tol) if nmses else None,
            nmse_id_median=_median(ids),
            nmse_ood_median=_median(oods),
            acc_id=acc_at(ids, tol) if ids else None,
            acc_ood=acc_at(oods, tol) if oods else None,
        ))
    return summaries


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3g}"


def _fmt_acc(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def format_table(summaries: Sequence[GroupSummary], tol: float = DEFAULT_ACC_TOL) -> str:
    """渲染成 markdown 表（论文 E1 的行骨架）。

    ID/OOD 列在未计算时显示 ``—``（``--holdout`` 才会填）——"列在但空"比"列直接消失"
    更诚实：读者能看出这一列是**没测**，而不是不存在这个指标。
    """
    lines = [
        f"| 问题 | runs | 有 NMSE | NMSE 中位数 | NMSE 最小 | Acc@{tol:g} "
        f"| NMSE_ID | NMSE_OOD |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in summaries:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            item.problem, item.n_runs, item.n_with_nmse,
            _fmt(item.nmse_median), _fmt(item.nmse_min), _fmt_acc(item.acc_at),
            _fmt(item.nmse_id_median), _fmt(item.nmse_ood_median)))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m drsr_420.harness.aggregate [root] [--holdout]``：扫描并打印汇总表。

    ``--holdout`` 额外在 ID/OOD 测试集上求值（纯本地计算，不调 LLM），用于补论文的
    ``NMSE_ID`` / ``NMSE_OOD`` 两列。
    """
    args = list(argv if argv is not None else [])
    holdout = "--holdout" in args
    positional = [a for a in args if not a.startswith("-")]
    root = positional[0] if positional else DEFAULT_ROOT
    runs = collect_runs(root, holdout=holdout)
    summaries = summarize(runs)
    scope = "样本内 + ID/OOD" if holdout else "仅样本内（加 --holdout 才在测试集上求值）"
    print(f"# 跨 run 汇总：{root}（{len(runs)} 个 run；口径：{scope}）\n")
    print(format_table(summaries))
    if _suppressed_lines:
        print(f"\n> 已折叠 {_suppressed_lines} 行单 run 求值日志（含路径探测与表达式代入提示）。")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))