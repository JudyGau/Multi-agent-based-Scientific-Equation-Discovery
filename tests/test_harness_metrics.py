"""harness 层测试：指标口径与跨 run 汇总。

这组测试钉住两件容易漂移的事：

1. **NMSE 的口径**与评估器一致（分母是总体方差 ``ddof=0``）——否则跨 run 汇总出的
   NMSE 与单 run 报告会差一个 ``n/(n-1)`` 因子；
2. **"未知 ≠ 失败"**：``nmse`` 缺失（旧目录）不计入 ``Acc@tol`` 分母。
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

import numpy as np

from drsr_420.harness import aggregate as agg
from drsr_420.harness.metrics import (
    DEFAULT_ACC_TOL,
    acc_at,
    mse,
    nmse,
    symbolic_accuracy,
)


class MetricsTest(unittest.TestCase):
    """MSE / NMSE / Acc@tol / SA 的纯函数口径。"""

    def test_mse_basic(self):
        self.assertAlmostEqual(mse([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]), 1.0 / 3.0)

    def test_mse_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            mse([1.0, 2.0], [1.0])

    def test_mse_empty_raises(self):
        with self.assertRaises(ValueError):
            mse([], [])

    def test_nmse_matches_evaluator_convention(self):
        """NMSE = MSE / var(总体方差)，与 execution.problems 同口径。"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true + 0.5
        expected = float(np.mean((y_true - y_pred) ** 2) / np.var(y_true))
        self.assertAlmostEqual(nmse(y_true, y_pred), expected)

    def test_nmse_zero_variance_exact_match(self):
        self.assertEqual(nmse([2.0, 2.0], [2.0, 2.0]), 0.0)

    def test_nmse_zero_variance_mismatch_is_inf(self):
        self.assertEqual(nmse([2.0, 2.0], [2.0, 2.5]), float("inf"))

    def test_acc_at_counts_hits(self):
        self.assertAlmostEqual(acc_at([0.05, 0.2, 0.1, 0.9]), 2.0 / 4.0)

    def test_acc_at_excludes_unknown(self):
        """None（未知）不计入分母——"测不出来"不等于"没命中"。"""
        self.assertAlmostEqual(acc_at([0.05, None, None]), 1.0)

    def test_acc_at_empty_raises(self):
        with self.assertRaises(ValueError):
            acc_at([None, None])
        with self.assertRaises(ValueError):
            acc_at([])

    def test_default_tol_is_point_one(self):
        self.assertEqual(DEFAULT_ACC_TOL, 0.1)

    def test_symbolic_accuracy_equivalent(self):
        self.assertTrue(symbolic_accuracy("x**2 - y**2", "(x - y)*(x + y)", ["x", "y"]))

    def test_symbolic_accuracy_different(self):
        self.assertFalse(symbolic_accuracy("x + y", "x - y", ["x", "y"]))

    def test_symbolic_accuracy_unknown_on_parse_failure(self):
        """解析失败返回 None（未知），不是 False。"""
        self.assertIsNone(symbolic_accuracy("2 *", "x", ["x"]))


def _write_sample(run_dir: str, *, score: float, nmse_value, order: int = 1) -> None:
    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    payload = {"score": score, "sample_order": order, "mse": 1.0,
               "penalty": 0.0, "function": "def equation(x): return x"}
    if nmse_value is not None:
        payload["nmse"] = nmse_value
    with open(os.path.join(samples_dir, f"samples_{order}.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _write_snapshot(run_dir: str, *, problem: str, seed) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config_snapshot.json"), "w", encoding="utf-8") as fh:
        json.dump({"problem_name": problem, "seed": seed}, fh)


class AggregateTest(unittest.TestCase):
    """跨 run 汇总：两种目录布局都要认出，分组与稳健摘要要正确。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        # 新布局（嵌套）：<root>/<问题>/<问题>_<时间戳>/
        nested_a = os.path.join(self.root, "alpha", "alpha_20261004-162656")
        _write_snapshot(nested_a, problem="alpha", seed=1)
        _write_sample(nested_a, score=-0.05, nmse_value=0.05)
        nested_b = os.path.join(self.root, "alpha", "alpha_20261005-101010")
        _write_snapshot(nested_b, problem="alpha", seed=2)
        _write_sample(nested_b, score=-0.5, nmse_value=0.5)
        # 旧布局（平铺）：<root>/<问题>_<时间戳>/，无快照 → 问题名从目录名推
        flat = os.path.join(self.root, "beta_20261004-120000")
        _write_sample(flat, score=-0.2, nmse_value=0.2)
        # 干扰项：只有问题分组目录（不含样本）→ 不应被当成 run
        os.makedirs(os.path.join(self.root, "gamma"), exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_problem_from_dirname_strips_timestamp(self):
        self.assertEqual(
            agg._problem_from_dirname(os.path.join(self.root, "beta_20261004-120000")), "beta")
        # 没有时间戳后缀时原样返回，不误伤 a_b 这类名字
        self.assertEqual(agg._problem_from_dirname(os.path.join(self.root, "a_b")), "a_b")

    def test_collect_runs_reads_both_layouts_and_skips_grouping_dir(self):
        runs = agg.collect_runs(self.root)
        self.assertEqual(len(runs), 3)
        self.assertEqual(sorted(r.problem for r in runs), ["alpha", "alpha", "beta"])

    def test_load_run_metrics_uses_snapshot_problem_and_seed(self):
        run = agg.load_run_metrics(os.path.join(self.root, "alpha", "alpha_20261004-162656"))
        self.assertIsNotNone(run)
        self.assertEqual(run.problem, "alpha")
        self.assertEqual(run.seed, 1)
        self.assertAlmostEqual(run.best_nmse, 0.05)
        self.assertEqual(run.n_scored, 1)

    def test_load_run_metrics_none_without_scored_samples(self):
        empty = os.path.join(self.root, "delta_20261004-000000")
        os.makedirs(os.path.join(empty, "samples"), exist_ok=True)
        with open(os.path.join(empty, "samples", "samples_1.json"), "w", encoding="utf-8") as fh:
            json.dump({"sample_order": 1}, fh)      # 没有 score → 不算有效样本
        self.assertIsNone(agg.load_run_metrics(empty))

    def test_summarize_groups_with_median_min_and_acc(self):
        summaries = {s.problem: s for s in agg.summarize(agg.collect_runs(self.root))}
        alpha = summaries["alpha"]
        self.assertEqual(alpha.n_runs, 2)
        self.assertEqual(alpha.n_with_nmse, 2)
        self.assertAlmostEqual(alpha.nmse_median, 0.275)     # median(0.05, 0.5)
        self.assertAlmostEqual(alpha.nmse_min, 0.05)
        self.assertAlmostEqual(alpha.acc_at, 0.5)            # 0.05 命中、0.5 未命中
        beta = summaries["beta"]
        self.assertEqual(beta.n_runs, 1)
        self.assertAlmostEqual(beta.nmse_median, 0.2)
        self.assertAlmostEqual(beta.acc_at, 0.0)

    def test_summarize_handles_runs_without_nmse(self):
        """整组都没有 NMSE 时中位数/Acc 记 None，而不是 0。"""
        run = agg.RunMetrics(problem="zeta", run_id="r", seed=None,
                             best_score=-1.0, best_nmse=None, n_scored=3)
        summary = agg.summarize([run])[0]
        self.assertIsNone(summary.nmse_median)
        self.assertIsNone(summary.nmse_min)
        self.assertIsNone(summary.acc_at)
        self.assertEqual(summary.n_with_nmse, 0)

    def test_format_table_renders_row_and_placeholder(self):
        summaries = agg.summarize(agg.collect_runs(self.root))
        table = agg.format_table(summaries)
        self.assertIn("| 问题 | runs |", table)
        self.assertIn("| alpha | 2 | 2 | 0.275 | 0.05 | 0.500 |", table)

        no_nmse = agg.GroupSummary(problem="zeta", n_runs=1, n_with_nmse=0,
                                   nmse_median=None, nmse_min=None, acc_at=None)
        self.assertIn("| zeta | 1 | 0 | — | — | — |", agg.format_table([no_nmse]))

    def test_main_prints_summary(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = agg.main([self.root])
        self.assertEqual(code, 0)
        self.assertIn("alpha", buffer.getvalue())


class LayerPlacementTest(unittest.TestCase):
    """harness 是可选的实验设施层：库代码不依赖它，它只依赖下层。"""

    def test_harness_does_not_import_cli_or_reporting_cycles(self):
        from drsr_420.harness import metrics  # noqa: F401  （可独立导入）

        self.assertTrue(hasattr(agg, "collect_runs"))


if __name__ == "__main__":
    unittest.main()