"""敏感度剪枝单元测试：求值器（采样/指标/聚合/缓存）与剪枝器（阈值、可复现性、边界）。"""
import unittest

import numpy as np
import sympy as sp

from drsr_420.analysis.expr_evaluation import ExpressionEvaluator
from drsr_420.analysis.prune_stats import PruneRecord, PruneStats
from drsr_420.analysis.sensitivity_prune import SensitivityPruner, sensitivity_prune

x, y, z = sp.symbols("x y z", real=True)
eps = sp.Rational(1, 1000)


class PruneStatsTest(unittest.TestCase):
    def test_prune_rate_zero_when_no_visits(self):
        self.assertEqual(PruneStats().prune_rate, 0.0)

    def test_prune_rate_ratio(self):
        st = PruneStats(nodes_visited=4, nodes_pruned=1)
        self.assertAlmostEqual(st.prune_rate, 0.25)

    def test_summary_contains_counts_and_records(self):
        st = PruneStats(nodes_visited=3, nodes_pruned=1)
        st.records.append(PruneRecord("term_of_Add", x, 1e-4, 0))
        text = st.summary()
        self.assertIn("节点访问数", text)
        self.assertIn("节点剪枝数", text)
        self.assertIn("term_of_Add", text)


class InitValidationTest(unittest.TestCase):
    def test_empty_symbols_raises(self):
        with self.assertRaises(ValueError):
            SensitivityPruner([])

    def test_bad_metric_raises(self):
        with self.assertRaises(ValueError):
            SensitivityPruner([x], metric="nope")

    def test_bad_reduction_raises(self):
        with self.assertRaises(ValueError):
            SensitivityPruner([x], reduction="nope")

    def test_samples_shape(self):
        ev = ExpressionEvaluator([x, y], num_samples=37, seed=1)
        self.assertEqual(ev.samples.shape, (37, 2))
        self.assertEqual(len(ev.points), 2)
        self.assertEqual(ev.points[0].shape, (37,))


class EvaluatorValidationTest(unittest.TestCase):
    def test_empty_symbols_raises(self):
        with self.assertRaises(ValueError):
            ExpressionEvaluator([])

    def test_bad_metric_raises(self):
        with self.assertRaises(ValueError):
            ExpressionEvaluator([x], metric="nope")

    def test_bad_reduction_raises(self):
        with self.assertRaises(ValueError):
            ExpressionEvaluator([x], reduction="nope")

    def test_points_are_columns_of_samples(self):
        ev = ExpressionEvaluator([x, y], num_samples=11, seed=3)
        np.testing.assert_array_equal(ev.points[1], ev.samples[:, 1])


class ReproducibilityTest(unittest.TestCase):
    def test_same_seed_same_samples_and_result(self):
        p1 = SensitivityPruner([x, y], seed=42)
        p2 = SensitivityPruner([x, y], seed=42)
        np.testing.assert_array_equal(p1.evaluator.samples, p2.evaluator.samples)
        expr = x**2 + y**2 + eps * x * y
        self.assertEqual(str(p1.prune(expr)), str(p2.prune(expr)))

    def test_different_seed_differs(self):
        e1 = ExpressionEvaluator([x, y], seed=1)
        e2 = ExpressionEvaluator([x, y], seed=2)
        self.assertFalse(np.array_equal(e1.samples, e2.samples))


class SensitivityMetricTest(unittest.TestCase):
    def test_absolute_max(self):
        ev = ExpressionEvaluator([x], metric="absolute", reduction="max")
        # diff = [0, 2] -> max = 2
        s = ev.sensitivity(np.array([1.0, 2.0]), np.array([1.0, 4.0]))
        self.assertAlmostEqual(s, 2.0)

    def test_relative_max(self):
        ev = ExpressionEvaluator([x], metric="relative", reduction="max")
        # diff=[1,2], denom=[1,2] -> [1,1] -> max=1
        s = ev.sensitivity(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
        self.assertAlmostEqual(s, 1.0)

    def test_reduction_variants(self):
        orig = np.array([1.0, 1.0, 1.0, 1.0])
        pruned = np.array([1.0, 1.0, 1.0, 5.0])  # diff=[0,0,0,4]
        self.assertAlmostEqual(
            ExpressionEvaluator([x], metric="absolute", reduction="mean")
            .sensitivity(orig, pruned), 1.0)
        self.assertAlmostEqual(
            ExpressionEvaluator([x], metric="absolute", reduction="median")
            .sensitivity(orig, pruned), 0.0)
        self.assertAlmostEqual(
            ExpressionEvaluator([x], metric="absolute", reduction="p95")
            .sensitivity(orig, pruned), 3.4, places=6)  # 线性插值：0.95*(4-1)=2.85 -> 3.4
        self.assertAlmostEqual(
            ExpressionEvaluator([x], metric="absolute", reduction="max")
            .sensitivity(orig, pruned), 4.0)

    def test_all_nan_returns_inf(self):
        # 求值不可用 = "测不出来"，不是"敏感度 0"：后者会让每一项都被判为可删
        ev = ExpressionEvaluator([x])
        s = ev.sensitivity(np.array([np.nan]), np.array([np.nan]))
        self.assertEqual(s, float("inf"))


class UnevaluableExpressionTest(unittest.TestCase):
    """表达式求不出有限值时：测不出敏感度 → 全部保留，且统计里必须可见。

    真实事故（MRFCompress-Cuboid_20260919-143926）：含孤儿符号的表达式求值全 NaN，
    旧实现把 NaN 判成敏感度 0 → 每一项都被删 → 公式塌缩成常数 193.054。
    """

    def test_nothing_pruned_and_visible_in_stats(self):
        expr = sp.Symbol("t") + x**2 + y**2      # t 未定义：lambdify/subs 都求不出数
        pruner = SensitivityPruner([x, y], threshold=0.5, sample_range=(1, 6), seed=42)
        published = pruner.prune(expr)
        self.assertEqual(pruner.stats.nodes_pruned, 0)
        self.assertIs(published, expr)
        self.assertGreater(pruner.stats.nonfinite_sensitivity, 0)
        self.assertIn("无法测量项", pruner.stats.summary())

    def test_evaluable_subterm_alone_is_still_prunable(self):
        # 单个候选自身不可求值时不参与排序；可求值的父节点仍按原判据正常工作
        expr = x**2 + y**2 + eps * x * y
        pruner = SensitivityPruner([x, y], threshold=0.01, sample_range=(1, 6), seed=42)
        pruner.prune(expr)
        self.assertEqual(pruner.stats.nonfinite_sensitivity, 0)
        self.assertGreater(pruner.stats.nodes_pruned, 0)


class PruneBehaviorTest(unittest.TestCase):
    def test_drops_negligible_cross_term(self):
        expr = x**2 + y**2 + z**2 + eps * x * y
        pruner = SensitivityPruner([x, y, z], threshold=0.02, seed=42)
        pruned = pruner.prune(expr)
        self.assertNotIn("x*y", str(pruned))
        for keep in ("x**2", "y**2", "z**2"):
            self.assertIn(keep, str(pruned))
        self.assertGreater(pruner.stats.nodes_pruned, 0)

    def test_negative_case_no_pruning(self):
        expr = x**2 + 2 * x + 1
        pruner = SensitivityPruner([x], threshold=0.01, seed=42)
        pruned = pruner.prune(expr)
        self.assertEqual(pruner.stats.nodes_pruned, 0)
        self.assertEqual(sp.expand(pruned - expr), 0)

    def test_threshold_monotonic(self):
        expr = x**2 + 0.1 * x * y + y**2
        low = SensitivityPruner([x, y], threshold=1e-6, seed=42)
        high = SensitivityPruner([x, y], threshold=0.9, seed=42)
        low.prune(expr)
        high.prune(expr)
        self.assertGreaterEqual(high.stats.nodes_pruned, low.stats.nodes_pruned)

    def test_atom_unchanged(self):
        pruner = SensitivityPruner([x])
        self.assertEqual(pruner.prune(x), x)
        self.assertEqual(pruner.stats.nodes_visited, 0)

    def test_single_term_add_not_removed(self):
        # Add 只有一项时不得清空（至少保留一项）
        pruner = SensitivityPruner([x], threshold=0.99, seed=42)
        pruned = pruner.prune(x**2)
        self.assertEqual(sp.simplify(pruned - x**2), 0)

    def test_mul_keeps_one_symbolic_factor(self):
        # 乘积即使阈值极高也应保留至少一个符号因子（不塌缩为 1）
        expr = eps * x * y
        pruner = SensitivityPruner([x, y], threshold=0.99, seed=42)
        pruned = pruner.prune(expr)
        self.assertNotEqual(pruned, sp.Integer(1))

    def test_verbose_flag_smoke(self):
        # verbose 仅影响打印，不应抛错
        pruner = SensitivityPruner([x], threshold=0.5, seed=42)
        pruner.prune(x**2 + eps, verbose=True)


class EvaluateCacheTest(unittest.TestCase):
    def test_repeated_evaluate_hits_cache(self):
        ev = ExpressionEvaluator([x], num_samples=10, seed=42)
        expr = x + 1
        r1 = ev.evaluate(expr)
        r2 = ev.evaluate(expr)
        self.assertIs(r1, r2)  # 同一对象：命中 repr 缓存

    def test_clear_cache_forces_recompute(self):
        ev = ExpressionEvaluator([x], num_samples=10, seed=42)
        r1 = ev.evaluate(x + 1)
        ev.clear_cache()
        r2 = ev.evaluate(x + 1)
        self.assertIsNot(r1, r2)
        np.testing.assert_array_equal(r1, r2)

    def test_evaluate_shapes(self):
        ev = ExpressionEvaluator([x, y], num_samples=15, seed=42)
        self.assertEqual(ev.evaluate(x + y).shape, (15,))

    def test_scalar_expression_broadcasts_to_samples(self):
        ev = ExpressionEvaluator([x], num_samples=7, seed=42)
        self.assertEqual(ev.evaluate(sp.Integer(3)).shape, (7,))

    def test_eval_fallback_slow_path(self):
        # 构造 lambdify 无法处理的形式（sp.zeta 无 numpy 对应）时走 subs 备用路径
        ev = ExpressionEvaluator([x], num_samples=5, seed=42)
        vals = ev.evaluate(sp.zeta(x))
        self.assertEqual(vals.shape, (5,))


class TopLevelFunctionTest(unittest.TestCase):
    def test_returns_expr_and_stats(self):
        expr = x**2 + y**2 + eps * x * y
        pruned, stats = sensitivity_prune(expr, [x, y], threshold=0.02, seed=42)
        self.assertIsInstance(stats, PruneStats)
        self.assertNotIn("x*y", str(pruned))


if __name__ == "__main__":
    unittest.main()
