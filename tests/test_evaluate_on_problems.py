"""evaluation/problems.py 单元测试：least_squares 优化、统一返回契约、配置项。"""
import unittest
import warnings
from unittest import mock

import numpy as np

from drsr_420.evaluation import problems as eop


def make_dataset(n=200, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 + noise * rng.standard_normal(n)
    return {'inputs': X, 'outputs': y}


def linear_equation(x1, x2, params):
    return params[0] * x1 + params[1] * x2 + params[2]


class EvaluateTest(unittest.TestCase):
    def test_returns_negative_score_and_matrices(self):
        score, matrix, params = eop.evaluate(make_dataset(), linear_equation)
        self.assertIsInstance(score, float)
        self.assertLess(score, 0.0)
        self.assertEqual(matrix.shape, (200, 4))  # (输入2列, 输出, 残差)
        self.assertEqual(params.shape, (eop.MAX_NPARAMS,))

    def test_recovers_ground_truth_params(self):
        _, _, params = eop.evaluate(make_dataset(), linear_equation)
        np.testing.assert_allclose(params[:3], [3.0, -2.0, 0.5], atol=0.1)

    def test_score_approximates_noise_variance(self):
        # 模型形式正确时，score = -MSE ≈ -noise²
        score, _, _ = eop.evaluate(make_dataset(noise=0.01), linear_equation, seed=1)
        self.assertAlmostEqual(score, -0.01 ** 2, delta=2e-5)

    def test_failure_contract_for_nan_equation(self):
        """全起点因 NaN 抛错 → 真实原因上抛（不再伪装成无信息的 (None,None,None)）。

        scipy 对 NaN 初始残差抛 ValueError；若继续吞掉，_run_evaluation_task
        只会给经验回路喂 'no output'，模型学不到任何东西。"""
        def nan_equation(x1, x2, params):
            return np.full_like(x1, np.nan)

        with self.assertRaises(ValueError) as ctx:
            eop.evaluate(make_dataset(), nan_equation)
        self.assertIn('not finite', str(ctx.exception))

    def test_failure_contract_for_exception_equation(self):
        """方程自身异常（越界索引）必须穿透到 remark：'Execution Error: index...'。"""
        def bad_equation(x1, x2, params):
            return params[999] * x1  # IndexError：所有起点失败

        with self.assertRaises(IndexError):
            eop.evaluate(make_dataset(), bad_equation)

    def test_pure_nonfinite_loss_without_exception(self):
        """least_squares 正常返回但损失非有限（result.fun 含 inf）→ 不抛错、
        best_x 仍为 None，_multi_start 维持 (None, inf) 契约。

        用 mock 注入"成功但结果非有限"的返回值（_multi_start 只读 .fun/.x，
        鸭子类型即可），隔离 scipy 对 NaN 初始残差直接抛错的行为。"""
        import types
        fake = types.SimpleNamespace(fun=np.array([np.inf]), x=np.zeros(3))
        with mock.patch('scipy.optimize.least_squares', return_value=fake):
            best_x, best_loss = eop._multi_start_least_squares(
                lambda p: np.array([1.0, 2.0]), 3, n_starts=1)
        self.assertIsNone(best_x)
        self.assertEqual(best_loss, np.inf)

    def test_warm_start_x0(self):
        data = make_dataset()
        _, _, params = eop.evaluate(data, linear_equation, seed=1)
        score2, _, _ = eop.evaluate(data, linear_equation, x0=params, seed=1)
        self.assertLess(score2, 0.0)

    def test_decimal_places_respected(self):
        _, matrix, _ = eop.evaluate(make_dataset(), linear_equation, decimal_places=3)
        # 展示列（输入/输出）按 decimal_places 取整；残差列保持全精度——
        # 它是 ResidualAnalyzerAgent 的唯一输入，按绝对小数位取整会把
        # 拟合越好的样本清零（第 8 轮修复）。
        display = np.delete(matrix, matrix.shape[1] - 1, axis=1)
        np.testing.assert_allclose(display / 0.001, np.round(display / 0.001))


class HelperTest(unittest.TestCase):
    def test_clamp_params(self):
        out = eop._clamp_params(np.array([-20.0, 5.0, 20.0]), (-10.0, 10.0))
        np.testing.assert_array_equal(out, [-10.0, 5.0, 10.0])

    def test_multi_start_all_raise_propagates_reason(self):
        """所有起点都抛同一类异常 → 首个真实异常上抛（不再静默 (None, inf)）。"""
        def always_raise(params):
            raise ValueError('boom')

        with self.assertRaises(ValueError) as ctx:
            eop._multi_start_least_squares(always_raise, 3, n_starts=2)
        self.assertIn('boom', str(ctx.exception))


def mrf_like_dataset(n=8, seed=7):
    """量级与 data/MRFCompress-Cuboid 一致的数据集（输出 ~1e2，输入 1~14）。"""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(1.0, 5.0, n), rng.uniform(1.0, 14.0, n)])
    return {'inputs': X, 'outputs': rng.uniform(190.0, 350.0, n)}


def free_exponent_equation(l12, l23, params):
    """LLM 最常写的形态：参数当指数用 → 优化器探到边界附近必然溢出。"""
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        return (params[0] + params[1] * l12 ** params[2] + params[3] * l23 ** params[4]
                + params[5] * (l12 * l23) ** params[6])


class ResidualSanitizeTest(unittest.TestCase):
    """第 9 轮：残差必须先清洗/截断再交给 scipy（RuntimeWarning 治理）。

    实测（修复前）：scale≥1e50 的残差让 trf/common 内部刷出 876 条
    "overflow encountered in power/square"、"invalid value encountered in cast"，
    scale=1e200 一档直接 ValueError；修复后同规模扫描 0 条。
    """

    def test_sanitize_replaces_nonfinite_with_signed_cap(self):
        out = eop._sanitize_residual(
            np.array([1.0, np.nan, np.inf, -np.inf, 1e300]), cap=1e12)
        np.testing.assert_array_equal(out, [1.0, 1e12, 1e12, -1e12, 1e12])

    def test_sanitize_caps_huge_finite_values(self):
        """有限但巨大的残差同样会击穿 scipy 内部——必须一起截断。"""
        out = eop._sanitize_residual(np.array([-1e60, 1e150]), cap=1e12)
        np.testing.assert_array_equal(out, [-1e12, 1e12])

    def test_sanitize_returns_same_object_when_in_range(self):
        res = np.array([0.1, -3.0, 0.0])
        self.assertIs(eop._sanitize_residual(res, cap=1e12), res)  # 常规路径零拷贝

    def test_residual_cap_follows_output_scale(self):
        self.assertEqual(eop.residual_cap(np.array([1.0, 350.0])), eop.RESIDUAL_CAP)
        self.assertEqual(eop.residual_cap(np.array([1e14])), eop.RESIDUAL_CAP_RATIO * 1e14)
        self.assertEqual(eop.residual_cap(np.array([np.nan])), eop.RESIDUAL_CAP)

    def test_residual_handed_to_scipy_is_bounded(self):
        """契约测试：无论方程在极端参数下返回什么，交给 scipy 的残差都有界。"""
        dataset = mrf_like_dataset()
        captured = {}

        def fake_least_squares(fun, x0, **kwargs):
            captured['fun'] = fun
            return mock.Mock(fun=np.asarray(fun(x0), dtype=float),
                             x=np.asarray(x0, dtype=float))

        with mock.patch('scipy.optimize.least_squares', side_effect=fake_least_squares):
            eop.evaluate(dataset, free_exponent_equation, seed=0)

        cap = eop.residual_cap(dataset['outputs'])
        residual = captured['fun'](np.full(eop.MAX_NPARAMS, 1000.0))  # 参数顶到上界
        self.assertTrue(np.isfinite(residual).all())
        self.assertLessEqual(float(np.abs(residual).max()), cap)

    def test_overflow_prone_equation_emits_no_runtime_warning(self):
        """把告警升级成异常：溢出的 free-exponent 方程不得再刷 RuntimeWarning。

        修复前用同一数据集实测：seed=4/5/9 均触发
        "overflow encountered in power"（数据来自 20260918-195057 实验）。
        """
        dataset = mrf_like_dataset()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for seed in range(10):
                try:
                    eop.evaluate(dataset, free_exponent_equation, seed=seed)
                except RuntimeWarning as exc:  # pragma: no cover - 修复后不再触发
                    self.fail(f'seed={seed} 触发 RuntimeWarning: {exc}')
                except Exception:
                    pass  # 起点全非有限等真实原因由其它用例覆盖

    def test_huge_residual_multi_start_stays_quiet_and_finite(self):
        """1e150 量级的残差（exp 型方程半溢出区间）曾经刷 876 条告警。"""
        X = np.linspace(1.0, 5.0, 8)
        y = np.linspace(190.0, 350.0, 8)
        cap = eop.residual_cap(y)

        def capped_residual(params):
            huge = 1e150 * (params[0] + 0.5 * params[1]) + X * params[2] - y
            return eop._sanitize_residual(huge, cap)

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            best_x, best_loss = eop._multi_start_least_squares(
                capped_residual, eop.MAX_NPARAMS, n_starts=2, seed=0)
        self.assertIsNotNone(best_x)
        self.assertTrue(np.isfinite(best_loss))


if __name__ == '__main__':
    unittest.main()
