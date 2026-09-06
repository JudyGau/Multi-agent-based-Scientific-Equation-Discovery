"""evaluate_on_problems 单元测试：least_squares 优化、统一返回契约、配置项。"""
import unittest

import numpy as np

from drsr_420 import evaluate_on_problems as eop


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
        def nan_equation(x1, x2, params):
            return np.full_like(x1, np.nan)

        self.assertEqual(eop.evaluate(make_dataset(), nan_equation), (None, None, None))

    def test_failure_contract_for_exception_equation(self):
        def bad_equation(x1, x2, params):
            return params[999] * x1  # IndexError：所有起点失败

        self.assertEqual(eop.evaluate(make_dataset(), bad_equation), (None, None, None))

    def test_warm_start_x0(self):
        data = make_dataset()
        _, _, params = eop.evaluate(data, linear_equation, seed=1)
        score2, _, _ = eop.evaluate(data, linear_equation, x0=params, seed=1)
        self.assertLess(score2, 0.0)

    def test_decimal_places_respected(self):
        _, matrix, _ = eop.evaluate(make_dataset(), linear_equation, decimal_places=3)
        # 取整后矩阵应为 0.001 的倍数
        np.testing.assert_allclose(matrix / 0.001, np.round(matrix / 0.001))


class HelperTest(unittest.TestCase):
    def test_clamp_params(self):
        out = eop._clamp_params(np.array([-20.0, 5.0, 20.0]), (-10.0, 10.0))
        np.testing.assert_array_equal(out, [-10.0, 5.0, 10.0])

    def test_multi_start_all_fail_returns_none(self):
        def always_raise(params):
            raise ValueError('boom')

        best_x, best_loss = eop._multi_start_least_squares(always_raise, 3, n_starts=2)
        self.assertIsNone(best_x)
        self.assertEqual(best_loss, np.inf)


if __name__ == '__main__':
    unittest.main()
