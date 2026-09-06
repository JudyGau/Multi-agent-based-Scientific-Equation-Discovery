"""evaluator.py（LocalSandbox 沙箱）单元测试：常驻 worker、超时重建、统一结果契约。"""
import unittest

import numpy as np

from drsr_420 import code_manipulation
from drsr_420 import config
from drsr_420 import buffer
from drsr_420 import evaluator
from drsr_420.evaluator import LocalSandbox, _run_evaluation_task, _sample_residuals

PROGRAM = (
    "import numpy as np\n"
    "def equation(x1, x2, params):\n"
    "    return params[0] * x1 + params[1] * x2 + params[2]\n"
)
HANG = (
    "import numpy as np\n"
    "def equation(x1, x2, params):\n"
    "    while True:\n"
    "        pass\n"
)
NAN = (
    "import numpy as np\n"
    "def equation(x1, x2, params):\n"
    "    return np.full_like(x1, np.nan)\n"
)


def make_inputs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 + 0.01 * rng.standard_normal(n)
    return {'data': {'inputs': X, 'outputs': y}}


class SampleResidualsTest(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(_sample_residuals(None, 100))

    def test_empty_returns_none(self):
        self.assertIsNone(_sample_residuals(np.empty((0, 4)), 100))

    def test_samples_without_replacement(self):
        full = np.arange(300).reshape(100, 3)
        out = _sample_residuals(full, 30)
        self.assertEqual(out.shape, (30, 3))
        self.assertEqual(len(np.unique(out[:, 0])), 30)  # 无放回采样


class RunEvaluationTaskTest(unittest.TestCase):
    """直接调用 worker 逻辑（不经过进程），覆盖统一 5 元组契约。"""

    def test_success_returns_unified_tuple(self):
        dataset = make_inputs()['data']
        grade, res, runs_ok, remark, params = _run_evaluation_task(
            PROGRAM, 'run', 'equation', dataset, False, {}, None)
        self.assertTrue(runs_ok)
        self.assertEqual(remark, 'yes')
        self.assertIsInstance(grade, float)
        self.assertLess(grade, 0.0)
        self.assertEqual(res.shape, (100, 4))
        self.assertIsInstance(params, np.ndarray)

    def test_failure_returns_no_output(self):
        dataset = make_inputs()['data']
        self.assertEqual(
            _run_evaluation_task(NAN, 'run', 'equation', dataset, False, {}, None),
            (None, None, False, 'no output', None))

    def test_program_missing_function_returns_error(self):
        dataset = make_inputs()['data']
        no_equation = (
            "import numpy as np\n"
            "def other(x1, x2, params):\n"
            "    return x1\n"
        )
        out = _run_evaluation_task(no_equation, 'run', 'equation', dataset, False, {}, None)
        self.assertFalse(out[2])
        self.assertIn('Execution Error', out[3])


class LocalSandboxTest(unittest.TestCase):
    """通过常驻 worker 进程的端到端测试。"""

    def test_run_success_and_warm_start(self):
        sb = LocalSandbox(numba_accelerate=False)
        inputs = make_inputs()
        results, res = sb.run(PROGRAM, 'run', 'equation', inputs, 'data', 30)
        grade, runs_ok, remark = results
        self.assertTrue(runs_ok)
        self.assertEqual(remark, 'yes')
        self.assertLess(grade, 0.0)
        self.assertEqual(res.shape, (100, 4))
        self.assertIsNotNone(sb._last_params)  # 为下一轮热启动保留参数

        # 热启动：复用上一轮参数，仍应正常评估
        results2, _ = sb.run(PROGRAM, 'run', 'equation', inputs, 'data', 30)
        self.assertTrue(results2[1])

    def test_timeout_respawns_worker(self):
        sb = LocalSandbox(numba_accelerate=False)
        inputs = make_inputs()
        results, res = sb.run(HANG, 'run', 'equation', inputs, 'data', 1)
        grade, runs_ok, remark = results
        self.assertFalse(runs_ok)
        self.assertEqual(remark, 'timeout01')
        self.assertIsNone(res)

        # worker 已被销毁重建，仍可继续正常评估
        results2, _ = sb.run(PROGRAM, 'run', 'equation', inputs, 'data', 30)
        self.assertTrue(results2[1])

    def test_nan_program_returns_no_output(self):
        sb = LocalSandbox(numba_accelerate=False)
        inputs = make_inputs()
        results, res = sb.run(NAN, 'run', 'equation', inputs, 'data', 30)
        self.assertFalse(results[1])
        self.assertEqual(results[2], 'no output')
        self.assertIsNone(res)


TEMPLATE_TEXT = """\
import numpy as np

def equation(x1, x2, params):
    return params[0] * x1 + params[1] * x2 + params[2]
"""

SAMPLE_BODY = "    return params[0] * x1 + params[1] * x2 + params[2]\n"


class EvaluatorAnalyseTest(unittest.TestCase):
    """Evaluator.analyse 端到端：模板编译 → 沙箱评估 → 经验缓冲注册。"""

    def test_analyse_returns_score_error_and_residual(self):
        template = code_manipulation.text_to_program(TEMPLATE_TEXT)
        db = buffer.ExperienceBuffer(
            config.ExperienceBufferConfig(num_islands=2),
            template,
            'equation',
        )
        ev = evaluator.Evaluator(
            db, template, 'equation', 'run', make_inputs(),
            timeout_seconds=30, sandbox_class=LocalSandbox)
        score, error_msg, res = ev.analyse(
            SAMPLE_BODY, island_id=0, version_generated=None)
        self.assertIsInstance(score, float)
        self.assertLess(score, 0.0)
        self.assertEqual(error_msg, 'yes')
        self.assertEqual(res.shape, (100, 4))


if __name__ == '__main__':
    unittest.main()
