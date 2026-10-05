"""evaluation/sandbox.py（LocalSandbox 沙箱）单元测试：常驻 worker、超时重建、统一结果契约。"""
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from drsr_420.core import code_manipulation
from drsr_420.core import config
from drsr_420.core import buffer
from drsr_420.agents import evaluator_agent
from drsr_420.execution import sandbox as sandbox_module
from drsr_420.execution.sandbox import LocalSandbox, _run_evaluation_task, _sample_residuals
from drsr_420.agents.messages import EvaluationRequest

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
    """直接调用 worker 逻辑（不经过进程），覆盖统一 6 元组契约。"""

    def test_success_returns_unified_tuple(self):
        dataset = make_inputs()['data']
        grade, res, runs_ok, remark, params, fit_mse = _run_evaluation_task(
            PROGRAM, 'run', 'equation', dataset, False, {}, None)
        self.assertTrue(runs_ok)
        self.assertEqual(remark, 'yes')
        self.assertIsInstance(grade, float)
        self.assertLess(grade, 0.0)
        self.assertEqual(res.shape, (100, 4))
        self.assertIsInstance(params, np.ndarray)
        # fit_mse = 原始拟合 MSE（**完整**残差矩阵算得，主进程收到的 res 只是它的
        # 随机子样，故不能拿 res 复算）。与评分的关系是 score = −(fit_mse + 罚分)：
        # 本夹具无病理器件 → 罚分 0 → 两者严格相等（有罚分时 grade 会更低）。
        self.assertIsInstance(fit_mse, float)
        self.assertGreaterEqual(fit_mse, 0.0)
        self.assertAlmostEqual(grade, -fit_mse, places=10)

    def test_failure_returns_informative_error(self):
        """NaN 方程 → remark 携带真实原因（'Execution Error: ...'），
        不再是无信息量的 'no output'（第 8 轮修复，喂给经验回路）。"""
        dataset = make_inputs()['data']
        grade, res, runs_ok, remark, params, fit_mse = _run_evaluation_task(
            NAN, 'run', 'equation', dataset, False, {}, None)
        self.assertEqual((grade, res, runs_ok, params, fit_mse), (None, None, False, None, None))
        self.assertIn('Execution Error', remark)
        self.assertIn('not finite', remark)

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
        # 原始拟合 MSE 由 sandbox 缓存（profile 用它写 mse/nmse 字段，与体检罚分分开）
        self.assertIsInstance(sb._last_fit_mse, float)
        self.assertAlmostEqual(grade, -sb._last_fit_mse, places=10)

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

    def test_nan_program_returns_informative_error(self):
        sb = LocalSandbox(numba_accelerate=False)
        try:
            inputs = make_inputs()
            results, res = sb.run(NAN, 'run', 'equation', inputs, 'data', 30)
            self.assertFalse(results[1])
            # 第 8 轮修复：evaluate 不再吞起点异常，remark 携带真实原因
            self.assertIn('Execution Error', results[2])
            self.assertIn('not finite', results[2])
            self.assertIsNone(res)
        finally:
            sb.close()


class WorkerStderrCaptureTest(unittest.TestCase):
    """第 9 轮：评估 worker 的 stderr 必须能落进实验 run.err。

    背景：worker 是 multiprocessing 子进程，只继承到控制台 fd，看不到主进程
    sys.stderr 的 run.err tee——20260918-195057 那场实验里成片的
    scipy RuntimeWarning 因此"只闪在 IDE 控制台，run.err 里查无此物"。
    """

    NOISY = (
        "import sys\n"
        "import numpy as np\n"
        "_warned = []\n"
        "def equation(x1, x2, params):\n"
        "    if not _warned:\n"
        "        _warned.append(1)\n"
        "        sys.stderr.write('worker-noise\\n')\n"
        "    return params[0] * x1 + params[1] * x2 + params[2]\n"
    )

    def test_noop_without_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(sandbox_module.WORKER_LOG_ENV, None)
            before = sys.stderr
            sandbox_module.attach_worker_stderr()
            self.assertIs(sys.stderr, before)

    def test_noop_with_unwritable_root(self):
        with mock.patch.dict(os.environ, {sandbox_module.WORKER_LOG_ENV: '\x00:/nowhere'}):
            before = sys.stderr
            sandbox_module.attach_worker_stderr()   # 打开失败必须静默降级
            self.assertIs(sys.stderr, before)

    def test_attaches_run_err_when_env_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {sandbox_module.WORKER_LOG_ENV: tmp}):
                before = sys.stderr
                sys.stderr = io.StringIO()   # 控制台侧换成内存缓冲，测试输出保持干净
                try:
                    sandbox_module.attach_worker_stderr()
                    sys.stderr.write('captured-line\n')
                    sys.stderr.flush()
                    tee = sys.stderr
                finally:
                    sys.stderr = before
                tee.close()
            with open(os.path.join(tmp, 'run.err'), encoding='utf-8') as fp:
                self.assertIn('captured-line', fp.read())

    def test_cli_tee_exports_results_root_for_workers(self):
        """CLI 侧的最后一环：setup_output_tee 必须把实验目录交给 worker（环境变量）。"""
        from drsr_420.cli import main as cli_main

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(sandbox_module.WORKER_LOG_ENV, None)
                stdout, stderr = sys.stdout, sys.stderr
                try:
                    out_fp, err_fp = cli_main.setup_output_tee(tmp)
                finally:
                    # setup_output_tee 会替换进程级 sys.stdout/stderr，这里只验证环境
                    # 变量传递，立刻还原并把文件句柄交回。
                    sys.stdout, sys.stderr = stdout, stderr
                try:
                    self.assertEqual(os.environ.get(sandbox_module.WORKER_LOG_ENV),
                                     os.path.abspath(tmp))
                finally:
                    out_fp.close()
                    err_fp.close()

    def test_cli_tee_refuses_a_non_empty_experiment_dir(self):
        """同一 ``--experiment_dir`` 被两跑写入时必须**立即失败**，而不是静默交织。

        实测 2026-09-28：并行启动多个 run 时 ``Get-Date -Format 'yyyyMMdd-HHmmss'`` 在
        同一秒取到同一时间戳，5 个 run 落进 2 个目录；而 run.out/run.err 是 append 打开
        的，两跑不会报错，只会把两条轨迹混成一份"看似存在、实际不可用"的产物。
        """
        from drsr_420.cli import main as cli_main

        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "run.out"), "w", encoding="utf-8") as fp:
                fp.write("上一跑的遗留输出\n")
            with self.assertRaises(SystemExit) as ctx:
                cli_main.setup_output_tee(tmp)
            self.assertIn("拒绝复用非空实验目录", str(ctx.exception))

            # 空目录仍然放行（本仓无 --resume，每个 run 的产物应是该目录唯一一条轨迹）
            empty = os.path.join(tmp, "fresh")
            os.makedirs(empty)
            stdout, stderr = sys.stdout, sys.stderr
            try:
                out_fp, err_fp = cli_main.setup_output_tee(empty)
            finally:
                sys.stdout, sys.stderr = stdout, stderr
            out_fp.close()
            err_fp.close()

    def test_worker_process_writes_into_run_err(self):
        """端到端：worker 里方程向 stderr 写的内容出现在实验 run.err。"""
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {sandbox_module.WORKER_LOG_ENV: tmp}):
                sb = LocalSandbox(numba_accelerate=False)
                try:
                    results, _ = sb.run(self.NOISY, 'run', 'equation',
                                        make_inputs(), 'data', 30)
                    self.assertTrue(results[1])
                finally:
                    sb.close()
            with open(os.path.join(tmp, 'run.err'), encoding='utf-8') as fp:
                self.assertIn('worker-noise', fp.read())


TEMPLATE_TEXT = """\
import numpy as np

def equation(x1, x2, params):
    return params[0] * x1 + params[1] * x2 + params[2]
"""

SAMPLE_BODY = "    return params[0] * x1 + params[1] * x2 + params[2]\n"


class EvaluatorAnalyzeTest(unittest.TestCase):
    """Evaluator.analyze 端到端：模板编译 → 沙箱评估 → 经验缓冲注册。"""

    def _make_evaluator(self):
        template = code_manipulation.text_to_program(TEMPLATE_TEXT)
        db = buffer.ExperienceBuffer(
            config.ExperienceBufferConfig(num_islands=2),
            template,
            'equation',
        )
        return evaluator_agent.EvaluatorAgent(
            db, template, 'equation', 'run', make_inputs(),
            timeout_seconds=30, sandbox_class=LocalSandbox)

    def test_analyze_returns_evaluation_outcome(self):
        ev = self._make_evaluator()
        outcome = ev.analyze(EvaluationRequest(
            sample=SAMPLE_BODY, island_id=0, version_generated=None))
        self.assertIsInstance(outcome.score, float)
        self.assertLess(outcome.score, 0.0)
        self.assertEqual(outcome.error, 'yes')
        self.assertEqual(outcome.residual.shape, (100, 4))

    def test_deprecated_analyse_returns_legacy_tuple(self):
        """兼容入口：旧签名 analyse(...) 仍可用，并返回旧的裸元组形式。"""
        ev = self._make_evaluator()
        score, error_msg, res = ev.analyse(
            SAMPLE_BODY, island_id=0, version_generated=None)
        self.assertIsInstance(score, float)
        self.assertLess(score, 0.0)
        self.assertEqual(error_msg, 'yes')
        self.assertEqual(res.shape, (100, 4))

    def test_analyse_rejects_misspelled_kwarg(self):
        """旧实现用 **kwargs 收参数，拼错参数名会静默丢 profiler 记录；现在直接报错。"""
        ev = self._make_evaluator()
        with self.assertRaises(TypeError):
            ev.analyse(SAMPLE_BODY, island_id=0, version_generated=None,
                       profile=None)   # 应为 profiler


if __name__ == '__main__':
    unittest.main()
