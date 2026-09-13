"""Windows GBK 控制台兼容性：凡是 print 出去的字符串，都必须能被 GBK 编码。

背景
----
项目里已三次踩到同一个坑：把 ``R²`` / ``▶`` / ``✂`` / emoji 之类字符写进面向
控制台的输出，在 Windows 默认代码页（cp936/GBK）下直接抛 ``UnicodeEncodeError``
**打断流程**——最典型的是 ``python -m drsr_420.analysis.prune_demo`` 与剪枝的
verbose 日志（两者都是"看起来只是打印"却让整个函数失败）。

这里用 ``PYTHONIOENCODING=gbk`` 起子进程跑几个典型入口：输出编码一旦不兼容，
子进程会以非 0 退出并在 stderr 留下 UnicodeEncodeError，测试即失败。
注意只覆盖**运行时打印**的字符串；注释与文档字符串不参与，无需限制字符集。
"""
import os
import subprocess
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 需要面向 Windows GBK 控制台的典型入口（argv 里第一个元素是展示名）。
_ENTRY_POINTS = (
    ("agents 组织图", ["-m", "drsr_420.agents"]),
    ("agents 契约自检", ["-m", "drsr_420.agents", "--check"]),
    ("剪枝演示", ["-m", "drsr_420.analysis.prune_demo"]),
    ("剪枝 verbose 日志", ["-m", "drsr_420.analysis.sensitivity_prune"]),
    ("CLI --help", ["-m", "drsr_420.cli.main", "--help"]),
)

#: 直接调用 API（verbose 剪枝会走 _log 打印路径）的探针脚本。
_VERBOSE_PROBE = (
    "import sympy as sp;"
    "from drsr_420.analysis.sensitivity_prune import SensitivityPruner;"
    "x, y = sp.symbols('x y');"
    "p = SensitivityPruner([x, y], threshold=0.5, seed=42);"
    "p.prune(x**2 + y**2 + sp.Rational(1, 1000)*x*y, verbose=True);"
    "print('PROBE-OK')"
)


class GbkConsoleTest(unittest.TestCase):
    """面向 GBK 控制台的输出不得含不可编码字符。"""

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        env = dict(os.environ, PYTHONIOENCODING="gbk", ZHIPU_API_KEY="dummy-test-key")
        return subprocess.run(
            [sys.executable, *args], cwd=_REPO_ROOT, env=env,
            capture_output=True, text=False, timeout=180,
        )

    def test_entry_points_print_gbk_safe_output(self):
        for label, args in _ENTRY_POINTS:
            with self.subTest(entry=label):
                proc = self._run(args)
                stderr = proc.stderr.decode("gbk", "replace")
                self.assertEqual(
                    proc.returncode, 0,
                    f"{label} 在 GBK 控制台下失败：\n{stderr[-1500:]}")
                self.assertNotIn("UnicodeEncodeError", stderr,
                                 f"{label} 打印了 GBK 编码不了的字符：\n{stderr[-1500:]}")

    def test_verbose_pruning_log_is_gbk_safe(self):
        proc = self._run(["-c", _VERBOSE_PROBE])
        stdout = proc.stdout.decode("gbk", "replace")
        stderr = proc.stderr.decode("gbk", "replace")
        self.assertEqual(proc.returncode, 0,
                         f"verbose 剪枝在 GBK 控制台下失败：\n{stderr[-1500:]}")
        self.assertIn("PROBE-OK", stdout)


if __name__ == "__main__":
    unittest.main()
