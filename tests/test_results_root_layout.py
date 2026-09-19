"""结果目录布局：``experiments/<问题名>/<问题名>_<时间戳>/``。

背景
----
同一种问题的实验原本平铺在 ``experiments/`` 下（``experiments/MRFCompress-Cuboid_20260918-195057/``），
跑几十上百次之后既难浏览也没有"同题成组"的结构。现在按问题名分组：

::

    experiments/
      MRFCompress-Cuboid/
        MRFCompress-Cuboid_20260918-195057/
          run.out / checkpoint.json / samples/ …

本文件锁死三条契约：默认路径带问题名子目录（并**真的把两级目录建出来**）、
``--experiment_dir`` 仍然完全接管（想放哪就放哪，不受分组规则影响）、时间戳格式不变。
另外覆盖手工排查用的 ``find_best_eq._latest_run_dir()``（新旧两种布局都能取到最近一次 run）。
"""
from __future__ import annotations

import os
import pathlib
import re
import tempfile
import unittest
from unittest import mock

from drsr_420.cli.main import resolve_results_root

_TS_RE = re.compile(r'^\d{8}-\d{6}$')


class ResultsRootLayoutTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)

    def _resolve(self, problem, experiment_dir=None):
        with mock.patch("os.getcwd", return_value=str(self.root)):
            return resolve_results_root(problem, experiment_dir)

    def test_default_is_grouped_under_problem_name(self):
        out = pathlib.Path(self._resolve("MRFCompress-Cuboid"))
        self.assertEqual(out.parent.name, "MRFCompress-Cuboid")
        self.assertEqual(out.parent.parent.name, "experiments")
        self.assertTrue(out.name.startswith("MRFCompress-Cuboid_"))
        self.assertTrue(_TS_RE.fullmatch(out.name[len("MRFCompress-Cuboid_"):]),
                        f"时间戳格式应为 YYYYMMDD-HHMMSS：{out.name}")

    def test_both_levels_are_created_on_disk(self):
        out = pathlib.Path(self._resolve("MRFCompress-Cuboid"))
        self.assertTrue(out.is_dir(), "实验目录必须真的被创建")
        self.assertTrue(out.parent.is_dir(), "问题名子目录必须连同创建")

    def test_same_problem_runs_share_the_group_directory(self):
        first = pathlib.Path(self._resolve("BPG0"))
        second = pathlib.Path(self._resolve("BPG0"))
        self.assertEqual(first.parent, second.parent)
        self.assertTrue(first.is_dir() and second.is_dir())

    def test_different_problems_are_separated(self):
        a = pathlib.Path(self._resolve("BPG0"))
        b = pathlib.Path(self._resolve("MRFShear-2"))
        self.assertNotEqual(a.parent, b.parent)

    def test_experiment_dir_still_wins_verbatim(self):
        """显式指定目录时不套用分组规则（也不要求目录名与问题名一致）。"""
        explicit = self.root / "tmp" / "my-run"
        out = self._resolve("MRFCompress-Cuboid", str(explicit))
        self.assertEqual(pathlib.Path(out), explicit)
        self.assertTrue(explicit.is_dir())


class LatestRunDirTest(unittest.TestCase):
    """手工排查入口的"最近一次 run"探测：两种布局都能认出来。"""

    def _touch_run(self, path: pathlib.Path, mtime: float):
        path.mkdir(parents=True, exist_ok=True)
        (path / "run.out").write_text("x", encoding="utf-8")
        os.utime(path, (mtime, mtime))
        return path

    def test_prefers_newest_run_in_nested_layout(self):
        from drsr_420.analysis.find_best_eq import _latest_run_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            old = self._touch_run(root / "experiments" / "BPG0" / "BPG0_20250101-000000", 1000.0)
            new = self._touch_run(root / "experiments" / "BPG0" / "BPG0_20250102-000000", 2000.0)
            self.assertEqual(_latest_run_dir(str(root / "experiments")), str(new))
            self.assertNotEqual(_latest_run_dir(str(root / "experiments")), str(old))

    def test_ignores_directories_without_run_artifacts(self):
        from drsr_420.analysis.find_best_eq import _latest_run_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            # samples/ 这类子目录不是实验根目录：没有 run.out / checkpoint.json 就不算
            (root / "experiments" / "BPG0" / "BPG0_20250102-000000" / "samples").mkdir(parents=True)
            self.assertIsNone(_latest_run_dir(str(root / "experiments")))

    def test_accepts_legacy_flat_layout(self):
        from drsr_420.analysis.find_best_eq import _latest_run_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            legacy = self._touch_run(root / "experiments" / "BPG0_20250101-000000", 1000.0)
            self.assertEqual(_latest_run_dir(str(root / "experiments")), str(legacy))

    def test_returns_none_for_empty_tree(self):
        from drsr_420.analysis.find_best_eq import _latest_run_dir

        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_latest_run_dir(tmp))


if __name__ == "__main__":
    unittest.main()
