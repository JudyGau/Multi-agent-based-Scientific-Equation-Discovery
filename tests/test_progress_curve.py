"""训练进度曲线（MSE 随 sample_order）与 report.md 机器小节的回归测试。

守四件事：
1. 数据**只**来自 ``best_history/*.json``（坏文件 / 非有限 mse 只跳过该文件，不丢整条曲线）；
2. 没有记录时不生成图、不给小节——报告里不能出现"本次无数据"这类非实验结果；
3. 小节排在「动态范围体检」之后、「参考文献」之前，且回填幂等（重复回填不出现两节）；
4. 阶梯/刷新点标记/对数轴的画法契约（对数轴只在全部 MSE > 0 时使用）。
"""
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from drsr_420.reporting import progress_curve as pcur
from drsr_420.reporting.report_sections import ReportData


def _matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False


def _write_best(root, order, mse, nmse=None, iteration=1, penalty=None, score=None):
    """在 ``best_history/`` 下写一个刷新点文件（与评估器的字段一致）。

    ``penalty`` 不给就不写该键——那正是口径拆分之前的历史目录形态（mse 内含罚分）。
    """
    d = pathlib.Path(root) / "best_history"
    d.mkdir(parents=True, exist_ok=True)
    rec = {"sample_order": order, "iteration": iteration, "mse": mse}
    if nmse is not None:
        rec["nmse"] = nmse
    if penalty is not None:
        rec["penalty"] = penalty
        if score is None:
            score = -(mse + penalty)
    if score is not None:
        rec["score"] = score
    (d / f"best_sample_{order}.json").write_text(json.dumps(rec), encoding="utf-8")


def _patched_subplots(axes):
    """把 ``plt.subplots`` 换成"每次调用新建一个假 Axes"，按调用顺序收集到 ``axes``。

    本模块按「刷新点 MSE → 刷新点罚分 → 刷新点评分 → 逐样本 MSE → 逐样本罚分 →
    逐样本评分」的顺序各画一幅图（缺数据的跳过），所以断言要按图取 Axes，而不是把
    所有调用记在同一个 Axes 上。
    """
    def _fake(*args, **kwargs):
        ax = _FakeAx()
        axes.append(ax)
        return _FakeFig(), ax
    return _fake


def _write_sample(root, order, mse, penalty=None, score=None, nmse=None,
                  function="def equation_v1(x, y, params):\n    return params[0]"):
    """在 ``samples/`` 下写一个逐样本记录（与 Profiler 落盘的字段一致）。"""
    d = pathlib.Path(root) / "samples"
    d.mkdir(parents=True, exist_ok=True)
    rec = {"sample_order": order, "iteration": 1, "mse": mse, "function": function,
           "params": [1.0]}
    if nmse is not None:
        rec["nmse"] = nmse
    if penalty is not None:
        rec["penalty"] = penalty
        if score is None:
            score = -(mse + penalty)
    if score is None:
        score = -mse
    rec["score"] = score
    (d / f"samples_{order}.json").write_text(json.dumps(rec), encoding="utf-8")


class _FakeAx:
    """记录调用参数的假 Axes（不真渲染）。"""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return _record

    def _named(self, name):
        return [c for c in self.calls if c[0] == name]


class _FakeFig:
    def savefig(self, *args, **kwargs):
        pass

    def tight_layout(self, *args, **kwargs):
        pass


class LoadBestHistoryTest(unittest.TestCase):
    def test_sorted_by_sample_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 83, 1.793076, nmse=8.9e-4, iteration=21)
            _write_best(tmp, 5, 18.846566, nmse=0.0076, iteration=2)
            rows = pcur.load_best_history(tmp)
        self.assertEqual([r["sample_order"] for r in rows], [5, 83])
        self.assertEqual(rows[0]["mse"], 18.846566)
        self.assertEqual(rows[1]["iteration"], 21)

    def test_score_is_derived_when_the_field_is_missing(self):
        """记录缺 ``score`` 时按同一口径推出来：有 penalty → −(mse+penalty)。"""
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp) / "best_history"
            d.mkdir(parents=True)
            (d / "best_sample_3.json").write_text(
                json.dumps({"sample_order": 3, "mse": 2.0, "penalty": 3.0}), encoding="utf-8")
            rows = pcur.load_best_history(tmp)
        self.assertEqual(rows[0]["score"], -5.0)

    def test_legacy_record_derives_score_as_minus_mse(self):
        """旧目录的 ``mse`` 本就内含罚分，故 score = −mse。"""
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 4, 7.0)
            rows = pcur.load_best_history(tmp)
        self.assertEqual(rows[0]["score"], -7.0)
        self.assertIsNone(rows[0]["penalty"])

    def test_score_is_consistent_with_mse_and_penalty_at_every_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 0, 102.5632, penalty=1.5)
            _write_best(tmp, 5, 0.2438, penalty=11.7553)
            rows = pcur.load_best_history(tmp)
        for row in rows:
            with self.subTest(order=row["sample_order"]):
                self.assertAlmostEqual(row["score"], -(row["mse"] + row["penalty"]), places=12)

    def test_broken_and_non_finite_records_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 4, 55.2658)
            d = pathlib.Path(tmp) / "best_history"
            (d / "best_sample_7.json").write_text("{不是 json", encoding="utf-8")
            (d / "best_sample_9.json").write_text(
                json.dumps({"sample_order": 9, "mse": None}), encoding="utf-8")
            (d / "best_sample_11.json").write_text(
                json.dumps({"sample_order": 11, "mse": float("nan")}), encoding="utf-8")
            with mock.patch("builtins.print"):
                rows = pcur.load_best_history(tmp)
        self.assertEqual([r["sample_order"] for r in rows], [4],
                         "坏文件只跳过它自己，好记录要留下")

    def test_non_finite_penalty_is_treated_as_missing(self):
        """penalty=inf 不能进曲线（会让罚分图的反向半径爆掉），按缺失处理。"""
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp) / "best_history"
            d.mkdir(parents=True)
            (d / "best_sample_2.json").write_text(
                json.dumps({"sample_order": 2, "mse": 1.0,
                            "penalty": float("inf"), "score": -1.0}), encoding="utf-8")
            rows = pcur.load_best_history(tmp)
        self.assertIsNone(rows[0]["penalty"])

    def test_missing_directory_yields_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(pcur.load_best_history(tmp), [])


class PlotProgressCurveTest(unittest.TestCase):
    def test_no_history_returns_none_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("builtins.print"):
                summary = pcur.plot_progress_curve(tmp)
            self.assertIsNone(summary)
            for name in (pcur.PROGRESS_PNG_NAME, pcur.PENALTY_PNG_NAME, pcur.SCORE_PNG_NAME):
                with self.subTest(png=name):
                    self.assertFalse((pathlib.Path(tmp) / name).exists(),
                                     "没有记录时不能留下空图（否则报告会引用一张空图）")

    def test_three_curves_share_the_same_refresh_points(self):
        """三条曲线的横轴必须逐点对齐（同一批刷新点），否则对照会错位。"""
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 0, 102.5632, penalty=0.0)
            _write_best(tmp, 5, 18.8466, penalty=2.0)
            _write_best(tmp, 7, 4.2504, penalty=0.5)
            axes = []
            import matplotlib.pyplot as plt
            with mock.patch.object(plt, "subplots", _patched_subplots(axes)), \
                 mock.patch.object(plt, "close"), \
                 mock.patch("builtins.print"):
                summary = pcur.plot_progress_curve(tmp)
            self.assertEqual(len(axes), 3, "MSE / 罚分 / 评分 各一幅图")
            for label, ax in zip(("mse", "penalty", "score"), axes):
                step = ax._named("step")
                self.assertEqual(len(step), 1, f"{label} 必须用阶梯线画")
                self.assertEqual(step[0][2].get("where"), "post")
                self.assertTrue(
                    np.allclose(np.asarray(step[0][1][0], dtype=float), [0, 5, 7]),
                    f"{label} 的横轴必须落在同一批刷新点上")
                self.assertEqual(len(ax._named("plot")), 1, "刷新点要有数据点标记")
                self.assertEqual(len(ax._named("legend")), 1)
            self.assertTrue(np.allclose(
                np.asarray(axes[0]._named("step")[0][1][1], dtype=float),
                [102.5632, 18.8466, 4.2504]))
            self.assertTrue(np.allclose(
                np.asarray(axes[1]._named("step")[0][1][1], dtype=float), [0.0, 2.0, 0.5]))
            # 评分 = −(拟合 MSE + 罚分)，逐点成立
            self.assertTrue(np.allclose(
                np.asarray(axes[2]._named("step")[0][1][1], dtype=float),
                [-102.5632, -20.8466, -4.7504]))
        self.assertEqual(summary["n_points"], 3)
        self.assertEqual(summary["first"]["sample_order"], 0)
        self.assertEqual(summary["best"]["sample_order"], 7, "best 取 MSE 最小的那个刷新点")
        self.assertIsNotNone(summary["penalty"])
        self.assertIsNotNone(summary["score"])

    def test_positive_penalty_uses_a_log_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 0, 102.5632, penalty=1.0)
            _write_best(tmp, 5, 4.2504, penalty=1.0e5)
            axes = []
            import matplotlib.pyplot as plt
            with mock.patch.object(plt, "subplots", _patched_subplots(axes)), \
                 mock.patch.object(plt, "close"), \
                 mock.patch("builtins.print"):
                pcur.plot_progress_curve(tmp)
            self.assertEqual(axes[0]._named("set_yscale")[0][1], ("log",))
            self.assertEqual(axes[1]._named("set_yscale")[0][1], ("log",))

    def test_zero_penalty_falls_back_to_a_linear_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 0, 102.5632, penalty=0.0)
            _write_best(tmp, 5, 4.2504, penalty=2.0)
            axes = []
            import matplotlib.pyplot as plt
            with mock.patch.object(plt, "subplots", _patched_subplots(axes)), \
                 mock.patch.object(plt, "close"), \
                 mock.patch("builtins.print"):
                summary = pcur.plot_progress_curve(tmp)
            self.assertEqual(axes[1]._named("set_yscale"), [], "罚分有 0 值时不能取对数刻度")
        self.assertFalse(summary["penalty"]["log_scale"])

    def test_score_curve_is_always_symlog(self):
        """评分 ≤ 0 且跨数量级：线性轴会把后段改善压平，对数轴表示不了负数。"""
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 0, 102.5632, penalty=0.0)
            _write_best(tmp, 42, 0.2716, penalty=0.0)
            axes = []
            import matplotlib.pyplot as plt
            with mock.patch.object(plt, "subplots", _patched_subplots(axes)), \
                 mock.patch.object(plt, "close"), \
                 mock.patch("builtins.print"):
                summary = pcur.plot_progress_curve(tmp)
            call = axes[2]._named("set_yscale")[0]
            self.assertEqual(call[1], ("symlog",))
            self.assertEqual(call[2]["linthresh"], pcur.SCORE_SYMLOG_LINTHRESH)
        self.assertEqual(summary["score"]["path"].endswith(pcur.SCORE_PNG_NAME), True)

    def test_legacy_directory_skips_the_penalty_curve(self):
        """旧目录没有 penalty 字段 → 不画罚分曲线（也不落盘），MSE 与评分照画。"""
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 0, 102.5632)
            _write_best(tmp, 40, 11.999038)
            axes = []
            import matplotlib.pyplot as plt
            with mock.patch.object(plt, "subplots", _patched_subplots(axes)), \
                 mock.patch.object(plt, "close"), \
                 mock.patch("builtins.print"):
                summary = pcur.plot_progress_curve(tmp)
            self.assertEqual(len(axes), 2, "只有 MSE 与评分两幅图")
            self.assertFalse((pathlib.Path(tmp) / pcur.PENALTY_PNG_NAME).exists())
        self.assertTrue(summary["legacy_records"])
        self.assertIsNone(summary["penalty"])
        self.assertIsNotNone(summary["score"])

    def test_non_positive_mse_falls_back_to_linear_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 0, 0.0, penalty=0.0)
            _write_best(tmp, 3, 1.0, penalty=0.0)
            axes = []
            import matplotlib.pyplot as plt
            with mock.patch.object(plt, "subplots", _patched_subplots(axes)), \
                 mock.patch.object(plt, "close"), \
                 mock.patch("builtins.print"):
                summary = pcur.plot_progress_curve(tmp)
            self.assertEqual(axes[0]._named("set_yscale"), [], "有 0 值时不能取对数刻度")
        self.assertFalse(summary["log_scale"])

    @unittest.skipUnless(_matplotlib_available(), "本机没有 matplotlib")
    def test_real_render_writes_three_pngs(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 0, 102.5632, nmse=0.05097, penalty=0.0)
            _write_best(tmp, 42, 1.113129, nmse=5.5e-4, penalty=2.5)
            with mock.patch("builtins.print"):
                summary = pcur.plot_progress_curve(tmp)
            path = pathlib.Path(summary["path"])
            self.assertTrue(path.exists(), "有记录时必须真的落盘")
            self.assertGreater(path.stat().st_size, 5000, "不能是空图")
            self.assertEqual(path.name, pcur.PROGRESS_PNG_NAME)
            for name in (pcur.PENALTY_PNG_NAME, pcur.SCORE_PNG_NAME):
                with self.subTest(png=name):
                    other = pathlib.Path(tmp) / name
                    self.assertTrue(other.exists(), f"{name} 必须落盘")
                    self.assertGreater(other.stat().st_size, 5000)


class LoadSamplePointsTest(unittest.TestCase):
    """逐样本数据源：``samples/*.json``（每个已落盘样本一个点）。"""

    def test_points_are_sorted_by_sample_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_sample(tmp, 7, 1.5, penalty=0.0)
            _write_sample(tmp, 2, 9.0, penalty=1.0)
            rows = pcur.load_sample_points(tmp)
        self.assertEqual([r["sample_order"] for r in rows], [2, 7])
        self.assertEqual(rows[0]["score"], -10.0)
        self.assertEqual(rows[1]["penalty"], 0.0)

    def test_topk_and_full_files_are_deduplicated_by_order(self):
        """旧实验只有 top-K：同一 order 的 top 副本与全量文件不能算两个点。"""
        with tempfile.TemporaryDirectory() as tmp:
            _write_sample(tmp, 3, 2.0, penalty=0.0)
            top = pathlib.Path(tmp) / "samples" / "top01_samples_3.json"
            top.write_text((pathlib.Path(tmp) / "samples" / "samples_3.json").read_text(
                encoding="utf-8"), encoding="utf-8")
            rows = pcur.load_sample_points(tmp)
        self.assertEqual(len(rows), 1)

    def test_non_finite_penalty_is_dropped_and_score_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp) / "samples"
            d.mkdir(parents=True)
            (d / "samples_1.json").write_text(
                json.dumps({"sample_order": 1, "mse": 2.0, "score": -5.0,
                            "penalty": float("inf")}), encoding="utf-8")
            (d / "samples_2.json").write_text(
                json.dumps({"sample_order": 2, "mse": 3.0, "score": -4.0, "penalty": 1.0}),
                encoding="utf-8")
            rows = pcur.load_sample_points(tmp)
        self.assertIsNone(rows[0]["penalty"], "inf 罚分不能进曲线（会让纵轴反向爆掉）")
        self.assertEqual(rows[1]["score"], -4.0)
        self.assertEqual(rows[1]["penalty"], 1.0)

    def test_missing_directory_yields_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(pcur.load_sample_points(tmp), [])


class PerSampleCurveTest(unittest.TestCase):
    """逐样本曲线：每个已落盘样本一个点、**折线**连接（不是阶梯线）。"""

    def _render(self, tmp, with_samples=True):
        _write_best(tmp, 0, 102.5632, penalty=0.0)
        _write_best(tmp, 7, 4.2504, penalty=0.5)
        if with_samples:
            _write_sample(tmp, 0, 102.5632, penalty=0.0)
            _write_sample(tmp, 1, 30.5, penalty=0.0)
            _write_sample(tmp, 2, 0.25, penalty=9.0)
            _write_sample(tmp, 3, 4.2504, penalty=0.5)
        axes = []
        import matplotlib.pyplot as plt
        with mock.patch.object(plt, "subplots", _patched_subplots(axes)), \
             mock.patch.object(plt, "close"), \
             mock.patch("builtins.print"):
            summary = pcur.plot_progress_curve(tmp)
        return axes, summary

    def test_every_persisted_sample_is_a_point_on_a_polyline(self):
        with tempfile.TemporaryDirectory() as tmp:
            axes, summary = self._render(tmp)
            # 前三条 = 刷新点（阶梯），后三条 = 逐样本（折线）
            self.assertEqual(len(axes), 6)
            for label, ax in zip(("mse", "penalty", "score"), axes[3:]):
                with self.subTest(curve=label):
                    self.assertEqual(ax._named("step"), [],
                                     "逐样本曲线不能画成阶梯线（没有「保持」语义）")
                    line = ax._named("plot")
                    self.assertEqual(len(line), 1, "逐样本曲线应是一条折线")
                    self.assertTrue(np.allclose(
                        np.asarray(line[0][1][0], dtype=float), [0, 1, 2, 3]))
            self.assertTrue(np.allclose(
                np.asarray(axes[3]._named("plot")[0][1][1], dtype=float),
                [102.5632, 30.5, 0.25, 4.2504]))
            self.assertTrue(np.allclose(
                np.asarray(axes[5]._named("plot")[0][1][1], dtype=float),
                [-102.5632, -30.5, -9.25, -4.7504]))
        self.assertEqual(summary["per_sample"]["n_points"], 4)
        self.assertEqual(summary["per_sample"]["n_clean"], 2)
        self.assertEqual(summary["per_sample"]["max_penalty"]["sample_order"], 2)
        self.assertEqual(summary["per_sample"]["best_score"]["sample_order"], 3)
        self.assertEqual(summary["per_sample"]["worst_score"]["sample_order"], 0)

    def test_per_sample_curves_are_skipped_without_persisted_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            axes, summary = self._render(tmp, with_samples=False)
            self.assertEqual(len(axes), 3, "没有 samples/ 时只画刷新点三条")
        self.assertIsNone(summary["per_sample"])

    def test_penalty_panel_falls_back_to_linear_when_a_sample_is_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            axes, _ = self._render(tmp)
            self.assertEqual(axes[4]._named("set_yscale"), [],
                             "有干净样本（罚分 0）时罚分面板不能取对数")
            self.assertEqual(axes[5]._named("set_yscale")[0][1], ("symlog",))

    @unittest.skipUnless(_matplotlib_available(), "本机没有 matplotlib")
    def test_real_render_writes_six_pngs(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 0, 102.5632, penalty=0.0)
            _write_best(tmp, 7, 4.2504, penalty=0.5)
            _write_sample(tmp, 0, 102.5632, penalty=0.0)
            _write_sample(tmp, 1, 30.5, penalty=0.0)
            _write_sample(tmp, 2, 0.25, penalty=9.0)
            _write_sample(tmp, 3, 4.2504, penalty=0.5)
            with mock.patch("builtins.print"):
                pcur.plot_progress_curve(tmp)      # 真实渲染（不打桩）
            for name in (pcur.PROGRESS_PNG_NAME, pcur.PENALTY_PNG_NAME, pcur.SCORE_PNG_NAME,
                         pcur.PER_SAMPLE_MSE_PNG_NAME, pcur.PER_SAMPLE_PENALTY_PNG_NAME,
                         pcur.PER_SAMPLE_SCORE_PNG_NAME):
                with self.subTest(png=name):
                    other = pathlib.Path(tmp) / name
                    self.assertTrue(other.exists(), f"{name} 必须落盘")
                    self.assertGreater(other.stat().st_size, 5000)


class SchemaCaliberTest(unittest.TestCase):
    """口径拆分（20260926 之后）：penalty 单列；旧目录据此打标。"""

    def test_record_with_penalty_is_not_marked_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 40, 0.2438, nmse=1.2e-4, penalty=11.7553)
            rows = pcur.load_best_history(tmp)
        self.assertEqual(rows[0]["penalty"], 11.7553)
        self.assertFalse(rows[0]["mse_includes_penalty"], "新口径的 mse 只含拟合")

    def test_record_without_penalty_is_marked_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 40, 11.999038, nmse=5.964e-3)
            rows = pcur.load_best_history(tmp)
        self.assertIsNone(rows[0]["penalty"])
        self.assertTrue(rows[0]["mse_includes_penalty"], "旧目录的 mse 内含罚分")

    def test_plot_marks_legacy_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 0, 102.5632)
            _write_best(tmp, 40, 11.999038)
            axes = []
            import matplotlib.pyplot as plt
            with mock.patch.object(plt, "subplots", _patched_subplots(axes)), \
                 mock.patch.object(plt, "close"), \
                 mock.patch("builtins.print"):
                summary = pcur.plot_progress_curve(tmp)
            label = axes[0]._named("set_ylabel")[0][1][0]
        self.assertTrue(summary["legacy_records"])
        self.assertIn("penalty", label, "旧口径纵轴必须写明含罚分")

    def test_plot_uses_plain_label_when_penalty_is_known(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 0, 102.5632, penalty=0.2)
            _write_best(tmp, 40, 0.2438, penalty=11.7553)
            axes = []
            import matplotlib.pyplot as plt
            with mock.patch.object(plt, "subplots", _patched_subplots(axes)), \
                 mock.patch.object(plt, "close"), \
                 mock.patch("builtins.print"):
                summary = pcur.plot_progress_curve(tmp)
            label = axes[0]._named("set_ylabel")[0][1][0]
        self.assertFalse(summary["legacy_records"])
        self.assertEqual(label, "MSE")


class RenderProgressSectionTest(unittest.TestCase):
    def _summary(self):
        """与 ``plot_progress_curve`` 的返回同形（含三条曲线的子摘要）。"""
        points = [
            {"sample_order": 0, "iteration": 1, "mse": 102.56318982336019,
             "nmse": 0.05097, "penalty": 0.0, "score": -102.56318982336019},
            {"sample_order": 87, "iteration": 22, "mse": 0.867473,
             "nmse": 0.000431133, "penalty": 0.0, "score": -0.867473},
        ]
        return {"path": "x.png", "n_points": 2, "log_scale": True,
                "first": points[0], "best": points[1], "points": points,
                "mse": {"path": "m.png", "n_points": 2, "log_scale": True, "symlog": False,
                        "scale": "log"},
                "penalty": {"path": "p.png", "n_points": 2, "log_scale": False,
                            "symlog": False, "scale": "linear"},
                "score": {"path": "s.png", "n_points": 2, "log_scale": False,
                          "symlog": True, "scale": "symlog"}}

    def test_none_yields_empty_string(self):
        self.assertEqual(pcur.render_progress_section(None), "")

    def test_section_carries_heading_png_and_numbers(self):
        text = pcur.render_progress_section(self._summary())
        self.assertIn(pcur.PROGRESS_HEADING, text)
        self.assertIn(pcur.PROGRESS_PNG_NAME, text)
        self.assertIn("sample_order=87", text)
        self.assertIn("0.867473", text)
        # NMSE 按 4 位有效数字渲染（与 explain/data_facts 的口径一致）
        self.assertIn("NMSE=0.0004311", text)
        self.assertIn("对数刻度", text)

    def test_section_shows_all_three_curves(self):
        text = pcur.render_progress_section(self._summary())
        for name in (pcur.PROGRESS_PNG_NAME, pcur.PENALTY_PNG_NAME, pcur.SCORE_PNG_NAME):
            with self.subTest(png=name):
                self.assertIn(name, text)
        self.assertIn("罚分=", text)
        self.assertIn("评分=", text)
        self.assertLess(text.index(pcur.PROGRESS_PNG_NAME), text.index(pcur.PENALTY_PNG_NAME))
        self.assertLess(text.index(pcur.PENALTY_PNG_NAME), text.index(pcur.SCORE_PNG_NAME))

    def test_per_sample_block_lists_the_three_images_and_counts(self):
        s = self._summary()
        s["per_sample"] = {
            "n_points": 93, "n_clean": 57,
            "best_score": {"sample_order": 71, "mse": 0.2716, "penalty": 0.0,
                           "score": -0.2716},
            "worst_score": {"sample_order": 0, "mse": 102.6, "penalty": 0.0,
                            "score": -102.6},
            "max_penalty": {"sample_order": 35, "mse": 12.9, "penalty": 41.1,
                            "score": -54.0},
            "mse": {"path": "a.png", "n_points": 93, "log_scale": True},
            "penalty": {"path": "b.png", "n_points": 93, "log_scale": True},
            "score": {"path": "c.png", "n_points": 93, "log_scale": False},
        }
        text = pcur.render_progress_section(s)
        for name in (pcur.PER_SAMPLE_MSE_PNG_NAME, pcur.PER_SAMPLE_PENALTY_PNG_NAME,
                     pcur.PER_SAMPLE_SCORE_PNG_NAME):
            with self.subTest(png=name):
                self.assertIn(name, text)
        self.assertIn("**93** 个样本点", text)
        self.assertIn("**57** 个", text)
        self.assertIn("最大罚分：sample_order=35，罚分=41.1", text)
        self.assertIn("体检罚分 == 0", text)
        # 逐样本块必须排在刷新点块之前（原始轨迹在前，收敛摘要在后）
        self.assertLess(text.index(pcur.PER_SAMPLE_MSE_PNG_NAME),
                        text.index(pcur.PROGRESS_PNG_NAME))

    def test_per_sample_block_discloses_the_topk_limitation(self):
        """旧实验只留 top-K：不写明会被读成"只评估了这么多次"。"""
        s = self._summary()
        s["per_sample"] = {
            "n_points": 10, "n_clean": 3,
            "best_score": {"sample_order": 9, "mse": 0.5, "penalty": 0.0, "score": -0.5},
            "worst_score": {"sample_order": 0, "mse": 9.0, "penalty": 0.0, "score": -9.0},
            "max_penalty": None,
            "mse": {"path": "a.png", "n_points": 10, "log_scale": True},
            "penalty": None, "score": {"path": "c.png", "n_points": 10, "log_scale": False},
        }
        text = pcur.render_progress_section(s)
        self.assertIn("已落盘", text)
        self.assertIn("top-K", text)
        self.assertNotIn("该批样本里最大罚分", text)

    def test_per_sample_block_is_absent_without_samples(self):
        text = pcur.render_progress_section(self._summary())
        self.assertNotIn(pcur.PER_SAMPLE_MSE_PNG_NAME, text)
        self.assertNotIn("### 逐样本轨迹", text)

    def test_missing_penalty_curve_is_disclosed_instead_of_an_image(self):
        s = self._summary()
        s["penalty"] = None
        s["legacy_records"] = True
        text = pcur.render_progress_section(s)
        self.assertNotIn(pcur.PENALTY_PNG_NAME, text)
        self.assertIn("不画罚分曲线", text)
        self.assertIn(pcur.SCORE_PNG_NAME, text)

    def test_caveat_states_the_metric_is_in_sample(self):
        text = pcur.render_progress_section(self._summary())
        self.assertIn("样本内", text)
        self.assertIn("不代表泛化", text)

    def test_section_reports_penalty_separately_from_mse(self):
        """MSE 与罚分必须分开写：否则读者会把 11.999 当拟合质量（真实拟合 0.2438）。"""
        s = self._summary()
        s["best"] = {"sample_order": 40, "iteration": 10, "mse": 0.2438,
                     "nmse": 1.21144e-4, "penalty": 11.7553, "score": -11.9991}
        text = pcur.render_progress_section(s)
        self.assertIn("MSE=0.2438", text)
        self.assertIn("动态范围体检罚分 11.7553", text)
        self.assertNotIn("MSE=11.9990", text)

    def test_legacy_records_are_flagged(self):
        s = self._summary()
        s["legacy_records"] = True
        text = pcur.render_progress_section(s)
        self.assertIn("没有 `penalty` 字段", text)
        self.assertIn("MSE (+ pathology penalty)", text)
        # 旧口径下**不能**再声称 MSE 是拟合本身（那是拆分后的口径）
        self.assertNotIn("**拟合本身**的均方误差", text)

    def test_current_caliber_states_the_split(self):
        text = pcur.render_progress_section(self._summary())
        self.assertIn("**拟合本身**的均方误差", text)
        self.assertIn("评分 = −(拟合 MSE + 罚分)", text)
        self.assertNotIn("没有 `penalty` 字段", text)

    def test_current_caliber_warns_the_mse_step_is_not_monotone(self):
        """只看 MSE 一条线会把"低 MSE 高罚分"读成改善，口径里必须点名。"""
        text = pcur.render_progress_section(self._summary())
        self.assertIn("不是单调的", text)
        self.assertIn("低 MSE 高罚分", text)

    def test_missing_nmse_is_not_printed_as_none(self):
        s = self._summary()
        s["first"] = {"sample_order": 0, "iteration": None, "mse": 5.0, "nmse": None}
        text = pcur.render_progress_section(s)
        self.assertNotIn("None", text)

    def test_missing_penalty_value_is_not_printed_as_none(self):
        """混合目录（首个刷新点没有 penalty 字段）：写 n/a，绝不把 None 写进报告。"""
        s = self._summary()
        s["points"] = [{"sample_order": 0, "mse": 5.0, "nmse": None,
                        "penalty": None, "score": -5.0},
                       {"sample_order": 5, "mse": 2.0, "nmse": None,
                        "penalty": 3.0, "score": -5.0}]
        s["first"] = s["points"][0]
        s["best"] = s["points"][1]
        text = pcur.render_progress_section(s)
        self.assertNotIn("None", text)
        self.assertIn("罚分=n/a", text)
        self.assertIn("罚分=3", text)


class UpsertProgressSectionTest(unittest.TestCase):
    _REPORT = ("# 标题\n\n## 1. 正文\n\n正文内容\n\n## 参考文献\n\n[1] 文献\n")

    def test_inserted_before_references(self):
        section = pcur.render_progress_section(
            {"n_points": 1, "log_scale": True,
             "first": {"sample_order": 0, "mse": 1.0, "nmse": None},
             "best": {"sample_order": 0, "mse": 1.0, "nmse": None}, "points": []})
        out = pcur.upsert_progress_section(self._REPORT, section)
        self.assertIn(pcur.PROGRESS_HEADING, out)
        self.assertLess(out.index(pcur.PROGRESS_HEADING), out.index("## 参考文献"))
        self.assertLess(out.index("## 1. 正文"), out.index(pcur.PROGRESS_HEADING))

    def test_idempotent(self):
        section = pcur.render_progress_section(
            {"n_points": 1, "log_scale": True,
             "first": {"sample_order": 0, "mse": 1.0, "nmse": None},
             "best": {"sample_order": 0, "mse": 1.0, "nmse": None}, "points": []})
        once = pcur.upsert_progress_section(self._REPORT, section)
        twice = pcur.upsert_progress_section(once, section)
        self.assertEqual(once, twice, "重复回填必须幂等")
        self.assertEqual(twice.count(pcur.PROGRESS_HEADING), 1)

    def test_existing_section_is_replaced(self):
        old = f"{pcur.PROGRESS_HEADING}\n\n旧数字 MSE=999\n\n## 参考文献\n\n[1] 文献\n"
        new_section = f"{pcur.PROGRESS_HEADING}\n\n新数字 MSE=1\n"
        out = pcur.upsert_progress_section(f"# T\n\n{old}", new_section)
        self.assertNotIn("999", out)
        self.assertIn("MSE=1", out)
        self.assertEqual(out.count(pcur.PROGRESS_HEADING), 1)

    def test_legacy_mse_only_section_is_replaced_not_duplicated(self):
        """历史报告里的标题是「## 训练进度：MSE 随 sample_order 的变化」（没有子标题）。

        回填必须整节替换——按前缀剥离，否则老标题小节会留在原地、与新小节并存。
        """
        legacy = "## 训练进度：MSE 随 sample_order 的变化\n\n旧数字 MSE=424242\n"
        report = f"# T\n\n## 1. 正文\n\n正文\n\n{legacy}\n## 参考文献\n\n[1] 文献\n"
        section = pcur.render_progress_section(
            {"n_points": 1, "log_scale": True,
             "first": {"sample_order": 0, "mse": 1.0, "nmse": None},
             "best": {"sample_order": 0, "mse": 1.0, "nmse": None}, "points": []})
        out = pcur.upsert_progress_section(report, section)
        self.assertNotIn("424242", out)
        self.assertEqual(out.count("## 训练进度"), 1)
        self.assertLess(out.index("## 1. 正文"), out.index(pcur.PROGRESS_HEADING))
        self.assertLess(out.index(pcur.PROGRESS_HEADING), out.index("## 参考文献"))

    def test_sub_headings_do_not_end_the_section(self):
        """### 子标题属于本节内容：剥离时不能在那里停下（否则留下半节旧内容）。"""
        report = (f"# T\n\n{pcur.PROGRESS_HEADING}\n\n### 子标题\n\n子节内容\n\n"
                  "## 参考文献\n\n[1] 文献\n")
        stripped = pcur._strip_progress_section(report)
        self.assertNotIn("子节内容", stripped)
        self.assertIn("## 参考文献", stripped)

    def test_anchor_absent_appends_at_end(self):
        out = pcur.upsert_progress_section("# T\n\n正文\n", "# 某小节\n")
        self.assertTrue(out.rstrip().endswith("# 某小节"))

    def test_empty_section_leaves_text_untouched(self):
        self.assertEqual(pcur.upsert_progress_section(self._REPORT, ""), self._REPORT)


class SectionOrderAndSingleSourceTest(unittest.TestCase):
    def test_anchor_matches_explain_reference_heading(self):
        """锚点是"参考文献之前"，必须与 explain 的标题字面一致（不能各写一份漂移）。"""
        from drsr_420.reporting import explain as explain_mod

        self.assertEqual(pcur._REFERENCE_ANCHOR, explain_mod.REFERENCE_HEADING)

    def test_progress_sits_between_range_check_and_references(self):
        from drsr_420.reporting import explain as explain_mod
        progress = {"n_points": 2, "log_scale": True,
                    "first": {"sample_order": 0, "mse": 100.0, "nmse": 0.05},
                    "best": {"sample_order": 7, "mse": 1.0, "nmse": 5e-4},
                    "points": []}
        text = explain_mod.assemble_explain(
            "正文", ReportData(
                refs=[{"title": "T", "doi": "10.1/x", "source": "s"}],
                range_check={"span_ratio": 1.0, "limit": 15.0, "slope_max": 0.1,
                             "slope_limit": 2.0, "coef_ratio": 1.0, "coef_limit": 8.0},
                progress=progress))
        i_range = text.index(explain_mod.RANGE_HEADING)
        i_prog = text.index(pcur.PROGRESS_HEADING)
        i_ref = text.index(explain_mod.REFERENCE_HEADING)
        self.assertLess(i_range, i_prog)
        self.assertLess(i_prog, i_ref)

    def test_no_progress_means_no_section(self):
        from drsr_420.reporting import explain as explain_mod
        text = explain_mod.assemble_explain("正文", ReportData(progress=None))
        self.assertNotIn(pcur.PROGRESS_HEADING, text)


class PruneAndVisualizeWiringTest(unittest.TestCase):
    """端到端：``prune_and_visualize`` 必须把进度摘要放进剪枝摘要（报告据此装配小节）。"""

    _FUNC = ("Variables:\n"
             "- Independents: x1, x2\n"
             "- Dependent: y\n"
             "def equation(x1, x2, params):\n"
             "    return params[0] + params[1]*x1 + params[2]*x2\n")

    def test_summary_carries_progress(self):
        from drsr_420.reporting.find_best_eq import prune_and_visualize
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "samples").mkdir(parents=True)
            (root / "samples" / "top01_samples_1.json").write_text(
                json.dumps({"score": -0.5, "sample_order": 1,
                            "function": self._FUNC, "params": [1.0, 2.0, 3.0]}),
                encoding="utf-8")
            (root / "config_snapshot.json").write_text(
                json.dumps({"data_csv": "data/tiny/train.csv"}), encoding="utf-8")
            data_dir = root / "data" / "tiny"
            data_dir.mkdir(parents=True)
            rows = ["x1,x2,y"] + [f"{v},{6.0 - v},{1.0 + 2.0 * v + 3.0 * (6.0 - v)}"
                                  for v in (1.0, 2.0, 3.0, 4.0, 5.0)]
            (data_dir / "train.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            _write_best(root, 0, 100.0, nmse=0.05)
            _write_best(root, 1, 1.0, nmse=5e-4)
            with mock.patch("builtins.print"):
                summary = prune_and_visualize(
                    str(root), self._FUNC, [1.0, 2.0, 3.0], threshold=0.1,
                    sample_range=(1, 6), test_csv="none")
            progress = summary.get("progress")
            self.assertIsNotNone(progress, "剪枝摘要必须带上进度（否则报告没有这一节）")
            self.assertEqual(progress["n_points"], 2)
            self.assertEqual(progress["best"]["sample_order"], 1)
            self.assertTrue((root / pcur.PROGRESS_PNG_NAME).exists())


if __name__ == "__main__":
    unittest.main()