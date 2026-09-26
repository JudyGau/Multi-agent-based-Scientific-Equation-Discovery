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

from drsr_420.analysis import progress_curve as pcur


def _matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False


def _write_best(root, order, mse, nmse=None, iteration=1, penalty=None):
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
    (d / f"best_sample_{order}.json").write_text(json.dumps(rec), encoding="utf-8")


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

    def test_missing_directory_yields_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(pcur.load_best_history(tmp), [])


class PlotProgressCurveTest(unittest.TestCase):
    def test_no_history_returns_none_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("builtins.print"):
                summary = pcur.plot_progress_curve(tmp)
            self.assertIsNone(summary)
            self.assertFalse((pathlib.Path(tmp) / pcur.PROGRESS_PNG_NAME).exists(),
                             "没有记录时不能留下空图（否则报告会引用一张空图）")

    def test_staircase_markers_and_log_axis_wiring(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 0, 102.5632)
            _write_best(tmp, 5, 18.8466)
            _write_best(tmp, 7, 4.2504)
            ax = _FakeAx()
            import matplotlib.pyplot as plt
            with mock.patch.object(plt, "subplots",
                                   lambda *a, **k: (_FakeFig(), ax)), \
                 mock.patch.object(plt, "close"), \
                 mock.patch("builtins.print"):
                summary = pcur.plot_progress_curve(tmp)
            step = ax._named("step")
            self.assertEqual(len(step), 1, "历史最优必须用阶梯线画")
            self.assertEqual(step[0][2].get("where"), "post")
            self.assertTrue(np.allclose(np.asarray(step[0][1][0], dtype=float), [0, 5, 7]))
            self.assertTrue(np.allclose(np.asarray(step[0][1][1], dtype=float), [102.5632, 18.8466, 4.2504]))
            self.assertEqual(len(ax._named("plot")), 1, "刷新点要有数据点标记")
            self.assertEqual(ax._named("set_yscale")[0][1], ("log",))
            self.assertEqual(len(ax._named("legend")), 1)
        self.assertTrue(summary["log_scale"])
        self.assertEqual(summary["n_points"], 3)
        self.assertEqual(summary["first"]["sample_order"], 0)
        self.assertEqual(summary["best"]["sample_order"], 7, "best 取 MSE 最小的那个刷新点")

    def test_non_positive_mse_falls_back_to_linear_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 0, 0.0)
            _write_best(tmp, 3, 1.0)
            ax = _FakeAx()
            import matplotlib.pyplot as plt
            with mock.patch.object(plt, "subplots",
                                   lambda *a, **k: (_FakeFig(), ax)), \
                 mock.patch.object(plt, "close"), \
                 mock.patch("builtins.print"):
                summary = pcur.plot_progress_curve(tmp)
            self.assertEqual(ax._named("set_yscale"), [], "有 0 值时不能取对数刻度")
        self.assertFalse(summary["log_scale"])

    @unittest.skipUnless(_matplotlib_available(), "本机没有 matplotlib")
    def test_real_render_writes_a_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 0, 102.5632, nmse=0.05097)
            _write_best(tmp, 42, 1.113129, nmse=5.5e-4)
            with mock.patch("builtins.print"):
                summary = pcur.plot_progress_curve(tmp)
            path = pathlib.Path(summary["path"])
            self.assertTrue(path.exists(), "有记录时必须真的落盘")
            self.assertGreater(path.stat().st_size, 5000, "不能是空图")
            self.assertEqual(path.name, pcur.PROGRESS_PNG_NAME)


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
            ax = _FakeAx()
            import matplotlib.pyplot as plt
            with mock.patch.object(plt, "subplots",
                                   lambda *a, **k: (_FakeFig(), ax)), \
                 mock.patch.object(plt, "close"), \
                 mock.patch("builtins.print"):
                summary = pcur.plot_progress_curve(tmp)
            label = ax._named("set_ylabel")[0][1][0]
        self.assertTrue(summary["legacy_records"])
        self.assertIn("penalty", label, "旧口径纵轴必须写明含罚分")

    def test_plot_uses_plain_label_when_penalty_is_known(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_best(tmp, 0, 102.5632, penalty=0.2)
            _write_best(tmp, 40, 0.2438, penalty=11.7553)
            ax = _FakeAx()
            import matplotlib.pyplot as plt
            with mock.patch.object(plt, "subplots",
                                   lambda *a, **k: (_FakeFig(), ax)), \
                 mock.patch.object(plt, "close"), \
                 mock.patch("builtins.print"):
                summary = pcur.plot_progress_curve(tmp)
            label = ax._named("set_ylabel")[0][1][0]
        self.assertFalse(summary["legacy_records"])
        self.assertEqual(label, "MSE")


class RenderProgressSectionTest(unittest.TestCase):
    def _summary(self):
        return {"path": "x.png", "n_points": 2, "log_scale": True,
                "first": {"sample_order": 0, "iteration": 1, "mse": 102.56318982336019,
                          "nmse": 0.05097},
                "best": {"sample_order": 87, "iteration": 22, "mse": 0.867473,
                         "nmse": 0.000431133},
                "points": []}

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

    def test_caveat_states_the_metric_is_in_sample(self):
        text = pcur.render_progress_section(self._summary())
        self.assertIn("样本内", text)
        self.assertIn("不代表泛化", text)

    def test_section_reports_penalty_separately_from_mse(self):
        """MSE 与罚分必须分开写：否则读者会把 11.999 当拟合质量（真实拟合 0.2438）。"""
        s = self._summary()
        s["best"] = {"sample_order": 40, "iteration": 10, "mse": 0.2438,
                     "nmse": 1.21144e-4, "penalty": 11.7553}
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

    def test_current_caliber_states_the_split(self):
        text = pcur.render_progress_section(self._summary())
        self.assertIn("拟合本身", text)
        self.assertIn("评分 = −(拟合 MSE + 罚分)", text)

    def test_missing_nmse_is_not_printed_as_none(self):
        s = self._summary()
        s["first"] = {"sample_order": 0, "iteration": None, "mse": 5.0, "nmse": None}
        text = pcur.render_progress_section(s)
        self.assertNotIn("None", text)


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

    def test_anchor_absent_appends_at_end(self):
        out = pcur.upsert_progress_section("# T\n\n正文\n", "# 某小节\n")
        self.assertTrue(out.rstrip().endswith("# 某小节"))

    def test_empty_section_leaves_text_untouched(self):
        self.assertEqual(pcur.upsert_progress_section(self._REPORT, ""), self._REPORT)


class SectionOrderAndSingleSourceTest(unittest.TestCase):
    def test_anchor_matches_explain_reference_heading(self):
        """锚点是"参考文献之前"，必须与 explain 的标题字面一致（不能各写一份漂移）。"""
        from drsr_420.analysis import explain as explain_mod
        self.assertEqual(pcur._REFERENCE_ANCHOR, explain_mod.REFERENCE_HEADING)

    def test_progress_sits_between_range_check_and_references(self):
        from drsr_420.analysis import explain as explain_mod
        progress = {"n_points": 2, "log_scale": True,
                    "first": {"sample_order": 0, "mse": 100.0, "nmse": 0.05},
                    "best": {"sample_order": 7, "mse": 1.0, "nmse": 5e-4},
                    "points": []}
        text = explain_mod._assemble_explain(
            "正文", [{"title": "T", "doi": "10.1/x", "source": "s"}],
            holdout=None, fit=None, range_check={"span_ratio": 1.0, "limit": 15.0,
                                                 "slope_max": 0.1, "slope_limit": 2.0,
                                                 "coef_ratio": 1.0, "coef_limit": 8.0},
            progress=progress)
        i_range = text.index(explain_mod.RANGE_HEADING)
        i_prog = text.index(pcur.PROGRESS_HEADING)
        i_ref = text.index(explain_mod.REFERENCE_HEADING)
        self.assertLess(i_range, i_prog)
        self.assertLess(i_prog, i_ref)

    def test_no_progress_means_no_section(self):
        from drsr_420.analysis import explain as explain_mod
        text = explain_mod._assemble_explain("正文", [], progress=None)
        self.assertNotIn(pcur.PROGRESS_HEADING, text)


class PruneAndVisualizeWiringTest(unittest.TestCase):
    """端到端：``prune_and_visualize`` 必须把进度摘要放进剪枝摘要（报告据此装配小节）。"""

    _FUNC = ("Variables:\n"
             "- Independents: x1, x2\n"
             "- Dependent: y\n"
             "def equation(x1, x2, params):\n"
             "    return params[0] + params[1]*x1 + params[2]*x2\n")

    def test_summary_carries_progress(self):
        from drsr_420.analysis.find_best_eq import prune_and_visualize
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