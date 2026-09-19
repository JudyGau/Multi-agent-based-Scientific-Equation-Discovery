"""样本外（held-out）验证 + 变量列名兜底的回归测试。

三件事：

1. **test.csv 只报告、不参与选择**：解析优先级（显式 → 快照 → 训练数据同目录 →
   按目录名推断）、指标口径（NMSE 用训练集方差）、explain.md 的「样本外验证」小节由
   系统生成且 LLM 自写的同名小节会被替换；
2. **列名兜底**：函数头里的变量名是 LLM 写的，与 CSV 表头不完全一致时（实测历史运行
   因变量写成 `um`，CSV 列是 `miu`）要按大小写/位置对上，而不是静默跳过整个分析；
3. **没有 config_snapshot.json 的历史目录**：按目录名 `<问题>_<时间戳>` 反推
   `data/<问题>/train.csv`，否则那批目录的剪枝拟合对比与曲线永远画不出来。
"""
import glob
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import sympy as sp

from drsr_420.analysis import holdout as ho
from drsr_420.analysis import prune_report as pr

_FUNC = ("Variables:\n"
         "- Independents: x1, x2\n"
         "- Dependent: y\n"
         "def equation(x1, x2, params):\n"
         "    return params[0]*x1 + params[1]*x2 + params[2]\n")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _make_experiment(root: pathlib.Path, *, snapshot: dict | None = None,
                     train_rows=("x1,x2,y",), test_rows=("x1,x2,y",)) -> None:
    """搭最小实验目录：样本、config_snapshot、训练/样本外 CSV。"""
    (root / "samples").mkdir(parents=True, exist_ok=True)
    (root / "samples" / "top01_samples_1.json").write_text(
        json.dumps({"score": -0.5, "sample_order": 1,
                    "function": _FUNC, "params": [2.0, 3.0, 1.0]}), encoding="utf-8")
    data_dir = root / "data" / "tiny"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    if test_rows:
        (data_dir / "test.csv").write_text("\n".join(test_rows) + "\n", encoding="utf-8")
    if snapshot is not None:
        (root / "config_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")


_TRAIN = ["x1,x2,y"] + [f"{v},{6 - v},{2 * v + 3 * (6 - v) + 1}" for v in (1, 2, 3, 4, 5)]
#: held-out：两个点都没参与拟合，但落在**同一个真公式**（2x1+3x2+1）上，
#: 因此"公式正确时样本外误差应当≈0"可以精确断言。
_TEST = ["x1,x2,y", "0.5,5.5,18.5", "5.5,0.5,13.5"]


class ResolveTestBase(unittest.TestCase):
    def setUp(self):
        # 模块级缓存会影响"同一目录、不同参数"的用例，逐个清空
        ho._resolved.clear()
        ho._logged.clear()


class ResolveTestCsvTest(ResolveTestBase):
    def test_explicit_path_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "p_20260101-000000"
            _make_experiment(root, snapshot={"data_csv": "data/tiny/train.csv"})
            ho._resolved.clear()
            got = ho.resolve_test_csv(str(root), str(root / "data" / "tiny" / "test.csv"))
            self.assertTrue(got and got.endswith("test.csv"))

    def test_explicit_none_disables_autodetect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "p_20260101-000000"
            _make_experiment(root, snapshot={"data_csv": "data/tiny/train.csv"})
            self.assertIsNone(ho.resolve_test_csv(str(root), "none"))

    def test_snapshot_path_wins_over_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "p_20260101-000000"
            _make_experiment(root, snapshot={"data_csv": "data/tiny/train.csv"})
            other = root / "special_test.csv"
            other.write_text("\n".join(_TEST) + "\n", encoding="utf-8")
            (root / "config_snapshot.json").write_text(
                json.dumps({"data_csv": "data/tiny/train.csv",
                            "test_csv": "special_test.csv"}), encoding="utf-8")
            self.assertEqual(ho.resolve_test_csv(str(root)), str(other))

    def test_snapshot_records_disabled(self):
        """当年 `--test_csv none` 关掉的运行，后来复跑不得又自动探测出来。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "p_20260101-000000"
            _make_experiment(root, snapshot={"data_csv": "data/tiny/train.csv",
                                             "test_csv": "none"})
            self.assertIsNone(ho.resolve_test_csv(str(root)))

    def test_autodetect_sibling_of_training_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "p_20260101-000000"
            _make_experiment(root, snapshot={"data_csv": "data/tiny/train.csv"})
            got = ho.resolve_test_csv(str(root))
            self.assertEqual(os.path.normpath(got),
                             os.path.normpath(str(root / "data" / "tiny" / "test.csv")))

    def test_returns_none_when_no_test_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "p_20260101-000000"
            _make_experiment(root, snapshot={"data_csv": "data/tiny/train.csv"},
                             test_rows=())
            self.assertIsNone(ho.resolve_test_csv(str(root)))

    def test_missing_explicit_file_warns_and_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("builtins.print") as printer:
            root = pathlib.Path(tmp) / "p_20260101-000000"
            _make_experiment(root, snapshot={"data_csv": "data/tiny/train.csv"})
            self.assertIsNone(ho.resolve_test_csv(str(root), "no/such/file.csv"))
            printed = "\n".join(str(c.args[0]) for c in printer.call_args_list if c.args)
            self.assertIn("--test_csv 指定的文件不存在", printed)

    def test_infers_dataset_from_run_dir_name(self):
        """历史目录（无 config_snapshot.json）按目录名反推 data/<问题>/train.csv。"""
        repo_data = _REPO_ROOT / "data" / "MRFCompress-Cuboid"
        if not (repo_data / "test.csv").is_file():
            self.skipTest("仓库里没有 data/MRFCompress-Cuboid/test.csv")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "MRFCompress-Cuboid_20260101-000000"
            (root / "samples").mkdir(parents=True, exist_ok=True)
            got = ho.resolve_test_csv(str(root))
        self.assertEqual(got, str(repo_data / "test.csv"))


class LoadTrainingFallbackTest(ResolveTestBase):
    def test_training_data_inferred_without_snapshot(self):
        repo_data = _REPO_ROOT / "data" / "MRFCompress-Cuboid"
        if not (repo_data / "train.csv").is_file():
            self.skipTest("仓库里没有 data/MRFCompress-Cuboid/train.csv")
        with tempfile.TemporaryDirectory() as tmp, mock.patch("builtins.print"):
            root = pathlib.Path(tmp) / "MRFCompress-Cuboid_20260101-000000"
            root.mkdir(parents=True)
            data = pr.load_training_data(str(root))
        self.assertIsNotNone(data)
        self.assertEqual(list(data.dtype.names), ["lambda12", "lambda23", "sigma"])

    def test_inference_requires_timestamped_dir_name(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("builtins.print"):
            root = pathlib.Path(tmp) / "not-a-run-dir"
            root.mkdir(parents=True)
            self.assertIsNone(pr.load_training_data(str(root)))


class ResolveColumnsTest(unittest.TestCase):
    def _data(self, header: str):
        return np.genfromtxt(_csv(header), delimiter=",", names=True)

    def test_exact_match(self):
        data = self._data("lambda12,lambda23,sigma")
        dep, ind, note = pr.resolve_columns(data, "sigma", ["lambda12", "lambda23"])
        self.assertEqual((dep, ind), ("sigma", ["lambda12", "lambda23"]))
        self.assertEqual(note, "")

    def test_case_insensitive_match(self):
        data = self._data("X1,x2,Y")
        dep, ind, note = pr.resolve_columns(data, "y", ["x1", "x2"])
        self.assertEqual((dep, ind), ("Y", ["X1", "x2"]))
        self.assertIn("忽略大小写", note)

    def test_dependent_falls_back_by_position(self):
        """实测场景：函数头写 um，CSV 列是 miu。"""
        data = self._data("alpha,lambda12,lambda23,miu")
        dep, ind, note = pr.resolve_columns(data, "um", ["alpha", "lambda12", "lambda23"])
        self.assertEqual(dep, "miu")
        self.assertIn("按位置兜底", note)
        self.assertEqual(ind, ["alpha", "lambda12", "lambda23"])

    def test_unknown_independent_raises(self):
        data = self._data("a,b,c")
        with self.assertRaises(KeyError):
            pr.resolve_columns(data, "c", ["a", "zzz"])

    def test_ambiguous_dependent_raises(self):
        data = self._data("a,b,c,d")
        with self.assertRaises(KeyError):
            pr.resolve_columns(data, "um", ["a", "b"])


def _csv(header: str) -> str:
    """按表头生成一个临时 CSV（每列 1,2,3 三行），返回路径（调用方负责删除）。"""
    cols = header.split(",")
    rows = [header] + [",".join(str(i) for _ in cols) for i in (1, 2, 3)]
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    f.write("\n".join(rows) + "\n")
    f.close()
    return f.name


class CompareFitsFallbackTest(unittest.TestCase):
    def test_compare_fits_survives_dependent_name_mismatch(self):
        """旧实现因列名对不上直接返回空字典（静默丢失整个剪枝拟合对比）。"""
        path = _csv("alpha,lambda12,lambda23,miu")
        self.addCleanup(os.unlink, path)
        data = np.genfromtxt(path, delimiter=",", names=True)
        alpha, l12, l23 = sp.symbols("alpha lambda12 lambda23")
        expr = alpha + l12 + l23
        with mock.patch("builtins.print") as printer:
            fit = pr.compare_fits("um", ["alpha", "lambda12", "lambda23"], data, expr, expr)
        self.assertIn("mse_before", fit)
        self.assertTrue(fit["identical"])
        printed = "\n".join(str(c.args[0]) for c in printer.call_args_list if c.args)
        self.assertIn("按位置兜底", printed)


class EvaluateHoldoutTest(unittest.TestCase):
    def _setup(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name) / "p_20260101-000000"
        _make_experiment(root, snapshot={"data_csv": "data/tiny/train.csv"},
                         train_rows=tuple(_TRAIN), test_rows=tuple(_TEST))
        return root

    def test_metrics_and_per_point_rows(self):
        root = self._setup()
        data = ho.load_test_data(str(root))
        self.assertEqual(len(data), 2)
        x1, x2 = sp.symbols("x1 x2")
        expr = 2.0 * x1 + 3.0 * x2 + 1.0        # 与数据同真公式
        holdout = ho.evaluate_holdout("y", ["x1", "x2"], expr, data,
                                      train_var=float(np.var([float(r.split(",")[2])
                                                              for r in _TRAIN[1:]])),
                                      path="x/test.csv")
        self.assertEqual(holdout["n_points"], 2)
        self.assertLess(holdout["mse"], 1e-20)
        self.assertLess(holdout["max_rel_err"], 1e-12)
        self.assertEqual(len(holdout["rows"]), 2)
        self.assertEqual(holdout["rows"][0]["variables"]["x1"], 0.5)
        self.assertEqual(holdout["path"], "x/test.csv")

    def test_nmse_uses_training_variance(self):
        root = self._setup()
        data = ho.load_test_data(str(root))
        x1, x2 = sp.symbols("x1 x2")
        holdout = ho.evaluate_holdout("y", ["x1", "x2"], 2.0 * x1 + 3.0 * x2, data,
                                      train_var=4.0)
        self.assertAlmostEqual(holdout["nmse"], holdout["mse"] / 4.0, places=12)

    def test_summary_mentions_not_used_for_selection(self):
        root = self._setup()
        data = ho.load_test_data(str(root))
        x1, x2 = sp.symbols("x1 x2")
        holdout = ho.evaluate_holdout("y", ["x1", "x2"], 2.0 * x1 + 3.0 * x2 + 5.0, data,
                                      train_var=10.0)
        text = ho.format_holdout_summary(holdout, {"nmse_before": 1e-6})
        self.assertIn("不参与采样、打分与样本选择", text)
        self.assertIn("倍", text)
        self.assertIn("最大相对误差", text)

    def test_summary_without_holdout_explains_why(self):
        text = ho.format_holdout_summary(None)
        self.assertIn("没有可用的 held-out 数据", text)

    def test_duplicate_test_set_is_flagged_not_reported_as_holdout(self):
        """实测 data/MRFShear-3 的 test.csv 与 train.csv 逐行相同：不能当独立验证集。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "p_20260101-000000"
            _make_experiment(root, snapshot={"data_csv": "data/tiny/train.csv"},
                             train_rows=tuple(_TEST), test_rows=tuple(_TEST))
            train = np.genfromtxt(root / "data" / "tiny" / "train.csv", delimiter=",",
                                  names=True)
            test = ho.load_test_data(str(root))
            x1, x2 = sp.symbols("x1 x2")
            holdout = ho.evaluate_holdout("y", ["x1", "x2"],
                                          2.0 * x1 + 3.0 * x2 + 1.0, test,
                                          train_var=2.0, train_data=train)
        self.assertTrue(holdout["overlaps_train"])
        self.assertEqual(holdout["n_overlap_train"], holdout["n_points"])
        summary = ho.format_holdout_summary(holdout)
        self.assertIn("全部出现在训练集中", summary)
        self.assertIn("不能当作泛化能力", summary)
        section = ho.render_holdout_section(holdout)
        self.assertIn("并不是独立的 held-out 集", section)

    def test_disjoint_test_set_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "p_20260101-000000"
            _make_experiment(root, snapshot={"data_csv": "data/tiny/train.csv"},
                             train_rows=tuple(_TRAIN), test_rows=tuple(_TEST))
            train = np.genfromtxt(root / "data" / "tiny" / "train.csv", delimiter=",",
                                  names=True)
            test = ho.load_test_data(str(root))
            holdout = ho.evaluate_holdout("y", ["x1", "x2"], sp.Symbol("x1"), test,
                                          train_var=2.0, train_data=train)
        self.assertFalse(holdout["overlaps_train"])
        self.assertEqual(holdout["n_overlap_train"], 0)


class HoldoutSectionTest(unittest.TestCase):
    def _holdout(self):
        return {
            "path": "data/tiny/test.csv", "n_points": 2, "mse": 1.0, "nmse": 0.25,
            "max_abs_err": 1.5, "max_rel_err": 0.1, "train_var": 4.0,
            "in_sample_nmse": None,
            "rows": [
                {"variables": {"x1": 0.5, "x2": 5.5}, "observed": 13.5,
                 "predicted": 14.0, "abs_err": 0.5, "rel_err": 0.037},
                {"variables": {"x1": 5.5, "x2": 0.5}, "observed": 13.5,
                 "predicted": 13.0, "abs_err": 0.5, "rel_err": 0.037},
            ],
        }

    def test_section_renders_table_and_points(self):
        text = ho.render_holdout_section(self._holdout(),
                                         {"n_points": 5, "mse_before": 1e-6,
                                          "nmse_before": 1e-7,
                                          "max_abs_err_before": 1e-3,
                                          "max_rel_err_before": 0.001})
        self.assertIn(ho.HOLDOUT_HEADING, text)
        self.assertIn("样本外", text)
        self.assertIn("data/tiny/test.csv", text)
        self.assertIn("| 1 |", text)
        self.assertIn("2.5e+06 倍", text)      # 0.25 / 1e-7
        self.assertIn("不参与采样、打分、早停与样本选择", text)

    def test_section_without_holdout_warns_in_sample_is_not_generalization(self):
        text = ho.render_holdout_section(None)
        self.assertIn(ho.HOLDOUT_HEADING, text)
        self.assertIn("样本内", text)
        self.assertIn("不能当作泛化误差", text)

    def test_strip_removes_llm_written_section(self):
        text = ("## 结论\n\n正文\n\n## 样本外验证\n\n我自己编的数字\n\n## 参考文献\n\n[1] x")
        stripped = ho.strip_holdout_section(text)
        self.assertNotIn("我自己编的数字", stripped)
        self.assertIn("## 结论", stripped)
        self.assertIn("## 参考文献", stripped)

    def test_assemble_keeps_body_and_appends_authoritative_section(self):
        from drsr_420.analysis.explain import _assemble_explain

        body = "## 结论\n\n正文\n\n## 样本外验证\n\nLLM 编的\n"
        text = _assemble_explain(body, [], holdout=self._holdout(),
                                 fit={"mse_before": 1e-6, "nmse_before": 1e-7})
        self.assertIn("正文", text)
        self.assertNotIn("LLM 编的", text)
        self.assertIn(ho.HOLDOUT_HEADING, text)
        self.assertIn("样本外 NMSE", text.replace("|", " "))


class PlotHoldoutWiringTest(unittest.TestCase):
    def _setup(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name) / "p_20260101-000000"
        _make_experiment(root, snapshot={"data_csv": "data/tiny/train.csv"},
                         train_rows=tuple(_TRAIN), test_rows=tuple(_TEST))
        return root

    def test_heldout_points_drawn_with_a_distinct_marker(self):
        """held-out 点必须用另一种标记画出来（否则无从判断是不是只在训练点上插值）。"""
        ho._resolved.clear()
        from drsr_420.analysis import expr_curves as ec
        import matplotlib.pyplot as plt

        scatters: list = []
        lines: list = []

        class _FakeAx:
            def scatter(self, x, y, **kw):
                scatters.append((np.asarray(x, dtype=float), kw))

            def plot(self, x, y, **kw):
                lines.append((np.asarray(x, dtype=float), kw))

            def set_xlabel(self, *_a, **_k): pass
            def set_ylabel(self, *_a, **_k): pass
            def set_title(self, *_a, **_k): pass
            def grid(self, *_a, **_k): pass
            def legend(self, *_a, **_k): pass

        class _FakeFig:
            def tight_layout(self): pass
            def savefig(self, *_a, **_k): pass

        root = self._setup()
        with mock.patch.object(plt, "subplots", lambda *a, **k: (_FakeFig(), _FakeAx())), \
             mock.patch.object(plt, "close"), \
             mock.patch("builtins.print"):
            written = ec.plot_data_curves(str(root), "y", ["x1", "x2"],
                                          2.0 * sp.Symbol("x1") + 3.0 * sp.Symbol("x2") + 1.0,
                                          test_csv=None)
        self.assertEqual(len(written), 2)
        held_out = [s for s in scatters if s[1].get("marker") == "^"]
        self.assertEqual(len(held_out), 2, "每个自变量一幅图，各画一次 held-out 散点")
        np.testing.assert_allclose(sorted(held_out[0][0]), [0.5, 5.5])
        # 曲线（plot）只有一条：未剪枝时不画重复曲线
        self.assertEqual(len(lines), 2)
        self.assertNotIn("after pruning", lines[0][1].get("label", ""))

    def test_plot_expr_curves_passes_test_csv_through(self):
        ho._resolved.clear()
        from drsr_420.analysis import expr_curves as ec

        root = self._setup()
        captured = {}

        def _spy(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return []

        with mock.patch.object(ec, "plot_data_curves", _spy), \
             mock.patch("builtins.print"):
            ec.plot_expr_curves(str(root), threshold=0.1, sample_range=(1, 6),
                                test_csv="none")
        self.assertEqual(captured["kwargs"].get("test_csv"), "none")

    def test_bad_holdout_columns_do_not_break_plotting(self):
        ho._resolved.clear()
        from drsr_420.analysis import expr_curves as ec

        root = self._setup()
        # held-out 文件缺一个自变量列：无法定位，但训练曲线仍要画出来
        (root / "data" / "tiny" / "test.csv").write_text("x1,z,w\n1,2,3\n",
                                                         encoding="utf-8")
        with mock.patch("builtins.print") as printer:
            written = ec.plot_data_curves(str(root), "y", ["x1", "x2"],
                                          sp.Symbol("x1") + sp.Symbol("x2"))
        self.assertTrue(written, "样本外数据有问题时仍应画出训练曲线")
        printed = "\n".join(str(c.args[0]) for c in printer.call_args_list if c.args)
        self.assertIn("样本外数据无法用于绘图", printed)


class FindBestEqPlumbingTest(unittest.TestCase):
    def test_test_csv_reaches_prune_and_visualize(self):
        from drsr_420.analysis import find_best_eq as fbe

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "p_20260101-000000"
            _make_experiment(root, snapshot={"data_csv": "data/tiny/train.csv"})
            seen = {}

            def _fake_prune(results_root, func, params, threshold, sample_range,
                            test_csv=None):
                seen["test_csv"] = test_csv
                return None

            with mock.patch.object(fbe, "prune_and_visualize", _fake_prune), \
                 mock.patch.object(fbe, "explain_best_sample", lambda *a, **k: None), \
                 mock.patch("builtins.print"):
                fbe.find_best_eq(str(root), test_csv="none")
        self.assertEqual(seen["test_csv"], "none")


class CliOptionTest(unittest.TestCase):
    def test_parser_accepts_test_csv(self):
        from drsr_420.cli.main import build_parser

        argv = ["--data_csv", "d.csv"]
        self.assertIsNone(build_parser().parse_args(argv).test_csv)
        self.assertEqual(build_parser().parse_args(argv + ["--test_csv", "t.csv"]).test_csv,
                         "t.csv")
        self.assertEqual(build_parser().parse_args(argv + ["--test_csv", "none"]).test_csv,
                         "none")


if __name__ == "__main__":
    unittest.main()
