"""收尾产物的两道护栏：病理门禁（发布哪个解）与 report.md 必出。

实测背景（2026-09-26 核验）：
* ``experiments/`` 下 490 个 run 只有 **4 份** ``report.md``——真因不是"启动路径没跑到
  收尾"（``pipeline.main`` 必定调用 ``find_best_eq``），而是 ``explain_best_sample`` 在
  "没匹配到 Good 经验 / LLM 返回空"时**直接 return**，连体检/样本外/进度这些纯机器
  小节也一并丢掉；
* 病理解仍会被发布（``20260925-134149`` 的发布式自认"病理性器件"），因为评分
  ``-(拟合 MSE + 体检罚分)`` 只把病理解**压低**，压不彻底时它仍可能是最高分。
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from drsr_420.analysis import explain as explain_mod
from drsr_420.analysis import find_best_eq as fbe
from drsr_420.core import sample_records as records_mod
from drsr_420.core.profile import Profiler


def _sample(score, penalty, order, mse=0.1):
    return {"sample_order": order, "score": score, "penalty": penalty, "mse": mse,
            "function": "def equation_v1(x, y, params):\n    return params[0]",
            "params": [1.0]}


def _write(root: pathlib.Path, name: str, payload: dict) -> None:
    (root / "samples").mkdir(parents=True, exist_ok=True)
    (root / "samples" / name).write_text(json.dumps(payload), encoding="utf-8")


class SampleRecordsTest(unittest.TestCase):
    """两种文件命名都必须读到——只 glob ``*_samples_*.json`` 会让全量落盘模式失效。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)

    def test_reads_both_naming_schemes_and_dedups_by_order(self):
        _write(self.root, "samples_3.json", _sample(-1.0, 0.0, 3))
        _write(self.root, "top01_samples_3.json", _sample(-1.0, 0.0, 3))   # 同一 order 的副本
        _write(self.root, "top02_samples_5.json", _sample(-2.0, 0.0, 5))
        records = fbe.load_sample_records(str(self.root))
        self.assertEqual([r["sample_order"] for r in records], [3, 5])
        self.assertEqual(len(records), 2)

    def test_full_named_file_alone_is_found(self):
        """回归：全量落盘模式下**只有** ``samples_N.json``，旧 glob 一个都读不到。"""
        _write(self.root, "samples_7.json", _sample(-0.25, 0.0, 7))
        self.assertEqual(fbe.find_best_sample(str(self.root))[0], -0.25)

    def test_non_numeric_score_is_skipped(self):
        _write(self.root, "samples_1.json", {"sample_order": 1, "score": None})
        _write(self.root, "samples_2.json", {"sample_order": 2, "score": "oops"})
        self.assertEqual(fbe.load_sample_records(str(self.root)), [])


class ScoreBreakdownTest(unittest.TestCase):
    """``score_breakdown`` 的口径：``score = −(拟合 MSE + 体检罚分)`` 必须被拆开。

    这不是"统计好看不好看"的问题：混在一起正是要治的病——实测 20260926-151008 的
    order 34 拟合 MSE 只有 0.197 却带 36.06 罚分，只看 MSE 会把它当成胜利。
    """

    @staticmethod
    def _record(order, mse, penalty, score=None):
        if score is None:
            score = -(mse + penalty) if isinstance(penalty, (int, float)) else -mse
        return {"sample_order": order, "mse": mse, "penalty": penalty, "score": score}

    def test_empty_records_give_an_empty_breakdown(self):
        breakdown = records_mod.score_breakdown([])
        self.assertEqual(breakdown["n_scored"], 0)
        self.assertIsNone(breakdown["best"])
        self.assertFalse(breakdown["penalty_known"])

    def test_best_clean_only_comes_from_zero_penalty_records(self):
        """回归：干净解**不能**从带罚分的记录里凑（那会把两类数字混起来）。

        这里让**总体最高分**恰恰是带罚分的那个（score −0.15 优于干净解的 −5），
        于是"最好干净解"必须是另一条记录：任何回退到 ``best`` 的实现都会断言失败。
        """
        breakdown = records_mod.score_breakdown([
            self._record(1, 0.05, 0.1),       # 分数最高，但由拟合+罚分共同构成
            self._record(2, 5.0, 0.0),        # 干净解，拟合差
        ])
        self.assertEqual(breakdown["best"]["sample_order"], 1)
        self.assertEqual(breakdown["best_clean"]["sample_order"], 2)
        self.assertEqual(breakdown["best_penalized"]["sample_order"], 1)
        self.assertEqual(breakdown["best_fit"]["sample_order"], 1)
        self.assertEqual(breakdown["n_penalized"], 1)
        self.assertEqual(breakdown["n_clean"], 1)

    def test_missing_penalty_is_unknown_not_zero(self):
        """旧目录没有 penalty 字段：罚分是**未知**，不许当成 0（那会被读成"干净"）。"""
        breakdown = records_mod.score_breakdown([self._record(1, 12.0, None)])
        self.assertTrue(breakdown["n_scored"] == 1)
        self.assertFalse(breakdown["penalty_known"])
        self.assertIsNone(breakdown["best_clean"])
        self.assertIsNone(breakdown["best_penalized"])

    def test_no_clean_record_leaves_best_clean_empty(self):
        breakdown = records_mod.score_breakdown([self._record(1, 0.2, 1.0),
                                                self._record(2, 0.3, 2.0)])
        self.assertIsNone(breakdown["best_clean"])
        self.assertEqual(breakdown["n_clean"], 0)
        self.assertEqual(breakdown["n_penalized"], 2)


class PublishGateTest(unittest.TestCase):
    """病理门禁：优先发布无病理的最高分样本，并把被跳过的候选记下来。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)

    def test_prefers_clean_candidate_over_higher_scoring_pathological(self):
        _write(self.root, "top01_samples_7.json", _sample(-0.5, 2.0, 7))    # 分高但有病理
        _write(self.root, "top02_samples_9.json", _sample(-1.0, 0.0, 9))    # 干净
        chosen, info = fbe.select_published_sample(str(self.root))
        self.assertEqual(chosen["sample_order"], 9)
        self.assertTrue(info["degraded"])
        self.assertEqual(info["n_clean"], 1)
        self.assertEqual([r["sample_order"] for r in info["rejected"]], [7])
        self.assertEqual(info["best"]["sample_order"], 7)

    def test_no_clean_candidate_keeps_best_and_flags_it(self):
        _write(self.root, "top01_samples_7.json", _sample(-0.5, 2.0, 7))
        _write(self.root, "top02_samples_9.json", _sample(-1.0, 3.0, 9))
        chosen, info = fbe.select_published_sample(str(self.root))
        self.assertEqual(chosen["sample_order"], 7)
        self.assertEqual(info["n_clean"], 0)
        self.assertFalse(info["degraded"])

    def test_unknown_penalty_is_neither_clean_nor_pathological(self):
        """旧产物缺拟合 MSE → 罚分 None：不能当干净解，也不能诬指为病理。"""
        _write(self.root, "top01_samples_4.json", _sample(-0.4, None, 4))
        _write(self.root, "top02_samples_5.json", _sample(-1.0, 0.0, 5))
        chosen, info = fbe.select_published_sample(str(self.root))
        self.assertEqual(chosen["sample_order"], 5)
        self.assertTrue(info["degraded"])
        self.assertEqual(info["n_rejected"], 0)          # 不是"因病理被跳过"
        self.assertEqual(info["n_unknown_skipped"], 1)

    def test_no_samples_returns_none(self):
        chosen, info = fbe.select_published_sample(str(self.root))
        self.assertIsNone(chosen)
        self.assertEqual(info["n_candidates"], 0)


class SelectionSectionTest(unittest.TestCase):
    """report.md 的「发布解选择」小节必须把降级与理由写出来。"""

    def _render(self, selection):
        return explain_mod.render_selection_section(selection)

    def test_degraded_case_names_both_entries(self):
        text = self._render({
            "n_candidates": 12, "n_clean": 3, "degraded": True,
            "best": {"sample_order": 40, "score": -11.999, "mse": 0.243752,
                     "penalty": 11.755},
            "chosen": {"sample_order": 52, "score": -1.113, "mse": 1.1, "penalty": 0.0},
            "rejected": [{"sample_order": 40, "score": -11.999, "mse": 0.243752,
                          "penalty": 11.755}],
            "n_rejected": 1, "n_unknown_skipped": 0,
        })
        self.assertIn(explain_mod.SELECTION_HEADING, text)
        self.assertIn("本次发生了降级", text)
        self.assertIn("sample_order=52", text)
        self.assertIn("因病理被跳过（分数更高）：sample_order=40", text)
        self.assertIn("11.755", text)

    def test_clean_case_says_no_degradation(self):
        text = self._render({"n_candidates": 5, "n_clean": 5, "degraded": False,
                             "best": {"sample_order": 1, "score": -1.0, "mse": 1.0,
                                      "penalty": 0.0},
                             "chosen": {"sample_order": 1, "score": -1.0, "mse": 1.0,
                                        "penalty": 0.0},
                             "rejected": [], "n_rejected": 0, "n_unknown_skipped": 0})
        self.assertIn("未发生降级", text)
        self.assertNotIn("发生了降级", text)

    def test_all_pathological_says_not_usable(self):
        text = self._render({"n_candidates": 3, "n_clean": 0, "degraded": False,
                             "best": {"sample_order": 2, "score": -1.0, "mse": 0.1,
                                      "penalty": 4.0},
                             "chosen": {"sample_order": 2, "score": -1.0, "mse": 0.1,
                                        "penalty": 4.0},
                             "rejected": [], "n_rejected": 0, "n_unknown_skipped": 0})
        self.assertIn("没有任何无病理候选", text)

    def test_missing_selection_renders_the_reason_instead_of_nothing(self):
        """缺选解信息时必须**留下标题 + 一行原因**，不能让整节消失（缺陷 ②）。

        实测 `ab-fix6-control/...20260928-154613` 与 `ab-iso6-no6/...20260929-091844`
        两份报告整节不见，读者与收尾脚本会把「没这一节」读成「这次没做体检门禁」；
        真实成因是选解所用 best 样本的 return 表达式无法解析（同目录 run.out 里有
        `[WARN] return 表达式解析失败`）。故缺失态要显式点名去查 WARN。
        """
        for empty in (None, {}):
            with self.subTest(selection=empty):
                block = self._render(empty)
                self.assertTrue(block.startswith(explain_mod.SELECTION_HEADING))
                self.assertIn("本节没有选解信息", block)
                self.assertIn("不等于", block)                     # 不是"没做门禁"
                self.assertIn("[WARN] return 表达式解析失败", block)  # 指向真正原因


class ReportAlwaysWrittenTest(unittest.TestCase):
    """物理解释生成失败时也必须留下 report.md（除非目录里已有一份可保留的报告）。"""

    FUNC = ("Variables:\n"
            "- Independents: lambda12, lambda23\n"
            "- Dependent: sigma\n"
            "def equation_v1(lambda12, lambda23, params):\n"
            "    return params[0] * lambda12 + params[1]")

    CLEAN_RANGE = {"span_ratio": 1.2, "grid_min": 193.0, "grid_max": 352.5,
                   "span_penalty": 0.0, "slope_max": 0.5, "slope_limit": 2.0,
                   "slope_penalty": 0.0, "penalty": 0.0, "limit": 15.0, "n_points": 640}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)

    def test_report_written_when_no_matching_good_experience(self):
        """没匹配到 Good 经验也要出报告：机器小节照写 + 开头写明原因。"""
        (self.root / "experiences.json").write_text(
            json.dumps({"Good": [{"sample_order": 99, "analysis": "x"}]}), encoding="utf-8")
        pruning = {"range_check": self.CLEAN_RANGE,
                   "selection": {"n_candidates": 1, "n_clean": 1, "degraded": False,
                                 "best": {"sample_order": 5, "score": -1.0, "mse": 1.0,
                                          "penalty": 0.0},
                                 "chosen": {"sample_order": 5, "score": -1.0, "mse": 1.0,
                                            "penalty": 0.0},
                                 "rejected": [], "n_rejected": 0,
                                 "n_unknown_skipped": 0}}
        with mock.patch.object(explain_mod, "retrieve_rag", return_value=[]), \
             mock.patch.object(explain_mod, "explain_re_act",
                               lambda *a, **k: (_ for _ in ()).throw(
                                   AssertionError("无匹配经验时不应调用 LLM"))):
            explain_mod.explain_best_sample(str(self.root), self.FUNC, "5",
                                            pruning=pruning)
        text = (self.root / explain_mod.REPORT_FILENAME).read_text(encoding="utf-8")
        self.assertIn("物理解释未生成", text)
        self.assertIn("未找到 sample_order=5 的 Good 经验", text)
        self.assertIn(explain_mod.SELECTION_HEADING, text)     # 机器小节照写
        self.assertIn(explain_mod.RANGE_HEADING, text)

    def test_report_written_even_without_experiences_file(self):
        with mock.patch.object(explain_mod, "retrieve_rag", return_value=[]):
            explain_mod.explain_best_sample(str(self.root), self.FUNC, "5")
        self.assertTrue((self.root / explain_mod.REPORT_FILENAME).exists())


class _FuncStub:
    """Profiler.register_function 需要的最小样本替身。"""

    def __init__(self, name, body, order, score=None):
        self.name = name
        self.body = body
        self.global_sample_nums = order
        self.score = score
        self.sample_time = 0.1
        self.evaluate_time = 0.2
        self.optimized_params = None
        self.fit_mse = None

    def __str__(self):
        return f"def {self.name}():\n  {self.body}"


class FullSampleRetentionTest(unittest.TestCase):
    """默认必须**全量落盘**：只留 top-10 会让整份搜索轨迹不可追溯（490 个 run 的实测）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.prof = Profiler(self._tmp.name)

    def test_every_sample_is_persisted_by_default(self):
        for order, score in ((1, -5.0), (2, -3.0), (3, -1.0)):
            self.prof.register_function(_FuncStub("equation", "return 1", order, score=score))
        samples = os.path.join(self._tmp.name, "samples")
        for order in (1, 2, 3):
            self.assertTrue(os.path.exists(os.path.join(samples, f"samples_{order}.json")),
                            f"缺失 samples_{order}.json——默认落盘没有生效")
        # Top-K 排行文件照旧存在（收尾分析依赖它）
        self.assertTrue(os.path.exists(os.path.join(samples, "top01_samples_3.json")))


class PruningSummarySchemaTest(unittest.TestCase):
    """剪枝摘要的**键集合**是声明式契约：写错一个键只会静默取到 None。

    ``prune_and_visualize`` 的返回值被 ``explain``（提示词块 + report.md 小节）与多个
    测试按字符串键消费，键名原先靠"约定对齐"。``PRUNING_SUMMARY_KEYS`` 把它变成可断言
    的对象，这里比对"实际返回的键 == 声明的键"，防止新增/改名时漏改消费方。
    """

    FUNC = ("Variables:\n"
            "- Independents: x1, x2\n"
            "- Dependent: y\n"
            "def equation_v1(x1, x2, params):\n"
            "    return params[0] * x1 + params[1] * x2")

    def test_returned_keys_match_declared_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "p_20260101-000000"
            root.mkdir()
            summary = fbe.prune_and_visualize(
                str(root), self.FUNC, [2.0, 3.0], threshold=0.1,
                sample_range=(1, 6), test_csv="none")
        self.assertIsNotNone(summary)
        self.assertEqual(sorted(summary), sorted(fbe.PRUNING_SUMMARY_KEYS))

    def test_selection_is_part_of_the_summary_not_filled_in_afterwards(self):
        """``selection`` 由调用方在调用时传入，不再返回后回填（旧写法见其 docstring）。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "p_20260101-000000"
            root.mkdir()
            marker = {"n_candidates": 0, "n_clean": 0, "degraded": False, "best": None,
                      "chosen": None, "rejected": [], "n_rejected": 0,
                      "n_unknown_skipped": 0}
            summary = fbe.prune_and_visualize(
                str(root), self.FUNC, [2.0, 3.0], threshold=0.1,
                sample_range=(1, 6), test_csv="none", selection=marker)
        self.assertIs(summary["selection"], marker)

    def test_unknown_pipeline_option_is_reported_not_silently_dropped(self):
        """``PipelineOptions`` 让"键名写错"从静默变成可诊断（原先只有 kwargs.get）。"""
        from drsr_420.runtime.pipeline import PipelineOptions

        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            options = PipelineOptions.from_kwargs({"test_csv": "a.csv", "test_cvs": "typo"})
            printed = buf.getvalue()
        self.assertEqual(options.test_csv, "a.csv")
        self.assertIn("test_cvs", printed)          # 拼错的键被点名
        # cli 会传但本层不用的键不应刷告警
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            PipelineOptions.from_kwargs({"llm_config": {"model": "m"}})
            self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()