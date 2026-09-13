"""Agent 间消息契约测试（阶段 2：`drsr_420/agents/messages.py`）。

覆盖两件事：

1. **落盘字段形状**：``experiences.json`` / ``residual_analyze.json`` / ``samples_N.json``
   的字段名是历史产物与既有分析脚本依赖的公开契约，必须逐字保持；
2. **一轮协作的真实数据流**：用真实的 CoordinatorAgent / ExperienceSummarizerAgent /
   ResidualAnalyzerAgent + 真实的 ExperienceBuffer/Profiler，只把"LLM 与评估器"
   替换成替身，跑完整一轮 ``sample()``，断言消息里的每个字段都落到了正确的位置。

这正是阶段 2 要修掉的隐患：原先靠裸元组 ``(score, error, residual)`` 与
``**kwargs`` 袋传数据、靠 ``zip`` 对齐 5 个平行列表，字段错位不会报错。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from drsr_420.core import buffer as buffer_lib
from drsr_420.core import code_manipulation as cm
from drsr_420.core import config as config_lib
from drsr_420.core import profile as profile_lib
from drsr_420.agents.coordinator_agent import CoordinatorAgent
from drsr_420.agents.experience_summarizer_agent import ExperienceSummarizerAgent
from drsr_420.agents.messages import (
    QUALITY_BAD,
    QUALITY_GOOD,
    QUALITY_NONE,
    EvaluationOutcome,
    EvaluationRequest,
    ExperienceEntry,
    ResidualInsight,
    SampleBatch,
    check_alignment,
)
from drsr_420.agents.residual_analyzer_agent import ResidualAnalyzerAgent

TEMPLATE = (
    "import numpy as np\n"
    "@equation.evolve\n"
    "def equation(x, params):\n"
    "    return params[0] * x\n"
)

S_BAD = "def equation(x, params):\n    return params[0] * x + 1\n"
S_NONE = "def equation(x, params):\n    return params[0] * x + 2\n"
S_GOOD = "def equation(x, params):\n    return params[1] * x + 3\n"

RESIDUAL_GOOD = np.arange(6, dtype=float).reshape(3, 2)


# ----------------------------------------------------------------------
# 替身：只替换 LLM 与评估器（其余全是真的）
# ----------------------------------------------------------------------
class _FakeLLMClient:
    """满足 coordinator 的 clone_for_task + chat 契约的最小替身。"""

    def __init__(self):
        self.kwargs = {}

    def clone_for_task(self, task):
        clone = _FakeLLMClient()
        clone.kwargs = dict(self.kwargs)
        clone.task = task
        return clone

    def chat(self, messages, on_delta=None):
        return {"content": "FAKE-ANALYSIS", "reasoning_content": ""}


class _FakeSampler:
    """替代 SamplerAgent：固定返回 3 个骨架。"""

    def __init__(self, samples_per_prompt, prompt_ctx=None, llm_client=None):
        self._samples = [S_BAD, S_NONE, S_GOOD][:samples_per_prompt]

    def draw_samples(self, prompt, config):
        return list(self._samples), ["" for _ in self._samples]


class _FakeEvaluator:
    """替代 EvaluatorAgent：按样本内容返回预置结果，并记录收到的请求。"""

    def __init__(self, outcomes):
        self._outcomes = outcomes
        self.requests: list[EvaluationRequest] = []

    def analyze(self, request: EvaluationRequest) -> EvaluationOutcome:
        self.requests.append(request)
        return self._outcomes[request.sample]


def _outcomes():
    return {
        S_BAD: EvaluationOutcome(score=-0.5, error=None, residual=None),
        S_NONE: EvaluationOutcome(score=None, error="Execution Error: boom", residual=None),
        S_GOOD: EvaluationOutcome(score=-0.1, error=None, residual=RESIDUAL_GOOD),
    }


class MessageShapeTest(unittest.TestCase):
    """落盘字段名 = 公开契约，逐字锁定。"""

    def test_experience_entry_json_keys(self):
        entry = ExperienceEntry(
            sample="CODE", quality=QUALITY_GOOD, analysis="A",
            error=None, score=-0.5, thinking_content="T",
            island_id=2, sample_order=7, sample_time=1.5)
        self.assertEqual(
            set(entry.to_json()),
            {"island_id", "analysis", "sample_order", "sample_time",
             "equation", "score", "thinking_content"},
        )
        self.assertNotIn("error", entry.to_json())   # 仅 None 类别带 error

    def test_experience_entry_error_only_for_none_quality(self):
        none_entry = ExperienceEntry(
            sample="CODE", quality=QUALITY_NONE, analysis="A", error="boom")
        self.assertEqual(none_entry.to_json()["error"], "boom")

        bad_entry = ExperienceEntry(
            sample="CODE", quality=QUALITY_BAD, analysis="A", error="boom")
        self.assertNotIn("error", bad_entry.to_json())

    def test_experience_entry_rejects_unknown_quality(self):
        with self.assertRaises(ValueError):
            ExperienceEntry(sample="CODE", quality="Excellent", analysis="A")

    def test_residual_insight_json_keys(self):
        insight = ResidualInsight(sample="CODE", analysis="A", island_id=0,
                                  sample_order=3, best_score=-0.1)
        self.assertEqual(
            set(insight.to_json()),
            {"sample_order", "island_id", "equation", "analysis", "best_score"},
        )
        self.assertEqual(insight.to_json()["equation"], "CODE")

    def test_evaluation_outcome_legacy_interop(self):
        outcome = EvaluationOutcome(score=-1.0, error="e", residual=[1, 2])
        self.assertEqual(outcome.to_legacy(), (-1.0, "e", [1, 2]))
        score, error, residual = outcome          # 旧式解包仍可用
        self.assertEqual((score, error, residual), (-1.0, "e", [1, 2]))

    def test_sample_batch_is_reexported_from_coordinator(self):
        """SampleBatch 迁到 messages 后，旧导入路径必须仍指向同一个类。"""
        import drsr_420.agents.coordinator_agent as coord_mod
        self.assertIs(coord_mod.SampleBatch, SampleBatch)

    def test_check_alignment_rejects_mismatched_lists(self):
        with self.assertRaises(ValueError):
            check_alignment(["a", "b"], ["Good"])       # 2 vs 1
        check_alignment(["a"], ["Good"], [None])        # 一致则不报错


class CoordinatorRoundTripTest(unittest.TestCase):
    """真实一轮 sample()：消息字段 → 落盘 JSON 的端到端一致性。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        # 全局采样计数是类属性：显式重置，避免与其它测试的执行顺序耦合
        CoordinatorAgent.set_global_sample_nums(1)

    def _run_one_round(self):
        eb_cfg = config_lib.ExperienceBufferConfig(num_islands=2)
        template = cm.text_to_program(TEMPLATE)
        database = buffer_lib.ExperienceBuffer(eb_cfg, template, "equation")
        # ExperienceBuffer 初始全为空岛：生产流程由 pipeline 先评估初始模板并注册，
        # 这里显式喂一个种子程序（只占岛屿 0 → 选岛确定），其分数 -0.3 决定了
        # 本轮三个样本的 Good/Bad/None 归属（见 _outcomes()）。
        database.register_program(
            cm.text_to_function("def equation(x, params):\n    return params[0] * x\n"),
            0, {"data": -0.3})

        exp_config = config_lib.Config(
            results_root=self.root,
            samples_per_prompt=3,
            experience_buffer=eb_cfg,
        )
        profiler = profile_lib.Profiler(
            self.root, samples_per_iteration=3, persist_all_samples=True)

        fake_client = _FakeLLMClient()
        evaluator = _FakeEvaluator(_outcomes())
        coord = CoordinatorAgent(
            database,
            [evaluator],
            3,
            config=exp_config,
            max_sample_nums=3,           # 一轮 3 个样本 → 跑完一轮即停
            llm_class=_FakeSampler,
            llm_client=fake_client,
        )
        coord.sample(profiler=profiler)
        return coord, evaluator

    def _load(self, name):
        with open(os.path.join(self.root, name), "r", encoding="utf-8") as f:
            return json.load(f)

    def test_one_round_writes_expected_experiences(self):
        coord, evaluator = self._run_one_round()

        # 评估请求必须是显式 EvaluationRequest（不再是 kwargs 袋）
        self.assertEqual(len(evaluator.requests), 3)
        for request in evaluator.requests:
            self.assertIsInstance(request, EvaluationRequest)
            self.assertEqual(request.sample_time is not None, True)
            self.assertIsInstance(request.global_sample_nums, int)

        experiences = self._load("experiences.json")
        self.assertEqual(len(experiences[QUALITY_GOOD]), 1)
        self.assertEqual(len(experiences[QUALITY_BAD]), 1)
        self.assertEqual(len(experiences[QUALITY_NONE]), 1)

        good = experiences[QUALITY_GOOD][0]
        self.assertEqual(good["equation"], S_GOOD)
        self.assertEqual(good["score"], -0.1)
        self.assertEqual(good["analysis"], "FAKE-ANALYSIS")
        self.assertEqual(good["island_id"], evaluator.requests[0].island_id)
        self.assertEqual(good["thinking_content"], "")

        bad = experiences[QUALITY_BAD][0]
        self.assertEqual(bad["equation"], S_BAD)
        self.assertEqual(bad["score"], -0.5)
        self.assertNotIn("error", bad)     # Bad 类别不带 error

        none_rec = experiences[QUALITY_NONE][0]
        self.assertEqual(none_rec["equation"], S_NONE)
        self.assertIsNone(none_rec["score"])
        self.assertEqual(none_rec["error"], "Execution Error: boom")

        # sample_order 必须与本轮全局计数一致（连续、1-based）
        orders = sorted(r["sample_order"] for cat in
                        (QUALITY_GOOD, QUALITY_BAD, QUALITY_NONE)
                        for r in experiences[cat])
        self.assertEqual(orders, [2, 3, 4])   # 全局计数从 1 起，本轮 3 个样本

    def test_one_round_writes_residual_insight_for_best_sample(self):
        self._run_one_round()
        records = self._load("residual_analyze.json")
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(set(record), {"sample_order", "island_id", "equation",
                                       "analysis", "best_score"})
        self.assertEqual(record["equation"], S_GOOD)      # 本轮最优样本
        self.assertEqual(record["best_score"], -0.1)
        self.assertEqual(record["analysis"], "FAKE-ANALYSIS")
        self.assertEqual(record["sample_order"], 4)

    def test_checkpoint_and_progress_are_written(self):
        self._run_one_round()
        self.assertTrue(os.path.exists(os.path.join(self.root, "checkpoint.json")))
        self.assertTrue(os.path.exists(os.path.join(self.root, "round_progress.csv")))
        with open(os.path.join(self.root, "checkpoint.json"), "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        self.assertEqual(ckpt["global_sample_nums"], 4)


class EvaluationOutcomeToSamplesJsonTest(unittest.TestCase):
    """EvaluationOutcome.score 必须原样落到 samples_N.json 的 score 字段。"""

    def test_outcome_score_reaches_samples_json(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = config_lib.ExperienceBufferConfig(num_islands=1)
            template = cm.text_to_program(TEMPLATE)
            database = buffer_lib.ExperienceBuffer(cfg, template, "equation")
            profiler = profile_lib.Profiler(
                root, samples_per_iteration=1, persist_all_samples=True)

            outcome = EvaluationOutcome(score=-0.25, error=None, residual=None)
            func = cm.text_to_function(
                "def equation(x, params):\n    return params[0] * x\n")

            # 与 EvaluatorAgent.analyze 内部调用完全一致的关键字参数
            database.register_program(
                func, 0, {"data": outcome.score},
                profiler=profiler, global_sample_nums=5,
                sample_time=1.25, evaluate_time=0.5)

            path = os.path.join(root, "samples", "samples_5.json")
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
            self.assertEqual(content["score"], outcome.score)
            self.assertEqual(content["sample_order"], 5)
            self.assertEqual(set(content),
                             {"iteration", "sample_order", "nmse", "mse", "score", "function"})


class SummarizerAlignmentTest(unittest.TestCase):
    """平行列表长度不一致必须报错，而不是静默错配样本与经验。"""

    def test_mismatched_lengths_raise_before_calling_llm(self):
        summarizer = ExperienceSummarizerAgent(_FakeLLMClient())
        with self.assertRaises(ValueError):
            summarizer.analyze(["s1", "s2"], ["Good"], [None], prompt="p")

    def test_returns_experience_entries(self):
        summarizer = ExperienceSummarizerAgent(_FakeLLMClient())
        entries = summarizer.analyze(
            ["s1", "s2"], ["Good", "None"], [None, "boom"], prompt="p")
        self.assertEqual(len(entries), 2)
        self.assertTrue(all(isinstance(e, ExperienceEntry) for e in entries))
        self.assertEqual([e.quality for e in entries], ["Good", "None"])
        self.assertEqual(entries[0].analysis, "FAKE-ANALYSIS")
        self.assertEqual(entries[1].error, "boom")


class ResidualAnalyzerInsightTest(unittest.TestCase):
    """残差分析者返回 ResidualInsight（归属字段留空给 coordinator 补）。"""

    def test_returns_insight_without_attribution(self):
        analyzer = ResidualAnalyzerAgent(
            _FakeLLMClient(), results_root=tempfile.gettempdir())
        insight = analyzer.analyze(S_GOOD, RESIDUAL_GOOD)
        self.assertIsInstance(insight, ResidualInsight)
        self.assertEqual(insight.sample, S_GOOD)
        self.assertEqual(insight.analysis, "FAKE-ANALYSIS")
        self.assertIsNone(insight.island_id)
        self.assertIsNone(insight.sample_order)
        self.assertIsNone(insight.best_score)


if __name__ == "__main__":
    unittest.main()
