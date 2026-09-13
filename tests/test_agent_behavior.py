"""Agent 层行为测试（工具循环 / 反思 / 分析 / 协调）。

与 tests/test_messages.py 的分工：那边锁定"消息与落盘字段的形状"，这边锁定
**每个角色在真实调用链路里的行为**——工具循环怎么收敛、反思结果怎么兜底、
协调者怎么判定 Good/Bad/None、落盘字段怎么算出来、失败路径是否真的不把主循环带崩。

全部用替身（假 LLM 客户端 / 假缓冲 / 假评估器），不触网、不依赖真实模型。
"""
import json
import os
import tempfile
import time
import unittest

import numpy as np

from drsr_420.agents.coordinator_agent import CoordinatorAgent
from drsr_420.agents.data_analyzer_agent import DataAnalyzerAgent
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
)
from drsr_420.agents.residual_analyzer_agent import ResidualAnalyzerAgent
from drsr_420.agents.tool_caller_agent import ToolCallerAgent
from drsr_420.core import buffer as buffer_lib
from drsr_420.core import config as config_lib


# ----------------------------------------------------------------------
# 替身
# ----------------------------------------------------------------------
class _ScriptedChat:
    """按脚本返回响应的假 LLM 客户端，并记录每次收到的消息。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.kwargs = {}
        self.sent: list[list[dict]] = []

    def chat(self, messages, on_delta=None):
        self.sent.append([dict(m) for m in messages])
        if not self._responses:
            raise AssertionError("chat 调用次数超出预置脚本")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if on_delta is not None:
            on_delta({"content": item.get("content", ""),
                      "reasoning_content": item.get("reasoning_content", "")})
        return item


def _tool_call(name="search_kb", arguments='{"query": "x"}', call_id="c1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


# ----------------------------------------------------------------------
# ToolCallerAgent：多轮工具循环
# ----------------------------------------------------------------------
class ToolCallerLoopTest(unittest.TestCase):
    def _agent(self, client, executor=None, max_tool_rounds=4):
        return ToolCallerAgent(
            client,
            tool_executor=executor if executor is not None else (lambda name, args: "RESULT"),
            max_tool_rounds=max_tool_rounds,
        )

    def test_repeat_returns_parallel_lists(self):
        client = _ScriptedChat([{"content": "A", "reasoning_content": "rA"},
                                {"content": "B", "reasoning_content": "rB"}])
        responses, thinking = self._agent(client).complete("Q", repeat=2)
        self.assertEqual(responses, ["A", "B"])
        self.assertEqual(thinking, ["rA", "rB"])
        self.assertEqual(len(client.sent), 2)

    def test_reasoning_is_used_when_content_is_empty(self):
        client = _ScriptedChat([{"content": "", "reasoning_content": "THINK"}])
        responses, thinking = self._agent(client).complete("Q")
        self.assertEqual(responses, ["THINK"])
        self.assertEqual(thinking, ["THINK"])

    def test_tool_result_is_fed_back_to_model(self):
        seen = []

        def executor(name, args):
            seen.append((name, args))
            return "KB-HITS"

        client = _ScriptedChat([
            {"content": "let me search", "reasoning_content": "",
             "tool_calls": [_tool_call()]},
            {"content": "FINAL", "reasoning_content": ""},
        ])
        responses, _thinking = self._agent(client, executor).complete("Q")

        self.assertEqual(responses, ["FINAL"])
        self.assertEqual(seen, [("search_kb", {"query": "x"})])
        # 第二轮请求必须带上 assistant 的 tool_calls 与 role=tool 的执行结果
        second_round = client.sent[1]
        self.assertEqual(second_round[-1]["role"], "tool")
        self.assertEqual(second_round[-1]["tool_call_id"], "c1")
        self.assertEqual(second_round[-1]["content"], "KB-HITS")
        self.assertEqual(second_round[-2]["tool_calls"][0]["function"]["name"], "search_kb")

    def test_malformed_arguments_fall_back_to_empty_dict(self):
        seen = []
        client = _ScriptedChat([
            {"content": "", "reasoning_content": "",
             "tool_calls": [_tool_call(arguments="{not json")]},
            {"content": "FINAL", "reasoning_content": ""},
        ])
        self._agent(client, lambda n, a: seen.append(a) or "R").complete("Q")
        self.assertEqual(seen, [{}])

    def test_tool_round_limit_forces_final_answer(self):
        client = _ScriptedChat([
            {"content": f"round-{i}", "reasoning_content": "", "tool_calls": [_tool_call()]}
            for i in range(5)
        ])
        calls = []
        responses, _thinking = self._agent(
            client, lambda n, a: calls.append(n) or "R", max_tool_rounds=2).complete("Q")

        self.assertEqual(len(client.sent), 2)          # 只允许两轮工具调用
        self.assertEqual(len(calls), 2)
        self.assertEqual(responses, ["round-1"])       # 用第二轮响应作为最终结果

    def test_executor_exception_yields_empty_response(self):
        """工具执行器抛异常时不炸整批：该样本退化为空响应（上层会重采样）。"""
        def boom(name, args):
            raise RuntimeError("mcp down")

        client = _ScriptedChat([{"content": "", "reasoning_content": "",
                                 "tool_calls": [_tool_call()]}])
        responses, thinking = self._agent(client, boom).complete("Q")
        self.assertEqual(responses, [""])
        self.assertEqual(thinking, [""])


# ----------------------------------------------------------------------
# ExperienceSummarizerAgent：逐样本反思
# ----------------------------------------------------------------------
class SummarizerBehaviorTest(unittest.TestCase):
    def _run(self, client, samples, qualities, errors, prompt=None):
        agent = ExperienceSummarizerAgent(client)
        return agent.analyze(samples, qualities, errors,
                             prompt if prompt is not None else "PROMPT-CODE")

    def test_one_entry_per_sample_carrying_quality_and_error(self):
        client = _ScriptedChat([{"content": f"analysis-{i}", "reasoning_content": ""}
                                for i in range(3)])
        entries = self._run(client, ["s0", "s1", "s2"],
                            [QUALITY_GOOD, QUALITY_BAD, QUALITY_NONE],
                            [None, "bad error", "boom"])

        self.assertEqual([e.quality for e in entries],
                         [QUALITY_GOOD, QUALITY_BAD, QUALITY_NONE])
        self.assertEqual([e.sample for e in entries], ["s0", "s1", "s2"])
        self.assertEqual([e.analysis for e in entries],
                         ["analysis-0", "analysis-1", "analysis-2"])
        self.assertIsNone(entries[0].error)
        self.assertEqual(entries[2].error, "boom")

    def test_reasoning_fallback_when_content_empty(self):
        client = _ScriptedChat([{"content": "", "reasoning_content": "REASONED"}])
        entries = self._run(client, ["s0"], [QUALITY_GOOD], [None])
        self.assertEqual(entries[0].analysis, "REASONED")

    def test_llm_failure_is_captured_instead_of_raising(self):
        client = _ScriptedChat([RuntimeError("api down")])
        entries = self._run(client, ["s0"], [QUALITY_BAD], [None])
        self.assertEqual(len(entries), 1)
        self.assertIn("分析请求发生错误", entries[0].analysis)
        self.assertIn("api down", entries[0].analysis)

    def test_question_differs_by_quality_and_none_carries_error(self):
        client = _ScriptedChat([{"content": "x", "reasoning_content": ""} for _ in range(3)])
        self._run(client, ["s0", "s1", "s2"],
                  [QUALITY_GOOD, QUALITY_BAD, QUALITY_NONE], [None, None, "ERR-XYZ"])

        prompts = [msgs[1]["content"] for msgs in client.sent]
        self.assertEqual(len(set(prompts)), 3, "三类样本的分析问题不应完全相同")
        self.assertIn("ERR-XYZ", prompts[2])
        self.assertNotIn("ERR-XYZ", prompts[0])

    def test_prompt_object_code_is_embedded(self):
        client = _ScriptedChat([{"content": "x", "reasoning_content": ""}])
        prompt = buffer_lib.Prompt(code="HEADER-CODE", version_generated=1, island_id=0)
        self._run(client, ["s0"], [QUALITY_GOOD], [None], prompt=prompt)
        self.assertIn("HEADER-CODE", client.sent[0][1]["content"])

    def test_prompt_context_question_is_preferred(self):
        class Ctx:
            def __init__(self):
                self.calls = []

            def render_analysis_question(self, quality, error):
                self.calls.append((quality, error))
                return "CTX-QUESTION"

        ctx = Ctx()
        client = _ScriptedChat([{"content": "x", "reasoning_content": ""} for _ in range(2)])
        ExperienceSummarizerAgent(client, prompt_ctx=ctx).analyze(
            ["s0", "s1"], [QUALITY_GOOD, QUALITY_NONE], [None, "boom"], "CODE")

        self.assertIn("CTX-QUESTION", client.sent[0][1]["content"])
        # error 只在 None 类别传下去（Good 样本不该看到失败原因）
        self.assertEqual(ctx.calls, [(QUALITY_GOOD, None), (QUALITY_NONE, "boom")])


# ----------------------------------------------------------------------
# ResidualAnalyzerAgent：残差洞察
# ----------------------------------------------------------------------
class ResidualAnalyzerBehaviorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.residual = np.array([[1.0, 2.0, 0.5], [2.0, 4.0, -0.25]])

    def _write_previous(self, data):
        with open(os.path.join(self.root, "residual_analyze.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def test_returns_insight_without_attribution_fields(self):
        client = _ScriptedChat([{"content": "INSIGHT", "reasoning_content": ""}])
        insight = ResidualAnalyzerAgent(client, results_root=self.root).analyze("CODE", self.residual)

        self.assertIsInstance(insight, ResidualInsight)
        self.assertEqual(insight.sample, "CODE")
        self.assertEqual(insight.analysis, "INSIGHT")
        self.assertIsNone(insight.island_id)      # 归属字段由协调者补齐
        self.assertIsNone(insight.sample_order)

    def test_previous_analysis_is_used_as_context(self):
        self._write_previous([{"analysis": "PREVIOUS-LESSON", "sample_order": 3}])
        client = _ScriptedChat([{"content": "INSIGHT", "reasoning_content": ""}])
        ResidualAnalyzerAgent(client, results_root=self.root).analyze("CODE", self.residual)

        self.assertIn("PREVIOUS-LESSON", client.sent[0][1]["content"])

    def test_residual_and_sample_are_sent_to_model(self):
        client = _ScriptedChat([{"content": "INSIGHT", "reasoning_content": ""}])
        ResidualAnalyzerAgent(client, results_root=self.root).analyze("SAMPLE-CODE", self.residual)

        sent = client.sent[0][1]["content"]
        self.assertIn("SAMPLE-CODE", sent)
        self.assertIn("0.5", sent)          # 残差矩阵进入提示词

    def test_corrupt_previous_file_does_not_break_analysis(self):
        with open(os.path.join(self.root, "residual_analyze.json"), "w", encoding="utf-8") as f:
            f.write("{ broken")
        client = _ScriptedChat([{"content": "INSIGHT", "reasoning_content": ""}])
        insight = ResidualAnalyzerAgent(client, results_root=self.root).analyze("CODE", self.residual)
        self.assertEqual(insight.analysis, "INSIGHT")

    def test_llm_failure_is_captured(self):
        client = _ScriptedChat([RuntimeError("api down")])
        insight = ResidualAnalyzerAgent(client, results_root=self.root).analyze("CODE", self.residual)
        self.assertIn("分析请求发生错误", insight.analysis)
        self.assertIn("api down", insight.analysis)

    def test_results_root_defaults_to_cwd(self):
        agent = ResidualAnalyzerAgent(_ScriptedChat([]), results_root=None)
        self.assertEqual(agent._results_root, ".")


# ----------------------------------------------------------------------
# DataAnalyzerAgent：初次数据分析与落盘
# ----------------------------------------------------------------------
class _RecordingClient:
    """假客户端：记录 prompt 与克隆，clone_for_task 返回同样记录的新实例。"""

    def __init__(self, content="INITIAL-ANALYSIS", reasoning=""):
        self.kwargs = {}
        self.content = content
        self.reasoning = reasoning
        self.sent: list[list[dict]] = []
        self.clones: list["_RecordingClient"] = []

    def clone_for_task(self, task):
        clone = _RecordingClient(self.content, self.reasoning)
        clone.kwargs = dict(self.kwargs)
        clone.task = task
        self.clones.append(clone)
        return clone

    def chat(self, messages, on_delta=None):
        self.sent.append([dict(m) for m in messages])
        return {"content": self.content, "reasoning_content": self.reasoning}


class DataAnalyzerBehaviorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.csv = os.path.join(self.root, "train.csv")
        with open(self.csv, "w", encoding="utf-8") as f:
            f.write("x,y\n")
            for i in range(10):
                f.write(f"{i}.123456,{i * 2}.987654\n")

    def _records(self):
        with open(os.path.join(self.root, "residual_analyze.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    def test_csv_is_rounded_and_sampled(self):
        agent = DataAnalyzerAgent(base_dir=self.root, decimal_places=2, sample_size=3, seed=0)
        text = agent._read_csv_data(self.csv)
        rows = [ln for ln in text.strip().splitlines()[1:]]
        self.assertEqual(len(rows), 3)
        self.assertNotIn(".123456", text)
        self.assertIn(".12", text)

    def test_missing_csv_returns_empty_string(self):
        agent = DataAnalyzerAgent(base_dir=self.root)
        self.assertEqual(agent._read_csv_data(os.path.join(self.root, "nope.csv")), "")

    def test_dataset_dict_is_converted_and_sliced(self):
        agent = DataAnalyzerAgent(base_dir=self.root, decimal_places=1, sample_size=0)
        data = {"data": {"inputs": [[1.11, 2.22], [3.33, 4.44], [5.55, 6.66]],
                         "outputs": [1.0, 2.0, 3.0]}}
        arr = agent._read_dataset_and_to_array(data, max_rows=2)
        self.assertEqual(arr.shape, (2, 3))
        self.assertEqual(arr[0].tolist(), [1.1, 2.2, 1.0])

    def test_dataset_dict_is_sampled_without_replacement(self):
        agent = DataAnalyzerAgent(base_dir=self.root, decimal_places=1, sample_size=3, seed=1)
        data = {"data": {"inputs": [[i] for i in range(20)], "outputs": list(range(20))}}
        self.assertEqual(agent._read_dataset_and_to_array(data).shape, (3, 2))

    def test_bad_dataset_dict_returns_empty_array(self):
        agent = DataAnalyzerAgent(base_dir=self.root)
        self.assertEqual(agent._read_dataset_and_to_array({"nope": 1}).size, 0)

    def test_decimal_places_are_instance_level(self):
        """实例级配置不得互相污染（历史上写的是类属性）。"""
        a = DataAnalyzerAgent(base_dir=self.root, decimal_places=1, sample_size=0)
        b = DataAnalyzerAgent(base_dir=self.root, decimal_places=4, sample_size=0)
        self.assertEqual(a.decimal_places, 1)
        self.assertEqual(b.decimal_places, 4)

        one_decimal = a._read_csv_data(self.csv)
        four_decimals = b._read_csv_data(self.csv)
        self.assertIn("1.1,", one_decimal)
        self.assertNotIn("1.1235", one_decimal)
        self.assertIn("1.1235", four_decimals)

    def test_custom_prompt_placeholder_substituted(self):
        agent = DataAnalyzerAgent(base_dir=self.root)
        self.assertEqual(agent._create_prompt("CSV-BODY", "请分析：{csv_data}"), "请分析：CSV-BODY")

    def test_analysis_is_persisted_even_when_not_verbose(self):
        """回归：写盘逻辑曾被误嵌在 `if verbose:` 内，verbose=False 时初始分析丢失。"""
        agent = DataAnalyzerAgent(base_dir=self.root, llm_client=_RecordingClient(),
                                  sample_size=0)
        result = agent.analyze(self.csv, verbose=False)

        self.assertEqual(result, "INITIAL-ANALYSIS")
        records = self._records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sample_order"], 0)
        self.assertEqual(records[0]["island_id"], "this is the initial data")
        self.assertEqual(records[0]["analysis"], "INITIAL-ANALYSIS")

    def test_existing_records_are_appended(self):
        with open(os.path.join(self.root, "residual_analyze.json"), "w", encoding="utf-8") as f:
            json.dump([{"sample_order": 0, "analysis": "OLD"}], f)

        agent = DataAnalyzerAgent(base_dir=self.root, llm_client=_RecordingClient("NEW"),
                                  sample_size=0)
        agent.analyze(self.csv, verbose=False)

        records = self._records()
        self.assertEqual([r["analysis"] for r in records], ["OLD", "NEW"])

    def test_unreadable_csv_short_circuits_without_calling_model(self):
        client = _RecordingClient()
        agent = DataAnalyzerAgent(base_dir=self.root, llm_client=client)
        self.assertEqual(agent.analyze(os.path.join(self.root, "missing.csv")), "无法读取数据文件")
        self.assertEqual(client.clones, [])

    def test_missing_client_is_reported_not_raised(self):
        agent = DataAnalyzerAgent(base_dir=self.root, llm_client=None, sample_size=0)
        result = agent.analyze(self.csv, verbose=False)
        self.assertIn("请求出错", result)

    def test_analysis_task_clone_gets_output_cap(self):
        client = _RecordingClient()
        agent = DataAnalyzerAgent(base_dir=self.root, llm_client=client, sample_size=0)
        agent.analyze(self.csv, verbose=False)

        self.assertEqual(len(client.clones), 1)
        self.assertEqual(client.clones[0].task, "analysis")
        self.assertEqual(client.clones[0].kwargs["max_tokens"], 32768)


# ----------------------------------------------------------------------
# CoordinatorAgent：分类 / 最优追踪 / 落盘 / 停止条件
# ----------------------------------------------------------------------
class _FakeEvaluator:
    def __init__(self, outcomes):
        self._outcomes = outcomes
        self.requests: list[EvaluationRequest] = []

    def analyze(self, request: EvaluationRequest) -> EvaluationOutcome:
        self.requests.append(request)
        return self._outcomes[request.sample]


class _FakeDatabase:
    """只实现协调者真正用到的那几个接口。"""

    def __init__(self, best_score=-0.3, prompt=None):
        self._best_score_per_island = {0: best_score}
        self._prompt = prompt or buffer_lib.Prompt(
            code="PROMPT-CODE", version_generated=1, island_id=0)
        self.checkpoints: list[tuple] = []

    def get_prompt(self):
        return self._prompt

    def save_checkpoint(self, path, extra=None):
        self.checkpoints.append((path, extra))


class _FailingDatabase(_FakeDatabase):
    """checkpoint 落盘失败的缓冲（磁盘满/权限不足）。"""

    def save_checkpoint(self, path, extra=None):
        raise OSError("disk full")


class _DummySampler:
    def __init__(self, samples_per_prompt, batch_inference=True, trim=True,
                 prompt_ctx=None, llm_client=None):
        self.samples: list[str] = []

    def draw_samples(self, prompt, config):
        return list(self.samples), ["" for _ in self.samples]


def _batch(samples, thinking=None, sample_time=1.0, island_id=0, version=1):
    return SampleBatch(
        prompt=buffer_lib.Prompt(code="C", version_generated=version, island_id=island_id),
        samples=list(samples),
        thinking_contents=list(thinking if thinking is not None else [""] * len(samples)),
        sample_time=sample_time,
    )


class CoordinatorBehaviorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        CoordinatorAgent.set_global_sample_nums(1)

    def _coordinator(self, outcomes=None, best_score=-0.3, results_root=None,
                     database=None, **config_kwargs):
        config = config_lib.Config(
            results_root=results_root or self.root, samples_per_prompt=3, **config_kwargs)
        evaluators = [_FakeEvaluator(outcomes or {})]
        coord = CoordinatorAgent(
            database if database is not None else _FakeDatabase(best_score=best_score),
            evaluators, 3, config=config, llm_class=_DummySampler, llm_client=None)
        return coord, evaluators[0]

    # ── 质量分类 ──────────────────────────────────────────────
    def test_classify_quality_maps_none_bad_good(self):
        coord, _ = self._coordinator(best_score=-0.3)
        batch = _batch(["a", "b", "c"])
        batch.scores = [None, -0.5, -0.1]
        coord._classify_quality(batch, best_score=-0.3)
        self.assertEqual(batch.qualities, [QUALITY_NONE, QUALITY_BAD, QUALITY_GOOD])

    def test_score_equal_to_baseline_is_bad_not_good(self):
        coord, _ = self._coordinator()
        batch = _batch(["a"])
        batch.scores = [-0.3]
        coord._classify_quality(batch, best_score=-0.3)
        self.assertEqual(batch.qualities, [QUALITY_BAD])

    # ── 评估与最优追踪 ────────────────────────────────────────
    def test_evaluate_batch_records_scores_and_advances_global_counter(self):
        coord, evaluator = self._coordinator({
            "a": EvaluationOutcome(score=-0.5, error=None),
            "b": EvaluationOutcome(score=-0.2, error=None),
        })
        batch = _batch(["a", "b"])
        coord._evaluate_batch(batch, best_score=-0.3)

        self.assertEqual(batch.scores, [-0.5, -0.2])
        self.assertEqual(batch.errors, [None, None])
        self.assertEqual(coord._get_global_sample_nums(), 3)   # 1 + 两个样本

    def test_best_sample_only_counts_improvement_over_round_start(self):
        coord, _ = self._coordinator({
            "a": EvaluationOutcome(score=-0.5, error=None),
            "b": EvaluationOutcome(score=-0.2, error=None, residual="RESIDUAL"),
            "c": EvaluationOutcome(score=-0.4, error=None),
        })
        batch = _batch(["a", "b", "c"])
        coord._evaluate_batch(batch, best_score=-0.3)

        self.assertEqual(batch.best_sample, "b")
        self.assertEqual(batch.best_id, 2)          # 1-based 下标
        self.assertEqual(batch.best_score, -0.2)
        self.assertEqual(batch.best_residual, "RESIDUAL")

    def test_no_improvement_leaves_best_empty(self):
        coord, _ = self._coordinator({
            "a": EvaluationOutcome(score=-0.9, error="e"),
            "b": EvaluationOutcome(score=None, error="boom"),
        })
        batch = _batch(["a", "b"])
        coord._evaluate_batch(batch, best_score=-0.3)

        self.assertIsNone(batch.best_sample)
        self.assertIsNone(batch.best_id)
        self.assertIsNone(batch.best_residual)

    def test_tie_takes_the_later_sample(self):
        coord, _ = self._coordinator({
            "a": EvaluationOutcome(score=-0.1, error=None),
            "b": EvaluationOutcome(score=-0.1, error=None),
        })
        batch = _batch(["a", "b"])
        coord._evaluate_batch(batch, best_score=-0.3)
        self.assertEqual(batch.best_id, 2)

    def test_evaluation_request_carries_round_context(self):
        coord, evaluator = self._coordinator({"a": EvaluationOutcome(score=-0.1, error=None)})
        batch = _batch(["a"], sample_time=2.5, island_id=7, version=9)
        coord._evaluate_batch(batch, best_score=-0.3, profiler="PROFILER")

        request = evaluator.requests[0]
        self.assertEqual(request.sample, "a")
        self.assertEqual(request.island_id, 7)
        self.assertEqual(request.version_generated, 9)
        self.assertEqual(request.sample_time, 2.5)
        self.assertEqual(request.global_sample_nums, 2)   # 评估前 +1
        self.assertEqual(request.profiler, "PROFILER")

    # ── 经验落盘 ──────────────────────────────────────────────
    def _batch_with_entries(self):
        batch = _batch(["s0", "s1", "s2"], thinking=["t0", "t1", "t2"], sample_time=1.5)
        batch.scores = [-0.1, -0.5, None]
        batch.experience_entries = [
            ExperienceEntry(sample="s0", quality=QUALITY_GOOD, analysis="A0"),
            ExperienceEntry(sample="s1", quality=QUALITY_BAD, analysis="A1"),
            ExperienceEntry(sample="s2", quality=QUALITY_NONE, analysis="A2", error="boom"),
        ]
        return batch

    def _load_experiences(self):
        with open(os.path.join(self.root, "experiences.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    def test_persist_experiences_completes_attribution_fields(self):
        coord, _ = self._coordinator()
        CoordinatorAgent.set_global_sample_nums(10)     # 本轮 3 个样本 → 序号 8/9/10
        coord._persist_experiences(self._batch_with_entries())

        data = self._load_experiences()
        self.assertEqual(len(data[QUALITY_GOOD]), 1)
        self.assertEqual(len(data[QUALITY_BAD]), 1)
        self.assertEqual(len(data[QUALITY_NONE]), 1)

        good = data[QUALITY_GOOD][0]
        self.assertEqual(good["sample_order"], 8)
        self.assertEqual(good["island_id"], 0)
        self.assertEqual(good["sample_time"], 1.5)
        self.assertEqual(good["score"], -0.1)
        self.assertEqual(good["thinking_content"], "t0")
        self.assertEqual(good["equation"], "s0")
        self.assertNotIn("error", good)                 # 仅 None 类别带 error
        self.assertEqual(data[QUALITY_NONE][0]["error"], "boom")
        self.assertEqual(data[QUALITY_NONE][0]["sample_order"], 10)

    def test_persist_experiences_appends_to_existing_file(self):
        with open(os.path.join(self.root, "experiences.json"), "w", encoding="utf-8") as f:
            json.dump({QUALITY_GOOD: [{"analysis": "OLD"}]}, f)

        coord, _ = self._coordinator()
        coord._persist_experiences(self._batch_with_entries())

        data = self._load_experiences()
        self.assertEqual([e["analysis"] for e in data[QUALITY_GOOD]], ["OLD", "A0"])

    def test_persist_experiences_recovers_from_corrupt_file(self):
        with open(os.path.join(self.root, "experiences.json"), "w", encoding="utf-8") as f:
            f.write("{ broken json")

        coord, _ = self._coordinator()
        coord._persist_experiences(self._batch_with_entries())
        self.assertEqual(len(self._load_experiences()[QUALITY_GOOD]), 1)

    def test_persist_residual_uses_best_id_for_sample_order(self):
        coord, _ = self._coordinator()
        CoordinatorAgent.set_global_sample_nums(10)
        batch = _batch(["s0", "s1", "s2"])
        batch.best_id = 2
        batch.best_score = -0.15

        coord._persist_residual(batch, ResidualInsight(sample="s1", analysis="INSIGHT"))

        with open(os.path.join(self.root, "residual_analyze.json"), "r", encoding="utf-8") as f:
            records = json.load(f)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sample_order"], 9)     # 10 - 3 + 2
        self.assertEqual(records[0]["best_score"], -0.15)
        self.assertEqual(records[0]["analysis"], "INSIGHT")
        self.assertEqual(records[0]["equation"], "s1")

    # ── 停止条件与可观测性 ────────────────────────────────────
    def test_stop_on_wall_time_limit(self):
        coord, _ = self._coordinator(wall_time_limit_seconds=0)
        self.assertTrue(coord._should_stop(start_time=time.time()))

    def test_stop_on_max_sample_nums(self):
        coord, _ = self._coordinator()
        coord._max_sample_nums = 5
        CoordinatorAgent.set_global_sample_nums(5)
        self.assertTrue(coord._should_stop(start_time=time.time()))

    def test_keeps_running_below_limits(self):
        coord, _ = self._coordinator()
        coord._max_sample_nums = 5
        CoordinatorAgent.set_global_sample_nums(1)
        self.assertFalse(coord._should_stop(start_time=time.time()))

    def test_progress_csv_header_written_once_with_blank_best_for_inf(self):
        coord, _ = self._coordinator(best_score=float("-inf"))
        path = os.path.join(self.root, "round_progress.csv")

        coord._append_progress(island_id=0, start_time=0.0)
        coord._append_progress(island_id=0, start_time=0.0)

        with open(path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 3)                       # 1 表头 + 2 行
        self.assertEqual(lines[0].split(",")[0], "timestamp")
        self.assertEqual(lines[1].split(",")[3], "")           # -inf → 留空
        self.assertEqual(lines[1].split(",")[2], "0")

    def test_checkpoint_failure_is_tolerated(self):
        coord, _ = self._coordinator(database=_FailingDatabase())
        coord._save_checkpoint()          # 内部告警，不得把主循环带崩

    def test_checkpoint_carries_global_counter(self):
        coord, _ = self._coordinator()
        CoordinatorAgent.set_global_sample_nums(42)
        coord._save_checkpoint()

        path, extra = coord._database.checkpoints[0]
        self.assertEqual(path, os.path.join(self.root, "checkpoint.json"))
        self.assertEqual(extra["global_sample_nums"], 42)
        self.assertIn("saved_at", extra)

    def test_global_counter_is_shared_across_instances(self):
        coord_a, _ = self._coordinator()
        coord_b, _ = self._coordinator()
        CoordinatorAgent.set_global_sample_nums(1)
        coord_a._global_sample_nums_plus_one()
        self.assertEqual(coord_b._get_global_sample_nums(), 2)

    # ── 采样批次封装 ──────────────────────────────────────────
    def test_sample_batch_averages_sample_time(self):
        coord, _ = self._coordinator()
        coord._llm.samples = ["s0", "s1", "s2"]
        batch = coord._sample_batch(buffer_lib.Prompt(
            code="C", version_generated=1, island_id=0))

        self.assertEqual(batch.samples, ["s0", "s1", "s2"])
        self.assertEqual(batch.thinking_contents, ["", "", ""])
        self.assertGreaterEqual(batch.sample_time, 0.0)
        self.assertLess(batch.sample_time, 5.0)

    def test_analysis_failure_does_not_stop_main_loop(self):
        """经验/残差反思是增强项：抛异常只告警，主循环继续落盘。"""
        coord, _ = self._coordinator({"s0": EvaluationOutcome(score=-0.1, error=None)})
        batch = _batch(["s0"])

        class Boom:
            def analyze(self, *a, **k):
                raise RuntimeError("summarizer down")

        coord._summarizer = Boom()
        with self.assertRaises(RuntimeError):
            coord._summarize_experience(batch)   # 直接调用会抛……

        # ……但 sample() 里被 try 包住：评估走真实路径（推进全局计数），反思换成会抛的替身
        coord._sample_batch = lambda prompt: batch
        coord._max_sample_nums = 2
        CoordinatorAgent.set_global_sample_nums(1)
        coord.sample()                            # 不应抛出

        self.assertEqual(coord._get_global_sample_nums(), 2)
        self.assertTrue(coord._database.checkpoints, "主循环应在反思失败后继续落 checkpoint")

    def test_sampling_orchestrator_alias_points_to_coordinator(self):
        """协议入口：SPEC.entrypoints 里声明的 sample 必须可调用（编排入口唯一）。"""
        self.assertTrue(callable(CoordinatorAgent.sample))
        self.assertEqual(CoordinatorAgent.SPEC.entrypoints, ("sample",))


if __name__ == "__main__":
    unittest.main()
