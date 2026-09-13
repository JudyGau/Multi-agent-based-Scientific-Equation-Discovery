"""SamplerAgent 行为测试：骨架提取、经验/残差注入策略、采样编排与有界重试。

结构护栏（tests/test_architecture.py）只保证"角色卡与目录结构自洽"，本文件补的是
**行为**：给定一段 LLM 回复，采样器到底切出什么、注入什么、失败时怎么退。

三者都是纯逻辑、可完全离线验证的部分，因此全部用替身（假工具调用者 / 假 JSON 文件）
覆盖，不触网、不依赖真实 LLM：

* :mod:`drsr_420.agents.skeleton` —— 从混合文本切出可执行骨架；
* :mod:`drsr_420.agents.prompt_injection` —— 选哪些历史经验/残差拼进提示词；
* :class:`~drsr_420.agents.sampler_agent.SamplerAgent` —— 采样编排与空骨架重采样。
"""
import json
import os
import tempfile
import unittest
from unittest import mock

from drsr_420.agents import sampler_agent as sampler_mod
from drsr_420.agents.base import BaseAgent
from drsr_420.agents.prompt_injection import PromptInjector, resolve_policy
from drsr_420.agents.sampler_agent import SamplerAgent
from drsr_420.agents.skeleton import extract_body, extract_code_fragment
from drsr_420.agents.tool_caller_agent import ToolCallerAgent
from drsr_420.core import config as config_lib
from drsr_420.core import prompt_config as pc

# 一次"有效骨架"的 LLM 回复（带代码围栏，最贴近真实输出）
GOOD_REPLY = "推理过程...\n```python\ndef equation(x1, params):\n    return params[0] * x1\n```\n补充说明"
# 一次"抽不出代码"的回复
EMPTY_REPLY = "我认为应该先观察数据分布，再决定方程形式。"


# ----------------------------------------------------------------------
# 骨架提取
# ----------------------------------------------------------------------
class ExtractCodeFragmentTest(unittest.TestCase):
    """`extract_code_fragment`：从混合文本抽代码片段，抽不到返回 None。"""

    def test_def_body_wins_over_later_return(self):
        text = "def equation(x1, params):\n    a = params[0] * x1\n    return a\nreturn 999"
        self.assertEqual(
            extract_code_fragment(text),
            "    a = params[0] * x1\n    return a",
        )

    def test_return_line_collects_following_indented_lines(self):
        self.assertEqual(
            extract_code_fragment("说明\nreturn params[0] * x1 + \n    params[1]\n后续文字"),
            "return params[0] * x1 + \n    params[1]",
        )

    def test_params_expression_gets_return_prefix(self):
        self.assertEqual(
            extract_code_fragment("那么结果应该是\nparams[0] * x1 ** 2"),
            "return params[0] * x1 ** 2",
        )

    def test_no_code_returns_none(self):
        self.assertIsNone(extract_code_fragment("只有自然语言，没有任何代码。"))


class ExtractBodyTest(unittest.TestCase):
    """`extract_body`：优先代码围栏，失败返回空串（由上层决定重采样）。"""

    def test_fenced_block_is_preferred(self):
        body = extract_body(GOOD_REPLY)
        self.assertIn("return params[0] * x1", body)
        self.assertNotIn("```", body)
        self.assertNotIn("推理过程", body)

    def test_unfenced_def_keeps_function_body(self):
        self.assertEqual(
            extract_body("解释文字\ndef equation(x1, params):\n    return params[0]*x1\n更多解释"),
            "    return params[0]*x1",
        )

    def test_bare_return_gets_indent(self):
        self.assertEqual(extract_body("return params[0] * x1"), "    return params[0] * x1")

    def test_over_indented_return_is_normalized(self):
        self.assertEqual(
            extract_body("        return params[0] * x1"),
            "    return params[0] * x1",
        )

    def test_language_tag_is_stripped(self):
        # 围栏里的 "python" 被整体删掉，留下一个前导换行——对下游解析无害
        # （text_to_program 按行定位函数体），这里按实际行为锁定，避免无意改动。
        self.assertEqual(extract_body("```python\nreturn params[0]\n```"), "\nreturn params[0]")

    def test_no_code_returns_empty_string(self):
        self.assertEqual(extract_body(EMPTY_REPLY), "")


# ----------------------------------------------------------------------
# 提示词注入
# ----------------------------------------------------------------------
class _FakePromptContext:
    """只用注入器真正读到的三个接口（鸭子类型）。"""

    def __init__(self, max_param_count=5):
        self.max_param_count = max_param_count
        self.head_calls = 0

    def render_head(self):
        self.head_calls += 1
        return "HEAD-FROM-CONTEXT"

    def render_residual_block_title(self):
        return "RESIDUAL-TITLE-FROM-CONTEXT\n"


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


class ResolvePolicyTest(unittest.TestCase):
    def test_none_uses_config_defaults(self):
        policy = resolve_policy(None)
        self.assertEqual(policy.optional_category_probability, 0.5)
        self.assertEqual(policy.max_per_category, {"None": 3, "Good": 2, "Bad": 2})
        self.assertEqual(policy.freshness_threshold, 50)
        self.assertEqual(policy.inject_residual_probability, 0.5)
        self.assertEqual(policy.max_analysis_chars, 500)

    def test_partial_object_falls_back_per_field(self):
        class Partial:
            optional_category_probability = 0.9

        policy = resolve_policy(Partial())
        self.assertEqual(policy.optional_category_probability, 0.9)
        self.assertEqual(policy.max_analysis_chars, 500)   # 缺字段 → 只该字段回落

    def test_real_config_passthrough(self):
        cfg = config_lib.ExperienceInjectionConfig(optional_category_probability=0.25)
        self.assertIs(resolve_policy(cfg), cfg)


class ExperienceSelectionTest(unittest.TestCase):
    """`select_experiences`：类别配额、概率、新鲜度窗口、排序与噪声过滤。"""

    def setUp(self):
        self.injector = PromptInjector()

    def _policy(self, **overrides):
        return resolve_policy(config_lib.ExperienceInjectionConfig(**overrides))

    def test_none_always_injected_good_bad_gated_by_probability(self):
        experiences = {
            "None": [{"analysis": "N", "sample_order": 1}],
            "Good": [{"analysis": "G", "sample_order": 1, "score": -0.1}],
            "Bad": [{"analysis": "B", "sample_order": 1, "score": -0.9}],
        }
        # 概率 0：Good/Bad 一律不参与，None 不受影响
        selected = self.injector.select_experiences(
            experiences, 1, self._policy(optional_category_probability=0.0))
        self.assertEqual([e["analysis"] for e in selected], ["N"])

    def test_good_sorted_by_score_desc_then_truncated(self):
        experiences = {"Good": [
            {"analysis": "worst", "score": -0.5, "sample_order": 1},
            {"analysis": "best", "score": -0.1, "sample_order": 1},
            {"analysis": "middle", "score": -0.3, "sample_order": 1},
        ]}
        selected = self.injector.select_experiences(
            experiences, 1, self._policy(optional_category_probability=1.0))
        self.assertEqual([e["analysis"] for e in selected], ["best", "middle"])

    def test_bad_sorted_by_score_asc_then_truncated(self):
        experiences = {"Bad": [
            {"analysis": "mild", "score": -0.2, "sample_order": 1},
            {"analysis": "worst", "score": -0.9, "sample_order": 1},
            {"analysis": "bad", "score": -0.5, "sample_order": 1},
        ]}
        selected = self.injector.select_experiences(
            experiences, 1, self._policy(optional_category_probability=1.0))
        self.assertEqual([e["analysis"] for e in selected], ["worst", "bad"])

    def test_entries_without_numeric_score_sort_last_for_good(self):
        experiences = {"Good": [
            {"analysis": "no-score", "sample_order": 1},
            {"analysis": "scored", "score": -0.4, "sample_order": 1},
        ]}
        selected = self.injector.select_experiences(
            experiences, 1, self._policy(optional_category_probability=1.0))
        self.assertEqual(selected[0]["analysis"], "scored")

    def test_freshness_window_keeps_only_recent(self):
        experiences = {"None": [
            {"analysis": "ancient", "sample_order": 10},
            {"analysis": "recent", "sample_order": 80},
            {"analysis": "newest", "sample_order": 95},
        ]}
        # current=100 > threshold=50 → 只保留 [70, 100]
        selected = self.injector.select_experiences(experiences, 100, self._policy())
        self.assertEqual([e["analysis"] for e in selected], ["recent", "newest"])

    def test_no_freshness_filter_before_threshold(self):
        experiences = {"None": [{"analysis": "old", "sample_order": 1}]}
        selected = self.injector.select_experiences(experiences, 50, self._policy())
        self.assertEqual([e["analysis"] for e in selected], ["old"])

    def test_known_noise_error_is_dropped_other_errors_kept(self):
        experiences = {"None": [
            {"analysis": "noise", "sample_order": 1,
             "error": "Execution Error: too many values to unpack (expected 5)"},
            {"analysis": "real", "sample_order": 1, "error": "Execution Error: boom"},
        ]}
        selected = self.injector.select_experiences(experiences, 1, self._policy())
        self.assertNotIn("error", selected[0])
        self.assertEqual(selected[1]["error"], "Execution Error: boom")

    def test_per_category_limit_override(self):
        experiences = {"None": [{"analysis": f"N{i}", "sample_order": 1} for i in range(5)]}
        selected = self.injector.select_experiences(
            experiences, 1, self._policy(max_per_category={"None": 1, "Good": 2, "Bad": 2}))
        self.assertEqual(len(selected), 1)


class ExperiencePromptTest(unittest.TestCase):
    def setUp(self):
        self.injector = PromptInjector()

    def test_analysis_text_truncated_with_ellipsis(self):
        prompt = self.injector.build_experience_prompt(
            [{"type": "Good", "analysis": "0123456789", "sample_order": 1}], 5)
        self.assertIn("01234...", prompt)
        self.assertNotIn("012345", prompt)

    def test_param_budget_note_added_for_failure_lessons(self):
        injector = PromptInjector(prompt_ctx=_FakePromptContext(max_param_count=6))
        prompt = injector.build_experience_prompt(
            [{"type": "None", "analysis": "A", "sample_order": 1}], 500)
        self.assertIn("exactly 6 trainable parameters", prompt)
        self.assertIn("params[0]..params[5]", prompt)

    def test_no_budget_note_without_failure_lesson(self):
        injector = PromptInjector(prompt_ctx=_FakePromptContext(max_param_count=6))
        prompt = injector.build_experience_prompt(
            [{"type": "Good", "analysis": "A", "sample_order": 1}], 500)
        self.assertNotIn("trainable parameters", prompt)


class ExperienceInjectionTest(unittest.TestCase):
    """`inject_experiences`：读盘、缓存、前置拼接。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.exp_path = os.path.join(self.root, "experiences.json")
        self.injector = PromptInjector(base_dir=self.root)

    def test_missing_file_returns_content_unchanged(self):
        self.assertEqual(self.injector.inject_experiences("BODY", resolve_policy(None)), "BODY")

    def test_experience_block_is_prepended(self):
        _write_json(self.exp_path, {"None": [{"analysis": "LESSON", "sample_order": 3}]})
        out = self.injector.inject_experiences("BODY", resolve_policy(None))
        self.assertTrue(out.endswith("BODY"))
        self.assertIn("LESSON", out)
        self.assertLess(out.index("LESSON"), out.index("BODY"))

    def test_corrupt_json_is_ignored(self):
        with open(self.exp_path, "w", encoding="utf-8") as f:
            f.write("{ this is not json")
        self.assertEqual(self.injector.inject_experiences("BODY", resolve_policy(None)), "BODY")

    def test_cache_reused_until_mtime_changes(self):
        _write_json(self.exp_path, {"None": [{"analysis": "FIRST", "sample_order": 3}]})
        first_mtime = os.path.getmtime(self.exp_path)
        self.assertIn("FIRST", self.injector.inject_experiences("BODY", resolve_policy(None)))

        # 内容改了但 mtime 没变（模拟原子替换落在同一时间戳）→ 仍读缓存
        _write_json(self.exp_path, {"None": [{"analysis": "SECOND", "sample_order": 3}]})
        os.utime(self.exp_path, (first_mtime, first_mtime))
        self.assertIn("FIRST", self.injector.inject_experiences("BODY", resolve_policy(None)))

        # mtime 前进 → 缓存失效，读到新内容
        os.utime(self.exp_path, (first_mtime + 10, first_mtime + 10))
        out = self.injector.inject_experiences("BODY", resolve_policy(None))
        self.assertIn("SECOND", out)
        self.assertNotIn("FIRST", out)

    def test_cache_cleared_when_file_disappears(self):
        _write_json(self.exp_path, {"None": [{"analysis": "FIRST", "sample_order": 3}]})
        self.assertIn("FIRST", self.injector.inject_experiences("BODY", resolve_policy(None)))
        os.remove(self.exp_path)
        self.assertEqual(self.injector.inject_experiences("BODY", resolve_policy(None)), "BODY")
        self.assertEqual(self.injector._cache, {})


class ResidualInjectionTest(unittest.TestCase):
    """`inject_residual`：概率门、前置条件、历史格式与截断。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.exp_path = os.path.join(self.root, "experiences.json")
        self.res_path = os.path.join(self.root, "residual_analyze.json")
        self.injector = PromptInjector(base_dir=self.root)

    def test_probability_zero_never_injects(self):
        _write_json(self.exp_path, {})
        _write_json(self.res_path, [{"analysis": "OLD", "sample_order": 3}])
        self.assertEqual(self.injector.inject_residual("BODY", 0.0), "BODY")

    def test_not_injected_before_sampling_loop_starts(self):
        """首轮没有 experiences.json：即便概率为 1 也不注入（避免空注入）。"""
        _write_json(self.res_path, [{"analysis": "OLD", "sample_order": 3}])
        self.assertEqual(self.injector.inject_residual("BODY", 1.0), "BODY")

    def test_latest_analysis_is_prepended(self):
        _write_json(self.exp_path, {})
        _write_json(self.res_path, [
            {"analysis": "OLDER", "sample_order": 1},
            {"analysis": "LATEST", "sample_order": 2},
        ])
        out = self.injector.inject_residual("BODY", 1.0)
        self.assertIn("LATEST", out)
        self.assertNotIn("OLDER", out)
        self.assertTrue(out.endswith("BODY"))

    def test_legacy_list_analysis_takes_first_item(self):
        _write_json(self.exp_path, {})
        _write_json(self.res_path, [{"analysis": ["FIRST", "SECOND"], "sample_order": 2}])
        out = self.injector.inject_residual("BODY", 1.0)
        self.assertIn("FIRST", out)
        self.assertNotIn("SECOND", out)

    def test_long_analysis_truncated_to_2000_chars(self):
        _write_json(self.exp_path, {})
        _write_json(self.res_path, [{"analysis": "x" * 2500, "sample_order": 2}])
        out = self.injector.inject_residual("BODY", 1.0)
        self.assertIn("x" * 2000 + "...", out)
        self.assertNotIn("x" * 2001, out)

    def test_uses_prompt_context_block_title(self):
        _write_json(self.exp_path, {})
        _write_json(self.res_path, [{"analysis": "LATEST", "sample_order": 2}])
        injector = PromptInjector(_FakePromptContext(), base_dir=self.root)
        self.assertIn("RESIDUAL-TITLE-FROM-CONTEXT", injector.inject_residual("BODY", 1.0))


class BuildRequestContentTest(unittest.TestCase):
    """`build_request_content`：任务头 + 注入块 + 原始 content 的组装顺序。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name

    def test_strips_and_prepends_head(self):
        injector = PromptInjector(base_dir=self.root)
        out = injector.build_request_content("\n\n  BODY  \n")
        head = pc.head_template.format(
            dependent=pc.dependent_name_in_prompt,
            problem=pc.problem_name_in_prompt,
            independent=pc.independent_name_in_prompt,
        )
        self.assertTrue(out.startswith(head + "\n"))
        self.assertTrue(out.endswith("BODY"))

    def test_head_from_prompt_context(self):
        ctx = _FakePromptContext()
        out = PromptInjector(ctx, base_dir=self.root).build_request_content("BODY")
        self.assertTrue(out.startswith("HEAD-FROM-CONTEXT"))
        self.assertEqual(ctx.head_calls, 1)

    def test_broken_config_object_does_not_break_prompt(self):
        """注入超参数对象缺字段时，提示词仍要能构造出来（回落到默认值）。"""
        class Broken:
            pass

        out = PromptInjector(base_dir=self.root).build_request_content("BODY", Broken())
        self.assertTrue(out.endswith("BODY"))


# ----------------------------------------------------------------------
# 采样编排
# ----------------------------------------------------------------------
class _ScriptExhausted(BaseException):
    """脚本用尽：继承 BaseException 以便穿过 sampler 的 `except Exception` 重试。"""


class _FakeToolCaller:
    """替身：按预置脚本返回 (responses, thinking)，并记录每次调用参数。"""

    def __init__(self, script):
        self._script = list(script)
        self.calls: list[dict] = []

    def complete(self, content, repeat=1):
        self.calls.append({"content": content, "repeat": repeat})
        if not self._script:
            raise _ScriptExhausted(f"complete 第 {len(self.calls)} 次调用超出预置脚本")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        responses, thinking = item
        return list(responses), list(thinking)


def _make_sampler(script, samples_per_prompt=2, **kwargs):
    sampler = SamplerAgent(samples_per_prompt, llm_client=None, **kwargs)
    fake = _FakeToolCaller(script)
    sampler._tool_caller = fake
    return sampler, fake


class SamplerWiringTest(unittest.TestCase):
    def test_tool_loop_is_delegated_to_tool_caller_agent(self):
        sampler = SamplerAgent(2, llm_client=None)
        self.assertIsInstance(sampler, BaseAgent)
        self.assertIsInstance(sampler._tool_caller, ToolCallerAgent)

    def test_results_root_is_pushed_to_injector(self):
        with tempfile.TemporaryDirectory() as tmp:
            sampler, _ = _make_sampler([(["ok", "ok"], ["", ""])], trim=False)
            sampler.draw_samples("PROMPT", config_lib.Config(results_root=tmp))
            self.assertEqual(sampler._prompt_injector.base_dir, tmp)

    def test_missing_results_root_falls_back_to_cwd(self):
        sampler, _ = _make_sampler([(["ok", "ok"], ["", ""])], trim=False)
        sampler.draw_samples("PROMPT", config_lib.Config(results_root=None))
        self.assertEqual(sampler._prompt_injector.base_dir, ".")


class SamplerDrawSamplesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config = config_lib.Config(results_root=self._tmp.name, samples_per_prompt=2)

    def test_batch_path_trims_all_samples(self):
        sampler, fake = _make_sampler([([GOOD_REPLY, "```\nreturn params[1]\n```"], ["t1", "t2"])])
        samples, thinking = sampler.draw_samples("PROMPT", self.config)

        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["repeat"], 2)
        self.assertIn("params[0] * x1", samples[0])
        self.assertEqual(samples[1], "    return params[1]")
        self.assertEqual(thinking, ["t1", "t2"])

    def test_trim_disabled_keeps_raw_responses(self):
        sampler, _ = _make_sampler(
            [([GOOD_REPLY, EMPTY_REPLY], ["t1", "t2"])], trim=False)
        samples, _thinking = sampler.draw_samples("PROMPT", self.config)
        self.assertEqual(samples[0], GOOD_REPLY)

    def test_empty_skeleton_is_resampled_once(self):
        sampler, fake = _make_sampler([
            ([EMPTY_REPLY, GOOD_REPLY], ["t1", "t2"]),          # 批量采样
            (["```\nreturn params[0]\n```"], ["retry-think"]),  # 第 1 个样本重采样
        ])
        samples, thinking = sampler.draw_samples("PROMPT", self.config)

        self.assertEqual([c["repeat"] for c in fake.calls], [2, 1])
        self.assertEqual(samples[0], "    return params[0]")
        self.assertEqual(thinking[0], "retry-think")
        self.assertIn("params[0] * x1", samples[1])

    def test_skeleton_still_empty_after_retries_is_dropped(self):
        script = [([EMPTY_REPLY], [""])] + [([EMPTY_REPLY], [""])] * sampler_mod.MAX_BODY_RETRIES
        sampler, fake = _make_sampler(script, samples_per_prompt=1)
        samples, thinking = sampler.draw_samples("PROMPT", self.config)

        self.assertEqual(samples, [])
        self.assertEqual(thinking, [])
        self.assertEqual(len(fake.calls), 1 + sampler_mod.MAX_BODY_RETRIES)

    def test_retry_gives_up_after_max_body_retries(self):
        """重采样次数有上界：不会无限向 LLM 讨要骨架。"""
        self.assertGreater(sampler_mod.MAX_BODY_RETRIES, 0)
        self.assertLessEqual(sampler_mod.MAX_BODY_RETRIES, 5)

    def test_sampling_exception_retries_then_returns_empty(self):
        script = [RuntimeError("boom")] * sampler_mod._MAX_SAMPLE_ATTEMPTS
        sampler, fake = _make_sampler(script, samples_per_prompt=1)
        samples, thinking = sampler.draw_samples("PROMPT", self.config)

        self.assertEqual((samples, thinking), ([], []))
        self.assertEqual(len(fake.calls), sampler_mod._MAX_SAMPLE_ATTEMPTS)

    def test_exception_then_success_returns_samples(self):
        sampler, fake = _make_sampler([
            RuntimeError("boom"),
            ([GOOD_REPLY, GOOD_REPLY], ["t1", "t2"]),
        ])
        samples, _thinking = sampler.draw_samples("PROMPT", self.config)
        self.assertEqual(len(samples), 2)
        self.assertEqual(len(fake.calls), 2)

    def test_non_batch_path_samples_one_at_a_time(self):
        sampler, fake = _make_sampler(
            [(["```\nreturn params[0]\n```"], [""]), (["```\nreturn params[1]\n```"], [""])],
            samples_per_prompt=2, batch_inference=False)
        samples, _thinking = sampler.draw_samples("PROMPT", self.config)

        self.assertEqual([c["repeat"] for c in fake.calls], [1, 1])
        self.assertEqual(samples, ["    return params[0]", "    return params[1]"])

    def test_injected_content_contains_instruction_and_experience(self):
        with open(os.path.join(self._tmp.name, "experiences.json"), "w", encoding="utf-8") as f:
            json.dump({"None": [{"analysis": "LESSON-X", "sample_order": 1}]}, f)

        sampler, fake = _make_sampler([([GOOD_REPLY, GOOD_REPLY], ["", ""])])
        sampler.draw_samples("PROMPT-BODY", self.config)

        sent = fake.calls[0]["content"]
        self.assertIn(sampler_mod.pc.instruction_prompt, sent)
        self.assertIn("PROMPT-BODY", sent)
        self.assertIn("LESSON-X", sent)

    def test_injector_failure_does_not_abort_sampling(self):
        """注入环节出错（经验文件损坏/自定义 context 异常）时仍要能采样。"""
        sampler, fake = _make_sampler([([GOOD_REPLY, GOOD_REPLY], ["", ""])])
        with mock.patch.object(
                sampler._prompt_injector, "inject_experiences",
                side_effect=RuntimeError("injection down")):
            samples, _thinking = sampler.draw_samples("PROMPT", self.config)

        self.assertEqual(len(samples), 2)
        self.assertEqual(len(fake.calls), 1)


if __name__ == "__main__":
    unittest.main()
