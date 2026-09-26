"""架构地形（`evaluation.architecture_facts`）与两条注入通道的回归。

背景（实测 MRFCompress-Cuboid_20260926-110809，87 个有评分样本）：
* 52 个是**同一个完整二阶响应面**（两个平方项 + 交叉项）的重新参数化——换记号、
  平移 `λ-1`、取对数、`(λ-1)/(λ+1)` 都被采样器当成"新架构"；
* 而"删掉一个平方项"的 5 参数**非对称**形式（该实验里分数最高的干净骨架）
  一次都没被提出过。

所以这里的断言分三层：
1. 结构指纹对**记号不变**（同一架构的不同写法必须落进同一指纹），且不会把
   "只对乘积 λ12·λ23 做二次"误判成双变量响应面；
2. "从未评估过的一阶删项邻域"必须逐样本核对得出来（含解析失败样本的显式披露）；
3. 采样通道与残差通道都真的把这段机器事实（以及"触底必须给删项候选"的要求）
   送进了提示词。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from drsr_420.agents.prompt_injection import PromptInjector
from drsr_420.agents.residual_analyzer_agent import ResidualAnalyzerAgent
from drsr_420.core import prompt_config as pc
from drsr_420.evaluation import architecture_facts as af

NAMES = ["lambda12", "lambda23"]

#: 完整二阶响应面：6 个项（含两个平方项与交叉项）。
FULL_TERMS = ("const", "lambda12", "lambda12*lambda23", "lambda12^2", "lambda23", "lambda23^2")


def _eq(body: str) -> str:
    return "def equation_v1(lambda12, lambda23, params):\n" + body


#: 同一个"完整二阶响应面"的五种写法（原坐标 / 平移 / 括号换序+逐项解包 / 对数坐标 / 重复相乘）。
FULL_QUADRATIC_VARIANTS = {
    "plain": _eq("    return (params[0] + params[1]*lambda12 + params[2]*lambda23\n"
                 "            + params[3]*lambda12**2 + params[4]*lambda23**2\n"
                 "            + params[5]*lambda12*lambda23)"),
    "shifted": _eq("    d12 = lambda12 - 1.0\n"
                   "    d23 = lambda23 - 1.0\n"
                   "    return (params[0] + params[1]*d12 + params[2]*d23\n"
                   "            + params[3]*d12**2 + params[4]*d23**2 + params[5]*d12*d23)"),
    "unpacked": _eq("    a0, a1, a2, a11, a22, a12 = (params[0], params[1], params[2],\n"
                    "                                 params[3], params[4], params[5])\n"
                    "    return (a0 + a1*lambda12 + a2*lambda23 + a11*lambda12**2\n"
                    "            + a22*lambda23**2 + a12*lambda12*lambda23)"),
    "log": _eq("    x = np.log(lambda12)\n    y = np.log(lambda23)\n"
               "    return (params[0] + params[1]*x + params[2]*y + params[3]*x**2\n"
               "            + params[4]*y**2 + params[5]*x*y)"),
    "repeated": _eq("    u = lambda12 - 1.0\n    v = lambda23 - 1.0\n"
                    "    return (params[0] + params[1]*u + params[2]*v + params[3]*u*u\n"
                    "            + params[4]*v*v + params[5]*u*v)"),
}

#: 非对称二次：删掉 λ12 的平方项（该实验里从未被提出过的 5 参数形式）。
ASYMMETRIC = _eq("    u = lambda12 - 1.0\n    v = lambda23 - 1.0\n"
                 "    return (params[0] + params[1]*u + params[2]*v + params[3]*v**2\n"
                 "            + params[4]*u*v)")


def _write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _entry(order: int, score: float, equation: str) -> dict:
    return {"sample_order": order, "score": score, "equation": equation}


def _stagnating_entries(count: int = 12, equation: str = FULL_QUADRATIC_VARIANTS["plain"]) -> list:
    """一批样本：最好分在第 1 个样本，其余都更差（用于"已触底"的判定）。"""
    entries = [_entry(1, -0.5, equation)]
    entries += [_entry(o, -2.0, equation) for o in range(2, count + 1)]
    return entries


class _ScriptedChat:
    """按脚本返回响应的假 LLM 客户端，并记录每次收到的消息。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.kwargs = {}
        self.sent: list[list[dict]] = []

    def chat(self, messages, on_delta=None):
        self.sent.append([dict(m) for m in messages])
        item = self._responses.pop(0)
        if on_delta is not None:
            on_delta({"content": item.get("content", ""), "reasoning_content": ""})
        return item


class FingerprintInvarianceTest(unittest.TestCase):
    """结构指纹必须对记号不变，同时不把"乘积的二次"当成双变量响应面。"""

    def test_same_architecture_in_five_notations_gives_one_fingerprint(self):
        prints = {name: af.architecture_fingerprint(text, NAMES)
                  for name, text in FULL_QUADRATIC_VARIANTS.items()}
        for name, fingerprint in prints.items():
            with self.subTest(notation=name):
                self.assertEqual(fingerprint, FULL_TERMS)

    def test_family_is_second_order_surface_for_all_notations(self):
        for name, text in FULL_QUADRATIC_VARIANTS.items():
            with self.subTest(notation=name):
                fingerprint = af.architecture_fingerprint(text, NAMES)
                self.assertEqual(af.family_of(fingerprint, NAMES), "second_order_surface")

    def test_asymmetric_quadratic_keeps_only_one_square(self):
        fingerprint = af.architecture_fingerprint(ASYMMETRIC, NAMES)
        self.assertIn("lambda23^2", fingerprint)
        self.assertIn("lambda12*lambda23", fingerprint)
        self.assertNotIn("lambda12^2", fingerprint)

    def test_quadratic_in_the_product_is_a_different_architecture(self):
        """只对乘积 λ12·λ23 做二次 = 一维山脊模型，不能算双变量响应面。"""
        text = _eq("    x = lambda12 * lambda23\n"
                   "    return params[0] + params[1]*x + params[2]*x**2")
        fingerprint = af.architecture_fingerprint(text, NAMES)
        self.assertIn("higher", fingerprint)
        self.assertNotIn("lambda12^2", fingerprint)
        self.assertNotIn("lambda23^2", fingerprint)
        self.assertEqual(af.family_of(fingerprint, NAMES), "coupled_not_second_order")

    def test_parametric_exponent_is_a_power_law_not_a_square(self):
        text = _eq("    return params[0]*lambda12**params[1]*lambda23**params[2] + params[3]")
        fingerprint = af.architecture_fingerprint(text, NAMES)
        self.assertEqual(fingerprint, ("const", "power(lambda12,lambda23)"))
        self.assertEqual(af.family_of(fingerprint, NAMES), "power_law")

    def test_np_power_with_numeric_exponent_counts_as_a_square(self):
        text = _eq("    return (params[0] + params[1]*np.power(lambda23, 2)\n"
                   "            + params[2]*lambda12)")
        self.assertEqual(af.architecture_fingerprint(text, NAMES),
                         ("const", "lambda12", "lambda23^2"))

    def test_extra_cubic_terms_keep_the_second_order_family(self):
        text = _eq("    return (params[0] + params[1]*lambda12 + params[2]*lambda23\n"
                   "            + params[3]*lambda12**2 + params[4]*lambda23**2\n"
                   "            + params[5]*lambda12*lambda23 + params[6]*lambda12**3)")
        fingerprint = af.architecture_fingerprint(text, NAMES)
        self.assertTrue(set(FULL_TERMS) <= set(fingerprint))
        self.assertIn("higher", fingerprint)
        self.assertEqual(af.family_of(fingerprint, NAMES), "second_order_surface")

    def test_unparsable_and_wrong_arity_return_none(self):
        self.assertIsNone(af.architecture_fingerprint("just prose, no statement at all", NAMES))
        self.assertIsNone(af.architecture_fingerprint("", NAMES))
        self.assertIsNone(af.architecture_fingerprint(FULL_QUADRATIC_VARIANTS["plain"],
                                                      ["x", "y", "z"]))
        self.assertIsNone(af.architecture_fingerprint(None, NAMES))

    def test_alias_wrapping_a_whole_sum_keeps_the_term_structure(self):
        """实测形态：``sigma = <6 项和式>`` 再 ``return sigma``。

        把整个和式压成一个原子，6 个项就只剩 1 个：这些完整二次型会被记成
        "只有一个交叉项的形式"，架构统计整体失真。
        """
        text = _eq("    sigma = (params[0] + params[1]*lambda12 + params[2]*lambda23\n"
                   "             + params[3]*lambda12**2 + params[4]*lambda23**2\n"
                   "             + params[5]*lambda12*lambda23)\n"
                   "    return sigma")
        self.assertEqual(af.architecture_fingerprint(text, NAMES), FULL_TERMS)

    def test_multiline_assignment_keeps_the_parenthesis_balance(self):
        text = _eq("    sigma = (params[0]\n"
                   "             + params[1]*lambda12\n"
                   "             + params[2]*lambda23\n"
                   "             + params[3]*lambda12**2\n"
                   "             + params[4]*lambda23**2)\n"
                   "    return sigma")
        self.assertEqual(af.architecture_fingerprint(text, NAMES),
                         ("const", "lambda12", "lambda12^2", "lambda23", "lambda23^2"))

    def test_parametric_exponent_on_a_parenthesized_composite(self):
        """``(λ23 + p*λ12)**p`` 是复合量的幂律，不是交叉项。"""
        text = _eq("    s = lambda23 + params[1] * lambda12\n"
                   "    return params[0]*s**params[2] + params[3]")
        self.assertEqual(af.architecture_fingerprint(text, NAMES),
                         ("const", "power(lambda12,lambda23)"))

    def test_features_can_be_recovered_from_the_signature(self):
        text = _eq("    return params[0]*lambda12 + params[1]")
        self.assertEqual(af.features_from_equation(text), ["lambda12", "lambda23"])
        annotated = "def equation_v1(a: np.ndarray, b: np.ndarray, params: np.ndarray):\n    return a"
        self.assertEqual(af.features_from_equation(annotated), ["a", "b"])
        self.assertIsNone(af.features_from_equation("no signature here"))

    def test_feature_names_are_interchangeable(self):
        """换变量名不影响结构标签（标签用调用方给的名字渲染）。"""
        text = ("def equation_v1(x, y, params):\n"
                "    return (params[0] + params[1]*x + params[2]*y + params[3]*x**2\n"
                "            + params[4]*y**2 + params[5]*x*y)")
        self.assertEqual(af.architecture_fingerprint(text, ["x", "y"]),
                         ("const", "x", "x*y", "x^2", "y", "y^2"))


class TerrainTest(unittest.TestCase):
    """已试架构的汇总与"一阶删项邻域"的差集。"""

    def test_stagnation_and_family_counts(self):
        terrain = af.sampling_terrain(_stagnating_entries(), NAMES)
        self.assertTrue(terrain["ok"])
        self.assertEqual(terrain["n_parsed"], 12)
        self.assertEqual(terrain["best_order"], 1)
        self.assertEqual(terrain["best_score"], -0.5)
        self.assertEqual(terrain["stagnant_samples"], 11)
        self.assertEqual(terrain["since_same_family"], 11)
        self.assertEqual(terrain["target_family_count"], 12)
        self.assertEqual(terrain["target_exact_count"], 12)

    def test_all_three_structural_deletions_are_untried(self):
        terrain = af.sampling_terrain(_stagnating_entries(), NAMES)
        self.assertEqual([item["dropped"] for item in terrain["untried_deletions"]],
                         ["lambda12*lambda23", "lambda12^2", "lambda23^2"])

    def test_a_tried_deletion_disappears_from_the_untried_list(self):
        """一旦有样本就是那个非对称形式，它就不能再被报成"从未评估过"。"""
        entries = _stagnating_entries() + [_entry(13, -2.5, ASYMMETRIC)]
        terrain = af.sampling_terrain(entries, NAMES)
        dropped = [item["dropped"] for item in terrain["untried_deletions"]]
        self.assertNotIn("lambda12^2", dropped)
        self.assertIn("lambda23^2", dropped)

    def test_entries_without_a_numeric_score_are_skipped_but_unparsed_are_counted(self):
        entries = _stagnating_entries() + [_entry(13, -3.0, "prose without return")]
        entries += [{"sample_order": 14, "score": None, "equation": ASYMMETRIC}]
        terrain = af.sampling_terrain(entries, NAMES)
        self.assertEqual(terrain["n_scored"], 13)      # score=None 的条目不计入
        self.assertEqual(terrain["n_parsed"], 12)
        self.assertEqual(terrain["n_unparsed"], 1)     # 但解析不出的必须显式披露

    def test_target_override_uses_the_analysed_equation(self):
        terrain = af.sampling_terrain(_stagnating_entries(), NAMES, target=ASYMMETRIC)
        self.assertFalse(terrain["target_is_best"])
        self.assertEqual(terrain["target_terms"], sorted(
            af.architecture_fingerprint(ASYMMETRIC, NAMES)))

    def test_non_binary_features_yield_empty_terrain(self):
        terrain = af.sampling_terrain(_stagnating_entries(), ["x", "y", "z"])
        self.assertFalse(terrain["ok"])
        self.assertEqual(terrain["untried_deletions"], [])


class TerrainRenderTest(unittest.TestCase):
    """注入闸门与渲染内容（数字必须来自汇总，不得写空话）。"""

    TITLE = "### TITLE ###"

    def test_renders_machine_numbers_and_untried_neighbourhoods(self):
        terrain = af.sampling_terrain(_stagnating_entries(), NAMES)
        block = af.render_terrain(terrain, NAMES, self.TITLE)
        self.assertTrue(block.startswith(self.TITLE))
        self.assertIn("scored samples: 12", block)
        self.assertIn("best score: -0.5", block)
        self.assertIn("samples since that best: 11", block)
        self.assertIn("drop lambda12^2 ->", block)
        self.assertIn("SAME", block)

    def test_not_injected_without_enough_samples(self):
        terrain = af.sampling_terrain(_stagnating_entries(count=5), NAMES)
        self.assertEqual(af.render_terrain(terrain, NAMES, self.TITLE), "")

    def test_not_injected_before_the_architecture_has_stagnated(self):
        # 分数逐样本变好（最高分就在最新样本上）→ 自最优以来 0 个样本，不注入
        entries = [_entry(o, -float(13 - o), FULL_QUADRATIC_VARIANTS["plain"])
                   for o in range(1, 13)]
        terrain = af.sampling_terrain(entries, NAMES)
        self.assertEqual(terrain["stagnant_samples"], 0)
        self.assertEqual(af.render_terrain(terrain, NAMES, self.TITLE), "")

    def test_empty_terrain_is_not_injected(self):
        self.assertEqual(af.render_terrain({}, NAMES, self.TITLE), "")
        self.assertEqual(af.render_terrain({"ok": False}, NAMES, self.TITLE), "")

    def test_exhausted_neighbourhood_says_so_instead_of_listing_nothing(self):
        entries = _stagnating_entries()
        entries += [_entry(13, -2.5, ASYMMETRIC)]
        terrain = af.sampling_terrain(entries, NAMES)
        self.assertTrue(terrain["untried_deletions"])
        block = af.render_terrain(terrain, NAMES, self.TITLE)
        self.assertIn("NEVER evaluated", block)


class SamplingInjectionTest(unittest.TestCase):
    """采样通道：地形块真的进了每条采样提示。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.exp_path = os.path.join(self.root, "experiences.json")

    def _ctx(self):
        return pc.PromptContext(n_features=2, feature_names=list(NAMES),
                               dependent_name="sigma", background="bg")

    def test_missing_experiences_injects_nothing(self):
        injector = PromptInjector(self._ctx(), base_dir=self.root)
        self.assertEqual(injector.inject_architecture_terrain("BODY"), "BODY")

    def test_terrain_block_is_prepended_before_the_task_content(self):
        _write_json(self.exp_path, {"Bad": _stagnating_entries()})
        injector = PromptInjector(self._ctx(), base_dir=self.root)
        out = injector.inject_architecture_terrain("BODY")
        self.assertTrue(out.endswith("BODY"))
        self.assertIn("architectures already tried were computed by code", out)
        self.assertIn("drop lambda12^2 ->", out)
        self.assertLess(out.index("drop lambda12^2"), out.index("BODY"))

    def test_build_request_content_carries_the_terrain_block(self):
        _write_json(self.exp_path, {"Bad": _stagnating_entries()})
        injector = PromptInjector(self._ctx(), base_dir=self.root)
        out = injector.build_request_content("RAW-CONTENT")
        self.assertIn("RAW-CONTENT", out)
        self.assertIn("architectures already tried were computed by code", out)

    def test_terrain_result_is_cached_per_experiences_mtime(self):
        _write_json(self.exp_path, {"Bad": _stagnating_entries()})
        injector = PromptInjector(self._ctx(), base_dir=self.root)
        injector.inject_architecture_terrain("BODY")
        cached = injector._terrain_cache[1]
        injector.inject_architecture_terrain("BODY-2")
        self.assertIs(injector._terrain_cache[1], cached)


class ResidualChannelInjectionTest(unittest.TestCase):
    """残差通道：架构地形进提示词，且模板要求给出删项候选。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.residual = np.array([[1.0, 2.0, 0.5], [2.0, 4.0, -0.25]])

    def _ctx(self):
        return pc.PromptContext(n_features=2, feature_names=list(NAMES),
                               dependent_name="sigma", background="bg")

    def test_terrain_block_targets_the_analysed_equation(self):
        _write_json(os.path.join(self.root, "experiences.json"),
                    {"Bad": _stagnating_entries()})
        agent = ResidualAnalyzerAgent(_ScriptedChat([]), prompt_ctx=self._ctx(),
                                     results_root=self.root)
        block = agent._load_terrain_block(ASYMMETRIC)
        self.assertIn("the term set of the equation under analysis", block)
        self.assertIn("lambda23^2", block)

    def test_terrain_block_is_absent_without_history(self):
        agent = ResidualAnalyzerAgent(_ScriptedChat([]), prompt_ctx=self._ctx(),
                                     results_root=self.root)
        self.assertEqual(agent._load_terrain_block(ASYMMETRIC), "")

    def test_analysis_prompt_carries_terrain_facts_and_the_deletion_rule(self):
        _write_json(os.path.join(self.root, "experiences.json"),
                    {"Bad": _stagnating_entries()})
        client = _ScriptedChat([{"content": "INSIGHT", "reasoning_content": ""}])
        agent = ResidualAnalyzerAgent(client, prompt_ctx=self._ctx(),
                                      results_root=self.root)
        agent.analyze(ASYMMETRIC, self.residual)

        sent = client.sent[0][1]["content"]
        self.assertIn("architectures already tried were computed by code", sent)
        self.assertIn("MUST include at least one DELETION", sent)

    def test_corrupt_experiences_file_does_not_break_analysis(self):
        with open(os.path.join(self.root, "experiences.json"), "w", encoding="utf-8") as f:
            f.write("{ broken")
        client = _ScriptedChat([{"content": "INSIGHT", "reasoning_content": ""}])
        agent = ResidualAnalyzerAgent(client, prompt_ctx=self._ctx(),
                                      results_root=self.root)
        insight = agent.analyze(ASYMMETRIC, self.residual)
        self.assertEqual(insight.analysis, "INSIGHT")


if __name__ == "__main__":
    unittest.main()