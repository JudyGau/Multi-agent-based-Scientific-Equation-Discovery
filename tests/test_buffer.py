"""buffer 单元测试：Cluster / Island / ExperienceBuffer 的注册、采样、序列化与断点续跑。"""
import json
import os
import tempfile
import unittest

from drsr_420 import code_manipulation as cm
from drsr_420 import config as config_lib
from drsr_420.buffer import (
    Cluster,
    ExperienceBuffer,
    Island,
    Prompt,
    _reduce_score,
    _softmax,
)

import numpy as np

TEMPLATE_TEXT = (
    "import numpy as np\n"
    "\n"
    "@evaluate.run\n"
    "def evaluate(data):\n"
    "    return 0.0\n"
    "\n"
    "@equation.evolve\n"
    "def equation(x, params):\n"
    "    return params[0] * x\n"
)

EQ_BODY = "def equation(x, params):\n    return params[0] * x\n"


def _mk_template():
    return cm.text_to_program(TEMPLATE_TEXT)


def _mk_function():
    return cm.text_to_function(EQ_BODY)


def _mk_buffer(num_islands=4):
    cfg = config_lib.ExperienceBufferConfig(num_islands=num_islands)
    return ExperienceBuffer(cfg, _mk_template(), "equation"), cfg


class HelpersTest(unittest.TestCase):
    def test_reduce_score_is_mean(self):
        self.assertAlmostEqual(_reduce_score({"a": 1.0, "b": 3.0}), 2.0)

    def test_softmax_sums_to_one(self):
        p = _softmax(np.array([1.0, 2.0, 3.0]), temperature=1.0)
        self.assertAlmostEqual(float(np.sum(p)), 1.0)


class ClusterTest(unittest.TestCase):
    def test_score_property(self):
        c = Cluster(-0.5, _mk_function())
        self.assertEqual(c.score, -0.5)

    def test_register_and_sample(self):
        f1 = cm.text_to_function("def equation(x, params):\n    return params[0]*x\n")
        f2 = cm.text_to_function("def equation(x, params):\n    return params[0]*x + params[1]\n")
        c = Cluster(-0.5, f1)
        c.register_program(f2)
        self.assertEqual(len(c._programs), 2)
        sampled = c.sample_program()
        self.assertIn(sampled, c._programs)

    def test_roundtrip(self):
        c = Cluster(-0.25, _mk_function())
        c.register_program(_mk_function())
        restored = Cluster.from_dict(c.to_dict())
        self.assertAlmostEqual(restored.score, c.score)
        self.assertEqual(len(restored._programs), len(c._programs))


class IslandTest(unittest.TestCase):
    def _mk_island(self):
        return Island(_mk_template(), "equation", functions_per_prompt=2,
                      cluster_sampling_temperature_init=0.1,
                      cluster_sampling_temperature_period=30000)

    def test_register_creates_cluster(self):
        island = self._mk_island()
        island.register_program(_mk_function(), {"data": -0.5})
        self.assertEqual(len(island._clusters), 1)
        self.assertEqual(island._num_programs, 1)

    def test_get_prompt_returns_str_and_version(self):
        island = self._mk_island()
        island.register_program(_mk_function(), {"data": -0.5})
        code, version = island.get_prompt()
        self.assertIsInstance(code, str)
        self.assertGreaterEqual(version, 1)
        self.assertIn("equation", code)

    def test_roundtrip(self):
        island = self._mk_island()
        island.register_program(_mk_function(), {"data": -0.5})
        restored = Island.from_dict(
            island.to_dict(), _mk_template(), "equation", 2, 0.1, 30000)
        self.assertEqual(restored._num_programs, island._num_programs)
        self.assertEqual(len(restored._clusters), len(island._clusters))


class ExperienceBufferTest(unittest.TestCase):
    def test_initial_state(self):
        eb, cfg = _mk_buffer(num_islands=3)
        self.assertEqual(len(eb._islands), 3)
        self.assertEqual(len(eb._best_score_per_island), 3)
        self.assertTrue(all(s == -float("inf") for s in eb._best_score_per_island))

    def test_get_prompt_returns_prompt(self):
        eb, _ = _mk_buffer()
        eb.register_program(_mk_function(), None, {"data": -0.5})  # 所有岛屿均非空
        prompt = eb.get_prompt()
        self.assertIsInstance(prompt, Prompt)
        self.assertIsInstance(prompt.code, str)
        self.assertIn("equation", prompt.code)
        self.assertIn(prompt.island_id, range(4))

    def test_get_prompt_on_empty_buffer_raises(self):
        eb, _ = _mk_buffer()
        with self.assertRaises(RuntimeError):
            eb.get_prompt()

    def test_get_prompt_only_picks_non_empty_island(self):
        eb, _ = _mk_buffer(num_islands=4)
        eb.register_program(_mk_function(), 2, {"data": -0.5})  # 只有岛屿 2 非空
        for _ in range(20):  # 多次采样都应命中岛屿 2，不再崩溃
            self.assertEqual(eb.get_prompt().island_id, 2)

    def test_island_num_programs_property(self):
        eb, _ = _mk_buffer(num_islands=2)
        self.assertEqual(eb._islands[0].num_programs, 0)
        eb.register_program(_mk_function(), 0, {"data": -0.5})
        self.assertEqual(eb._islands[0].num_programs, 1)

    def test_register_specific_island_updates_best(self):
        eb, _ = _mk_buffer()
        eb.register_program(_mk_function(), 0, {"data": -0.5})
        self.assertAlmostEqual(eb._best_score_per_island[0], -0.5)
        self.assertEqual(eb._best_score_per_island[1], -float("inf"))

    def test_register_all_islands(self):
        eb, _ = _mk_buffer(num_islands=3)
        eb.register_program(_mk_function(), None, {"data": -0.7})
        for i in range(3):
            self.assertAlmostEqual(eb._best_score_per_island[i], -0.7)

    def test_best_score_only_improves(self):
        eb, _ = _mk_buffer(num_islands=2)
        eb.register_program(_mk_function(), 0, {"data": -0.5})
        eb.register_program(_mk_function(), 0, {"data": -0.9})  # 更差
        self.assertAlmostEqual(eb._best_score_per_island[0], -0.5)
        eb.register_program(_mk_function(), 0, {"data": -0.1})  # 更好
        self.assertAlmostEqual(eb._best_score_per_island[0], -0.1)

    def test_reset_islands_keeps_founder(self):
        eb, _ = _mk_buffer(num_islands=4)
        eb.register_program(_mk_function(), None, {"data": -0.5})
        before = sum(len(i._clusters) for i in eb._islands)
        eb.reset_islands()
        # 岛屿数量不变，且重置后由 founder 恢复 best_score（不为 -inf）
        self.assertEqual(len(eb._islands), 4)
        self.assertTrue(all(s != -float("inf") for s in eb._best_score_per_island))
        after = sum(len(i._clusters) for i in eb._islands)
        self.assertLessEqual(after, before)

    def test_to_from_dict_roundtrip(self):
        eb, cfg = _mk_buffer(num_islands=2)
        eb.register_program(_mk_function(), 0, {"data": -0.42})
        restored = ExperienceBuffer.from_dict(
            eb.to_dict(), cfg, _mk_template(), "equation")
        self.assertEqual(len(restored._islands), 2)
        self.assertAlmostEqual(restored._best_score_per_island[0], -0.42)

    def test_checkpoint_roundtrip(self):
        eb, cfg = _mk_buffer(num_islands=2)
        eb.register_program(_mk_function(), 0, {"data": -0.33})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "checkpoint.json")
            eb.save_checkpoint(path, extra={"global_sample_nums": 7})
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.assertEqual(raw["global_sample_nums"], 7)

            eb2 = ExperienceBuffer(cfg, _mk_template(), "equation")
            eb2.load_checkpoint(path)
            self.assertAlmostEqual(eb2._best_score_per_island[0], -0.33)
            self.assertEqual(len(eb2._islands), 2)


if __name__ == "__main__":
    unittest.main()
