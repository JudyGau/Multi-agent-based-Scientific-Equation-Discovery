"""收敛型早停机制（target_nmse / 平台期 / 失败熔断）的行为回归。

======================
为什么要有这个机制
======================

重构前主循环只有两个**预算型**停止条件（墙钟 ``wall_time_limit_seconds`` 与样本数
``max_sample_nums``，见 ``CoordinatorAgent._should_stop``），没有任何**收敛型**条件。
experiments/MRFCompress-Cuboid 的两次实测暴露了代价：

* run 164828：第 7 批就达到 NMSE≈1e-6（机器精度量级），之后又空跑 20 批——
  约 2700 s 与 45 万 prompt tokens（占全程 72%）纯属浪费；
* run 163158：曾**连续 18 个批次**无全局最优改进、之后才出现全场最优（sample 90）
  ——朴素"N 批没进步就停"会在探索低谷把好 run 掐死；
* run.err 里同一分数以 1e-13 量级的抖动连续三次触发 "increased"——改进判定
  不带容差的话，平台期计数器永远被噪声清零，机制形同虚设。

由此得出三条设计约束，本文件逐条固化：

1. 绝对目标（``target_nmse``）是省钱主力，平台期 patience 必须给足
   （>= 2×num_islands）并有 warmup 保护；
2. 改进判定必须带容差 ``max(1e-12, 1e-9·|best|)``；
3. 全部早停条件默认关闭（None），预算型条件照常兜底；计数器是类级共享状态
   （多 sampler 线程任一判停、全体一起停），且随新实验归零不串台。
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest

from drsr_420.core import buffer as buffer_lib
from drsr_420.core import config as config_lib
from drsr_420.agents import coordinator_agent as coord_mod
from drsr_420.agents.coordinator_agent import CoordinatorAgent
from drsr_420.agents.messages import SampleBatch
from drsr_420.runtime import pipeline


class _FakeDatabase:
    """只实现早停路径真正用到的那几个接口。"""

    def __init__(self, best_score=-0.3):
        self._best_score_per_island = {0: best_score}

    def get_prompt(self):
        return buffer_lib.Prompt(code="C", version_generated=1, island_id=0)

    def save_checkpoint(self, path, extra=None):
        pass


class _DummySampler:
    def __init__(self, *args, **kwargs):
        self.samples: list[str] = []

    def draw_samples(self, prompt, config):
        return list(self.samples), ["" for _ in self.samples]


def _batch(scores) -> SampleBatch:
    """构造一个只填 scores 的批次（早停计数只看 scores）。"""
    n = len(scores)
    return SampleBatch(
        prompt=buffer_lib.Prompt(code="C", version_generated=1, island_id=0),
        samples=[f"s{i}" for i in range(n)],
        thinking_contents=[""] * n,
        sample_time=0.0,
        scores=list(scores),
    )


class EarlyStopTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        CoordinatorAgent.set_global_sample_nums(1)

    def _coordinator(self, database=None, target_score=None, **config_kwargs) -> CoordinatorAgent:
        config = config_lib.Config(
            results_root=self.root, samples_per_prompt=3, **config_kwargs)
        return CoordinatorAgent(
            database if database is not None else _FakeDatabase(),
            [], 3, config=config, llm_class=_DummySampler, llm_client=None,
            target_score=target_score)


# ----------------------------------------------------------------------
# 默认行为：全部关闭，不改变既有预算型语义
# ----------------------------------------------------------------------
class DefaultOffTest(EarlyStopTestBase):
    def test_default_config_never_early_stops(self):
        coord = self._coordinator()
        self.assertIsNone(coord._target_score)
        self.assertIsNone(coord._early_stop_patience)
        self.assertIsNone(coord._early_stop_max_failed)
        self.assertFalse(coord._should_stop(start_time=time.time()))
        self.assertIsNone(CoordinatorAgent._early_stop_reason)

    def test_budget_conditions_are_untouched(self):
        coord = self._coordinator()
        coord._max_sample_nums = 5
        CoordinatorAgent.set_global_sample_nums(5)
        self.assertTrue(coord._should_stop(start_time=time.time()))
        # Config 是 frozen dataclass：墙钟上限必须走构造参数
        coord2 = self._coordinator(wall_time_limit_seconds=0)
        self.assertTrue(coord2._should_stop(start_time=time.time()))

    def test_warmup_defaults_to_num_islands(self):
        coord = CoordinatorAgent(
            _FakeDatabase(), [], 3,
            config=config_lib.Config(
                experience_buffer=config_lib.ExperienceBufferConfig(num_islands=7)),
            llm_class=_DummySampler, llm_client=None)
        self.assertEqual(coord._early_stop_min_batches, 7)


# ----------------------------------------------------------------------
# ① 绝对目标：target_nmse → target_score
# ----------------------------------------------------------------------
class TargetScoreTest(EarlyStopTestBase):
    def test_stops_when_global_best_reaches_target(self):
        coord = self._coordinator(database=_FakeDatabase(best_score=-0.05),
                                  target_score=-0.1)
        self.assertTrue(coord._should_stop(start_time=time.time()))
        self.assertIn('目标分', CoordinatorAgent._early_stop_reason)

    def test_keeps_running_below_target(self):
        coord = self._coordinator(database=_FakeDatabase(best_score=-0.5),
                                  target_score=-0.1)
        self.assertFalse(coord._should_stop(start_time=time.time()))
        self.assertIsNone(CoordinatorAgent._early_stop_reason)

    def test_no_valid_score_yet_does_not_stop(self):
        coord = self._coordinator(database=_FakeDatabase(best_score=float('-inf')),
                                  target_score=-0.1)
        self.assertFalse(coord._should_stop(start_time=time.time()))

    def test_pipeline_converts_nmse_to_score(self):
        cfg = config_lib.Config(target_nmse=1e-6)
        inputs = {'data': {'inputs': [[0.0]], 'outputs': [1.0, 3.0]}}  # np.var=1
        self.assertAlmostEqual(
            pipeline._target_score_from_config(cfg, inputs), -1e-6)

    def test_pipeline_returns_none_without_target(self):
        inputs = {'data': {'inputs': [[0.0]], 'outputs': [1.0, 3.0]}}
        self.assertIsNone(
            pipeline._target_score_from_config(config_lib.Config(), inputs))

    def test_pipeline_returns_none_when_variance_unavailable(self):
        cfg = config_lib.Config(target_nmse=1e-6)
        # 常数输出 → 方差为 0 → 换算不可能，返回 None（只告警不崩溃）
        inputs = {'data': {'inputs': [[0.0]], 'outputs': [2.0, 2.0]}}
        self.assertIsNone(pipeline._target_score_from_config(cfg, inputs))


# ----------------------------------------------------------------------
# ③ 平台期：容差改进判定 + warmup
# ----------------------------------------------------------------------
class PlateauTest(EarlyStopTestBase):
    def _record_stale(self, coord, n):
        """让全局最优保持 -0.3 不变，连续记录 n 个批次。"""
        for _ in range(n):
            coord._record_batch_for_early_stop(_batch([-0.4, None]))

    def test_patience_triggers_only_after_warmup(self):
        coord = self._coordinator(early_stop_patience=3, min_batches_before_early_stop=2)
        # 第 1 个批次只建立基线（prev=None 记为改进），其后 3 批无改进 → stale=3
        self._record_stale(coord, 4)
        # stale 已达 3，且全局批次完成数 4 >= warmup 2 —— 已过 warmup，判停
        self.assertTrue(coord._should_stop(start_time=time.time()))
        self.assertIn('平台期', CoordinatorAgent._early_stop_reason)

    def test_patience_holds_during_warmup(self):
        coord = self._coordinator(early_stop_patience=3, min_batches_before_early_stop=10)
        self._record_stale(coord, 5)
        self.assertFalse(coord._should_stop(start_time=time.time()))
        self.assertIsNone(CoordinatorAgent._early_stop_reason)

    def test_real_improvement_resets_stale_counter(self):
        coord = self._coordinator(early_stop_patience=3)
        self._record_stale(coord, 2)
        # 全局最优真实改进：-0.3 → -0.2（远超容差）
        coord._database._best_score_per_island[0] = -0.2
        coord._record_batch_for_early_stop(_batch([-0.2]))
        self.assertEqual(CoordinatorAgent._early_stale_batches, 0)
        self.assertFalse(coord._should_stop(start_time=time.time()))

    def test_subtolerance_jitter_counts_as_stale(self):
        """回归：1e-13 量级的参数优化抖动不算改进（实测 run.err 连续三次 'increased'）。"""
        base = -0.0019974690717875
        # 岛屿基线就从 base 附近开始：否则 -0.3 → base 的跳变会被算成真改进
        coord = self._coordinator(
            database=_FakeDatabase(best_score=base), early_stop_patience=3,
            min_batches_before_early_stop=1)
        coord._record_batch_for_early_stop(_batch([base]))       # 建立基线
        for i in range(1, 4):
            coord._database._best_score_per_island[0] = base + i * 1e-13
            coord._record_batch_for_early_stop(_batch([base + i * 1e-13]))
        self.assertEqual(CoordinatorAgent._early_stale_batches, 3)
        self.assertTrue(coord._should_stop(start_time=time.time()))


# ----------------------------------------------------------------------
# ② 失败熔断
# ----------------------------------------------------------------------
class FailedBatchBreakerTest(EarlyStopTestBase):
    def test_triggers_after_consecutive_all_failed_batches(self):
        coord = self._coordinator(max_failed_batches=2)
        coord._record_batch_for_early_stop(_batch([None, None]))
        self.assertFalse(coord._should_stop(start_time=time.time()))
        coord._record_batch_for_early_stop(_batch([None, None]))
        self.assertTrue(coord._should_stop(start_time=time.time()))
        self.assertIn('熔断', CoordinatorAgent._early_stop_reason)

    def test_any_valid_score_resets_the_counter(self):
        coord = self._coordinator(max_failed_batches=2)
        coord._record_batch_for_early_stop(_batch([None, None]))
        coord._record_batch_for_early_stop(_batch([-0.4, None]))   # 有有效分：清零
        coord._record_batch_for_early_stop(_batch([None, None]))
        self.assertFalse(coord._should_stop(start_time=time.time()))


# ----------------------------------------------------------------------
# 共享状态：多线程一起停、新实验不串台、终止行只写一次
# ----------------------------------------------------------------------
class SharedStateTest(EarlyStopTestBase):
    def test_state_is_class_level_and_reset_per_experiment(self):
        a = self._coordinator(max_failed_batches=1)
        a._record_batch_for_early_stop(_batch([None]))
        self.assertTrue(a._should_stop(start_time=time.time()))
        # 新实验构造新协调器：计数器与原因全部归零
        b = self._coordinator()
        self.assertEqual(CoordinatorAgent._early_failed_batches, 0)
        self.assertIsNone(CoordinatorAgent._early_stop_reason)
        self.assertFalse(b._should_stop(start_time=time.time()))

    def test_stop_row_written_once_with_reason(self):
        coord = self._coordinator(database=_FakeDatabase(best_score=-0.05),
                                  target_score=-0.1)
        self.assertTrue(coord._should_stop(start_time=time.time()))
        coord._write_early_stop_row(start_time=0.0)
        coord._write_early_stop_row(start_time=0.0)   # 第二次必须是无操作

        path = os.path.join(self.root, "round_progress.csv")
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 2)                       # 1 表头 + 1 终止行
        self.assertEqual(lines[0].split(",")[-1], "stop_reason")
        self.assertIn('目标分', lines[1])

    def test_normal_progress_rows_carry_the_full_schema(self):
        coord = self._coordinator()
        coord._append_progress(island_id=0, start_time=0.0)
        path = os.path.join(self.root, "round_progress.csv")
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        self.assertEqual(lines[0].split(",")[-1], "stop_reason")
        self.assertEqual(lines[1].split(",")[-1], "")         # 正常行留空

    def test_no_stop_row_when_budget_stopped(self):
        coord = self._coordinator()
        coord._max_sample_nums = 1
        CoordinatorAgent.set_global_sample_nums(1)            # 1 >= 1 → 预算停
        self.assertTrue(coord._should_stop(start_time=time.time()))
        self.assertIsNone(CoordinatorAgent._early_stop_reason)
        coord._write_early_stop_row(start_time=0.0)
        self.assertFalse(os.path.exists(os.path.join(self.root, "round_progress.csv")))


if __name__ == '__main__':
    unittest.main()
