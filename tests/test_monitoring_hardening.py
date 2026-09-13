"""观测与调度层加固回归测试（第 8 轮）。

覆盖：
- Profiler：score=0.0（完美拟合）计入成功而非失败；samples/top-K JSON 落盘；
- rag_kb.get_embedder：双重检查锁定，并发首调只构造一次嵌入模型；
- LocalSandbox._respawn_workers：重建 worker 前清空队列中的陈旧任务。
"""
import multiprocessing
import queue
import tempfile
import threading
import unittest

import numpy as np

from drsr_420.knowledge import rag_kb as rk
from drsr_420.agents.evaluator_agent import LocalSandbox
from drsr_420.core.profile import Profiler


class _FuncStub:
    """模拟 code_manipulation.Function 的最小属性集。"""

    def __init__(self, name, body, global_sample_nums=None, score=None):
        self.name = name
        self.body = body
        self.global_sample_nums = global_sample_nums
        self.score = score
        self.sample_time = 0.1
        self.evaluate_time = 0.2
        self.optimized_params = None

    def __str__(self):
        return f"def {self.name}():\n  {self.body}"


class ProfilerCountingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prof = Profiler(
            self.tmp.name,
            samples_per_iteration=4,
            target_variance=2.0,
            persist_all_samples=True,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_zero_score_counts_as_success(self):
        """score = -MSE，完美拟合时是 ±0.0；按真值判断会被误计为失败。"""
        self.prof.register_function(_FuncStub("equation", "return 0", 3, score=0.0))
        self.assertEqual(self.prof._evaluate_success_program_num, 1)
        self.assertEqual(self.prof._evaluate_failed_program_num, 0)

    def test_negative_score_still_success_and_mse_roundtrip(self):
        self.prof.register_function(_FuncStub("equation", "return 1", 2, score=-4.0))
        self.assertEqual(self.prof._evaluate_success_program_num, 1)
        self.assertAlmostEqual(self.prof._score_to_mse(-4.0), 4.0)
        self.assertAlmostEqual(self.prof._mse_to_nmse(4.0), 2.0)

    def test_none_score_counts_as_failure(self):
        self.prof.register_function(_FuncStub("equation", "return 2", 5, score=None))
        self.assertEqual(self.prof._evaluate_failed_program_num, 1)
        self.assertEqual(self.prof._evaluate_success_program_num, 0)

    def test_sample_and_topk_files_written(self):
        func = _FuncStub("equation", "return 3", 3, score=-1.0)
        func.optimized_params = np.array([1.5, -2.0])
        self.prof.register_function(func)
        import os
        samples = os.path.join(self.tmp.name, "samples", "samples_3.json")
        top = os.path.join(self.tmp.name, "samples", "top01_samples_3.json")
        self.assertTrue(os.path.exists(samples))
        self.assertTrue(os.path.exists(top))
        import json
        with open(samples, encoding="utf-8") as f:
            content = json.load(f)
        self.assertEqual(content["params"], [1.5, -2.0])
        self.assertEqual(content["iteration"], 1)  # sample_order 3 / 每轮 4 个
        with open(os.path.join(self.tmp.name, "progress.json"), encoding="utf-8") as f:
            progress = json.load(f)
        self.assertEqual(progress[0]["best_sample_order"], 3)


class _CountingEmbedder(rk.EmbeddingModel):
    construct_count = 0

    def __init__(self, model_name, query_prefix=""):
        type(self).construct_count += 1
        self._evt = threading.Event()

    def embed(self, texts):
        return [[] for _ in texts]


class EmbedderSingletonTest(unittest.TestCase):
    def setUp(self):
        rk.reset_embedder()
        self._orig = rk.SentenceTransformerEmbedder
        rk.SentenceTransformerEmbedder = _CountingEmbedder
        _CountingEmbedder.construct_count = 0
        self.cfg = {"backend": "local", "model": "m1", "query_prefix": ""}

    def tearDown(self):
        rk.SentenceTransformerEmbedder = self._orig
        rk.reset_embedder()

    def test_same_config_reuses_instance(self):
        e1 = rk.get_embedder(self.cfg)
        e2 = rk.get_embedder(self.cfg)
        self.assertIs(e1, e2)
        self.assertEqual(_CountingEmbedder.construct_count, 1)

    def test_concurrent_first_call_constructs_once(self):
        """双重检查锁定：并发首调不得各自加载一份（模型）嵌入器。"""
        results = []

        def call():
            results.append(rk.get_embedder(self.cfg))

        threads = [threading.Thread(target=call) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(results), 8)
        self.assertEqual(len(set(map(id, results))), 1)
        self.assertEqual(_CountingEmbedder.construct_count, 1)

    def test_changed_config_rebuilds(self):
        rk.get_embedder(self.cfg)
        rk.get_embedder({**self.cfg, "model": "m2"})
        self.assertEqual(_CountingEmbedder.construct_count, 2)


class SandboxRespawnDrainTest(unittest.TestCase):
    def test_stale_queue_is_replaced_on_respawn(self):
        """超时重建 worker 时必须丢弃陈旧任务队列。

        旧实现用 ``get_nowait()`` 清空队列，但 ``multiprocessing.Queue`` 由后台
        feeder 线程异步投递——刚 put 进去的任务往往还没进管道，``get_nowait`` 看不到，
        于是"清空"只是尽力而为，残留任务仍会被新 worker 消费（其结果管道已关闭，
        只会白白重拟合并造成后续连锁超时）。该缺陷在本套件里表现为**偶发失败**：
        测试顺序/负载稍有变化就复现。

        现在改为换一条全新队列，并断言"新 worker 绑定的队列已不是那条陈旧队列"。
        """
        sb = object.__new__(LocalSandbox)
        sb._workers = []          # 没有存活进程可 terminate
        sb._pool_size = 0
        stale_queue = multiprocessing.Queue()
        for i in range(3):
            stale_queue.put(("stale-task", i))
        sb._task_queue = stale_queue
        sb._spawn_workers = lambda: []

        sb._respawn_workers()

        self.assertIsNot(sb._task_queue, stale_queue,
                         "重建后仍在用旧队列：陈旧任务会被新 worker 消费")
        with self.assertRaises(queue.Empty):
            sb._task_queue.get_nowait()   # 新队列必须是干净的
        self.assertEqual(sb._workers, [])


if __name__ == "__main__":
    unittest.main()
