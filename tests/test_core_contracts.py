"""核心契约回归测试（第 8 轮 b 批：审计确认的 live bug）。

覆盖审计实锤的问题：
- ToolCallerAgent.complete() 恒返回 list（repeat==1 返回标量曾被 sampler 的
  `list(str)` 炸成逐字符样本：1 次请求 → 81 个"样本" + 82 次 LLM 调用）；
- buffer.get_prompt 的 version_generated 与 prompt 头部编号一致（旧 +1 让
  _sample_to_program 的自递归改名恒为 no-op）；
- evaluate_on_problems：NaN 数据集显式报错、全起点异常保留真实错误、
  残差列不再被绝对 3 位小数取整清零、1 维 inputs 兼容、常数输出 R² 边界；
- llm.ClientFactory：提供商别名必须规范化（zhipu/bigmodel/glm4 → glm），
  否则 _adapt_payload 的提供商分支全部静默跳过；
- content token 负数 clamp 与全局统计并发正确性；
- LocalSandbox.close() 哨兵关停 worker。
"""
import threading
import time
import unittest

import numpy as np

from drsr_420 import llm
from drsr_420.core import code_manipulation as cm
from drsr_420.core import config as config_lib
from drsr_420.evaluation import problems as eop
from drsr_420.agents.tool_caller_agent import ToolCallerAgent
from drsr_420.core.buffer import ExperienceBuffer


class CompleteContractTest(unittest.TestCase):
    class _FakeClient:
        def __init__(self, text="hello world"):
            self.calls = 0
            self.text = text

        def chat(self, messages, on_delta=None):
            self.calls += 1
            return {"content": self.text, "reasoning_content": "r", "tool_calls": []}

    def _tc(self, client):
        return ToolCallerAgent(client, tool_executor=lambda *a: "{}")

    def test_repeat_one_returns_lists_not_scalars(self):
        """回归：repeat<=1 曾返回标量，被批量分支 list(str) 炸成单字符样本。"""
        tc = self._tc(self._FakeClient())
        responses, thinks = tc.complete("p", 1)
        self.assertIsInstance(responses, list)
        self.assertEqual(responses, ["hello world"])  # 一整条响应，而不是逐字符
        self.assertEqual(thinks, ["r"])

    def test_repeat_n_returns_n(self):
        tc = self._tc(self._FakeClient())
        responses, _ = tc.complete("p", 3)
        self.assertEqual(len(responses), 3)
        self.assertTrue(all(isinstance(r, str) for r in responses))

    def test_api_error_still_list_shaped(self):
        class Boom:
            def chat(self, messages, on_delta=None):
                raise RuntimeError("api down")
        responses, thinks = self._tc(Boom()).complete("p", 1)
        self.assertEqual(responses, [""])
        self.assertEqual(thinks, [""])


class VersionGeneratedTest(unittest.TestCase):
    TEMPLATE = (
        "import numpy as np\n"
        "@equation.evolve\n"
        "def equation(x, params):\n"
        "    return params[0] * x\n"
    )

    def _buffer(self, n_programs):
        cfg = config_lib.ExperienceBufferConfig(num_islands=1)
        eb = ExperienceBuffer(cfg, cm.text_to_program(self.TEMPLATE), "equation")
        for i in range(n_programs):
            func = cm.text_to_function(
                f"def equation(x, params):\n    return params[{i % 3}] * x + {i}\n")
            eb.register_program(func, 0, {"data": -1.0 - i})
        return eb

    def test_matches_prompt_header(self):
        import re
        eb = self._buffer(3)
        prompt = eb.get_prompt()  # 单岛：自动选岛 0
        codes = re.findall(r"def equation_v(\d+)\(", prompt.code)
        self.assertGreaterEqual(len(codes), 2)  # 至少 _v0.. 与一个头部
        # 契约（Prompt docstring）：待补全函数（prompt 最后一个 _v 头部）
        # 必须是 _v{version_generated}。旧实现 version_generated=len+1 会让
        # 它指向一个不存在的版本，_sample_to_program 的改名恒为 no-op。
        self.assertEqual(prompt.version_generated, int(codes[-1]))

    def test_recursive_rename_targets_real_header(self):
        """自递归样本（调用自己的头部名）必须被改名成 function_to_evolve。"""
        from drsr_420.agents.evaluator_agent import _sample_to_program
        eb = self._buffer(2)
        prompt = eb.get_prompt()
        header = f"equation_v{prompt.version_generated}"
        self.assertIn(f"def {header}(", prompt.code)  # 头部确实存在
        sample = f"def {header}(x, params):\n    return {header}(x, params) + 1\n"
        _, program = _sample_to_program(
            sample, prompt.version_generated,
            cm.text_to_program(self.TEMPLATE), "equation")
        self.assertNotIn(header + "(", program)  # 自递归调用已被改名
        self.assertIn("equation(x, params)", program)


class EvaluateContractTest(unittest.TestCase):
    @staticmethod
    def _data(n=100, seed=1):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1, 1, size=(n, 1))
        y = 2.0 * X[:, 0] + 1.0
        return {"inputs": X, "outputs": y}

    def test_nan_outputs_raise_informative(self):
        d = self._data()
        d["outputs"][5] = np.nan
        with self.assertRaises(ValueError) as ctx:
            eop.evaluate(d, lambda x, p: p[0] * x + p[1], n_params=2, n_starts=1)
        self.assertIn("NaN", str(ctx.exception))

    def test_all_starts_exception_preserves_reason(self):
        bad = lambda x, p: p[10] * x  # 越界索引
        with self.assertRaises(Exception) as ctx:
            eop.evaluate(self._data(), bad, n_params=2, n_starts=2)
        self.assertIn("index", str(ctx.exception).lower())

    def test_residual_column_keeps_precision(self):
        """回归：残差列按绝对 3 位小数取整会把好拟合清零（残差分析回路全盲）。"""
        d = self._data()
        d["outputs"] = d["outputs"] + 1e-6 * np.sin(np.arange(len(d["outputs"])))
        score, matrix, params = eop.evaluate(
            d, lambda x, p: p[0] * x + p[1], n_params=2, n_starts=3)
        self.assertIsNotNone(score)
        self.assertGreater(float(np.max(np.abs(matrix[:, -1]))), 0.0)

    def test_1d_inputs_accepted(self):
        d = {"inputs": np.linspace(-1, 1, 50), "outputs": 2 * np.linspace(-1, 1, 50) + 1}
        score, matrix, _ = eop.evaluate(d, lambda x, p: p[0] * x + p[1], n_params=2)
        self.assertIsNotNone(score)

    def test_constant_output_perfect_fit(self):
        d = self._data()
        d["outputs"] = np.full_like(d["outputs"], 3.0)
        score, matrix, _ = eop.evaluate(
            d, lambda x, p: np.full_like(x, 3.0), n_params=2, verbose=True)
        self.assertIsNotNone(score)


class ProviderAliasTest(unittest.TestCase):
    def _payload_for(self, model, effort=None):
        client = llm.ClientFactory.from_config({
            "api_key": "test-key",
            "model": model,
            "max_completion_tokens": 4096,
        })
        if effort:
            client.kwargs["reasoning_effort"] = effort
        return client, client._build_payload([{"role": "user", "content": "x"}])

    def test_glm_aliases_canonicalized(self):
        for alias in ("glm", "zhipu", "bigmodel", "glm4"):
            client, payload = self._payload_for(f"{alias}/glm-4.6", effort="high")
            self.assertEqual(client.provider, "glm", alias)
            self.assertEqual(payload.get("max_tokens"), 4096, alias)
            self.assertNotIn("max_completion_tokens", payload, alias)
            self.assertEqual(payload.get("thinking"), {"type": "enabled"}, alias)

    def test_other_aliases(self):
        for alias, canon in (("sflow", "siliconflow"), ("deep-infra", "deepinfra"),
                            ("bltcy", "blt"), ("keji", "cstcloud")):
            client, _ = self._payload_for(f"{alias}/m")
            self.assertEqual(client.provider, canon, alias)


class TokenAccountingTest(unittest.TestCase):
    def _client(self):
        return llm.LLMClient(api_key="k", model="m", base_url="http://x/v1")

    def test_negative_content_clamped(self):
        c = self._client()
        usage = {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17,
                 "completion_tokens_details": {"reasoning_tokens": 12}}
        out = c._finalize_response("x", "r", [], usage, time.time() - 0.01)
        self.assertEqual(out["tokens"]["content"], 0)
        self.assertGreaterEqual(c.tokens["content"], 0)

    def test_concurrent_accumulation_exact(self):
        llm.reset_global_tokens()
        llm.reset_global_time()
        threads = [threading.Thread(
            target=llm._accumulate_global_stats, args=(1, 2, 3, 6, 0.5))
            for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        snap = llm.get_global_tokens()
        self.assertEqual(snap["prompt"], 20)
        self.assertEqual(snap["content"], 60)
        self.assertAlmostEqual(llm.get_global_time(), 10.0, places=5)


class SandboxCloseTest(unittest.TestCase):
    def test_close_terminates_workers(self):
        from drsr_420.agents.evaluator_agent import LocalSandbox
        sb = LocalSandbox(numba_accelerate=False, pool_size=1)
        try:
            self.assertTrue(any(p.is_alive() for p in sb._workers))
        finally:
            sb.close()
        deadline = time.time() + 5
        while any(p.is_alive() for p in sb._workers) and time.time() < deadline:
            time.sleep(0.05)
        self.assertFalse(any(p.is_alive() for p in sb._workers))


if __name__ == "__main__":
    unittest.main()
