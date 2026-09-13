"""第 8 轮（工具/MCP/RAG CLI）加固回归测试。

覆盖：
- rag_build CLI：--ingest 方向不再反转（只 --query 不得触发入库）；--dir 按项目根解析；
- rag_kb 配置键名：端点统一为 ``api_base_url``，旧的 ``api_host`` 写了就报错（含改名提示）；
- rag_kb.chunk_text：超长段硬切不再重复当前块、硬切片段之间保留 overlap；
- read_paper._doi_filename：DOI 路径穿越/绝对路径净化，且常见 DOI 文件名向后兼容；
- read_paper._summarize_text：max_tokens 回退、工具调用/空内容显式报错；
- tool_runner._server_env：提供商 API key 环境变量并入 MCP 子进程；
- mcp_server：底层异常被包装为 {"error": ...} JSON（SDK 会吞掉抛出型错误的文本）。
"""
import io
import json
import os
import sys
import unittest
from unittest import mock

from drsr_420.knowledge import rag_build
from drsr_420.knowledge import rag_kb
from drsr_420.knowledge.rag_kb import chunk_text, DEFAULT_CONFIG
from drsr_420.knowledge.tools import read_paper as rp
from drsr_420.knowledge.tools import mcp_server as ms
from drsr_420.knowledge import tool_runner as tr


class ChunkTextTest(unittest.TestCase):
    def test_long_para_does_not_duplicate_pending_chunk(self):
        text = "SHORT PARA AAA\n\n" + "X" * 1200 + "\n\nTAIL PARA BBB"
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        # 回归：旧实现在硬切循环里把 pending 的 current 反复 append 又不清空
        self.assertEqual(sum(c == "SHORT PARA AAA" for c in chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 500)
        self.assertTrue(any("TAIL PARA BBB" in c for c in chunks))

    def test_hard_split_fragments_overlap(self):
        text = "".join(str(i % 10) for i in range(1300))
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        # 相邻硬切片段之间保留 50 字符重叠：chunk0 的尾部 == chunk1 的头部
        self.assertEqual(chunks[0][-50:], chunks[1][:50])
        self.assertTrue(all(len(c) <= 500 for c in chunks))

    def test_empty_text(self):
        self.assertEqual(chunk_text("   ", 500, 50), [])

    def test_normal_merge_keeps_doc_order(self):
        text = "A B\n\nC D"
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertEqual(chunks, ["A B\nC D"])


class ResolveDirTest(unittest.TestCase):
    def test_relative_falls_back_to_repo_root(self):
        import tempfile
        # 在无关目录下，"pdf_downloads" 相对当前目录不存在 → 应解析到项目根
        with tempfile.TemporaryDirectory() as unrelated_cwd:
            old = os.getcwd()
            os.chdir(unrelated_cwd)
            try:
                resolved = rag_build._resolve_dir("pdf_downloads")
            finally:
                os.chdir(old)
        self.assertEqual(os.path.normpath(resolved),
                         os.path.normpath(str(rag_build._REPO_ROOT / "pdf_downloads")))

    def test_existing_relative_path_untouched(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(rag_build._resolve_dir(tmp), tmp)


class RagBuildCliTest(unittest.TestCase):
    def _run_main(self, argv):
        kb = mock.MagicMock()
        kb.count.return_value = 0
        kb.search.return_value = []
        kb.ingest_dir.return_value = {"ingested": 0, "skipped": 0, "failed": 0, "chunks": 0}
        with mock.patch.object(sys, "argv", ["rag_build"] + argv), \
                mock.patch.object(rag_build, "RagKB", return_value=kb), \
                mock.patch.object(rag_build, "load_config", return_value=dict(DEFAULT_CONFIG)), \
                mock.patch.object(sys, "stdout", new_callable=io.StringIO):
            rag_build.main()
        return kb

    def test_query_only_does_not_ingest(self):
        """回归：--ingest 曾是 store_false，只传 --query 会默认进入入库分支。"""
        kb = self._run_main(["--query", "hello"])
        kb.ingest_dir.assert_not_called()
        kb.search.assert_called_once()

    def test_ingest_flag_triggers_ingest_with_repo_relative_dir(self):
        kb = self._run_main(["--ingest"])
        kb.ingest_dir.assert_called_once()
        passed_dir = kb.ingest_dir.call_args[0][0]
        self.assertTrue(passed_dir.endswith("pdf_downloads"))
        self.assertNotIn("..", os.path.basename(passed_dir))
        kb.search.assert_not_called()

    def test_no_args_prints_help_without_touching_kb(self):
        kb = self._run_main([])
        kb.ingest_dir.assert_not_called()
        kb.search.assert_not_called()


class DoiFilenameTest(unittest.TestCase):
    def test_traversal_and_absolute_path_neutralized(self):
        for doi in ("../../..\\Users\\Public\\evil",
                    "C:\\Windows\\Temp\\evil",
                    "/etc/passwd"):
            name = rp._doi_filename(doi)
            self.assertNotIn("/", name)
            self.assertNotIn("\\", name)
            self.assertNotIn(":", name)
            self.assertTrue(os.path.join("save_dir", name + ".pdf").startswith("save_dir"))

    def test_common_doi_compatible_with_old_scheme(self):
        # 旧实现只删 '/'；典型 DOI 无其他特殊字符，新旧文件名一致（缓存兼容）
        self.assertEqual(rp._doi_filename("10.1016/j.jmmm.2020.166652"),
                         "10.1016j.jmmm.2020.166652")

    def test_empty_fallback(self):
        self.assertEqual(rp._doi_filename("///"), "unnamed")


class SummarizeTextTest(unittest.TestCase):
    class _Client:
        def __init__(self, response):
            self.kwargs = {"temperature": 0.7, "max_tokens": 65536}
            self._response = response

        def chat(self, messages):
            return self._response

    def test_max_tokens_fallback_and_no_none_overwrite(self):
        cfg = {"max_tokens": 65536}  # 无 max_completion_tokens 键
        client = self._Client({"content": "摘要文本"})
        out = rp._summarize_text(client, cfg, "full text")
        self.assertEqual(out, "摘要文本")
        self.assertNotIn(None, client.kwargs.values())  # 回归：不得把 None 写进 kwargs
        self.assertEqual(client.kwargs.get("max_completion_tokens", 65536), 65536)

    def test_tool_call_response_rejected(self):
        client = self._Client({"content": "", "tool_calls": [{"id": "1"}]})
        with self.assertRaises(RuntimeError):
            rp._summarize_text(client, {"max_tokens": 100}, "text")

    def test_empty_response_rejected(self):
        client = self._Client({"content": "   "})
        with self.assertRaises(RuntimeError):
            rp._summarize_text(client, {"max_tokens": 100}, "text")


class ServerEnvTest(unittest.TestCase):
    def test_api_keys_forwarded(self):
        fake_env = {"ZHIPU_API_KEY": "k1", "UNPAYWALL_EMAIL": "a@b.c",
                    "SILICONFLOW_API_KEY": "", "PATH": "x"}
        with mock.patch.object(os, "environ", fake_env):
            env = tr._server_env()
        self.assertEqual(env["ZHIPU_API_KEY"], "k1")
        self.assertEqual(env["UNPAYWALL_EMAIL"], "a@b.c")
        self.assertNotIn("SILICONFLOW_API_KEY", env)  # 空值不透传
        self.assertEqual(env["PYTHONUTF8"], "1")

    def test_client_params_include_env(self):
        with mock.patch.dict(os.environ, {"ZHIPU_API_KEY": "kk"}):
            c = tr.MCPStdioClient()
        self.assertEqual(c._params.env.get("ZHIPU_API_KEY"), "kk")


class McpServerErrorWrappingTest(unittest.TestCase):
    def test_search_paper_exception_becomes_error_json(self):
        with mock.patch.object(ms, "_search_paper_impl",
                               side_effect=ConnectionError("crossref down")):
            out = json.loads(ms.search_paper("q"))
        self.assertIn("error", out)
        self.assertIn("crossref down", out["error"])

    def test_read_paper_exception_becomes_error_json(self):
        with mock.patch.object(ms, "_read_paper_impl",
                               side_effect=RuntimeError("boom")):
            out = json.loads(ms.read_paper([["t", "d"]]))
        self.assertIn("error", out)

    def test_read_paper_passthrough(self):
        with mock.patch.object(ms, "_read_paper_impl", return_value="[]") as impl:
            self.assertEqual(ms.read_paper([["t", "d"]]), "[]")


class RagConfigNamingTest(unittest.TestCase):
    """端点键名统一为 base_url：``api_host`` 已废弃，写了必须报错而不是被静默忽略。

    静默忽略的后果很隐蔽：`api_base_url` 留空 → 请求 URL 变成 ``/embeddings`` →
    报一个与"键名写错"毫无关系的 MissingSchema。
    """

    def _write(self, payload: dict) -> str:
        import tempfile
        tmp = tempfile.NamedTemporaryFile("w", suffix=".config", delete=False,
                                          encoding="utf-8")
        json.dump(payload, tmp)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    def test_default_config_uses_base_url(self):
        self.assertIn("api_base_url", DEFAULT_CONFIG)
        self.assertNotIn("api_host", DEFAULT_CONFIG)

    def test_base_url_key_is_accepted(self):
        path = self._write({"backend": "api",
                            "api_base_url": "https://api.siliconflow.cn/v1"})
        cfg = rag_kb.load_config(path)
        self.assertEqual(cfg["api_base_url"], "https://api.siliconflow.cn/v1")

    def test_renamed_api_host_key_is_rejected_with_a_hint(self):
        path = self._write({"backend": "api",
                            "api_host": "https://api.siliconflow.cn/v1"})
        with self.assertRaises(ValueError) as ctx:
            rag_kb.load_config(path)
        message = str(ctx.exception)
        self.assertIn("api_host", message)
        self.assertIn("api_base_url", message)      # 报错要自带改名方法

    def test_env_key_inference_follows_the_renamed_field(self):
        with mock.patch.dict(os.environ, {"SILICONFLOW_API_KEY": "env-key"}):
            self.assertEqual(
                rag_kb._env_key_for_base_url("https://api.siliconflow.cn/v1"),
                "env-key")
        self.assertEqual(rag_kb._env_key_for_base_url("https://unknown.example/v1"), "")


if __name__ == "__main__":
    unittest.main()
