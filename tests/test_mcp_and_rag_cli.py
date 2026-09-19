"""第 8 轮（工具/MCP/RAG CLI）加固回归测试。

覆盖：
- rag_build CLI：--ingest 方向不再反转（只 --query 不得触发入库）；--dir 按项目根解析；
- rag_kb 配置键名与端点写法：端点统一为 ``api_base_url``，旧的 ``api_host`` 写了就报错
  （含改名提示）；``backend="api"`` 时必须给**完整 URL**，裸主机域名/空值同样报错；
- rag_kb.chunk_text：超长段硬切不再重复当前块、硬切片段之间保留 overlap；
- read_paper._doi_filename：DOI 路径穿越/绝对路径净化，且常见 DOI 文件名向后兼容；
- read_paper 标题校验：编造的 (标题, DOI) 配对必须回传"标题不符"而不是无关论文摘要，
  且合法配对不得被误杀（对真实下错的 PDF 做回归）；
- read_paper._summarize_text：max_tokens 回退、工具调用/空内容显式报错；
- read_paper._stats_to_stderr：MCP 子进程里 LLM 统计改道 stderr（否则被 stdout 块缓冲吞掉）；
- tool_runner._server_env：提供商 API key 环境变量并入 MCP 子进程，实验目录（DRSR_ 前缀）透传；
- mcp_server.attach_server_stderr：子进程把 stderr 也旁路进 run.err（否则摘要统计只闪在终端）；
- mcp_server：底层异常被包装为 {"error": ...} JSON（SDK 会吞掉抛出型错误的文本）。
"""
import io
import json
import os
import sys
import tempfile
import unittest
import contextlib
from unittest import mock

from drsr_420.knowledge import rag_build
from drsr_420.knowledge import rag_kb
from drsr_420.knowledge.rag_kb import chunk_text, DEFAULT_CONFIG
from drsr_420.knowledge.tools import read_paper as rp
from drsr_420.knowledge.tools import mcp_server as ms
from drsr_420.knowledge import tool_runner as tr


class ChunkTextTest(unittest.TestCase):
    def test_long_para_does_not_duplicate_pending_chunk(self):
        # 正文用小写：全大写短行会被小节标题启发式识别为标题（那是新语义，见下）
        text = "short para aaa\n\n" + "X" * 1200 + "\n\ntail para bbb"
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        # 回归：旧实现在硬切循环里把 pending 的 current 反复 append 又不清空
        self.assertEqual(sum(c == "short para aaa" for c in chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 500)
        self.assertTrue(any("tail para bbb" in c for c in chunks))

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


class SectionChunkingTest(unittest.TestCase):
    """小节优先的语义分块：检索命中的片段自带"论文哪一节"的语境。"""

    def test_numbered_sections_become_separate_chunks(self):
        text = ("1. Introduction\n\nWe study magnetorheological fluids.\n\n"
                "2. Methods\n\nThe particles were measured.\n\n"
                "3. Conclusions\n\nShape ratios matter.")
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(chunks[0].startswith("1. Introduction"))
        self.assertIn("magnetorheological fluids", chunks[0])
        self.assertNotIn("Shape ratios", chunks[0])          # 小节之间不串块
        self.assertTrue(chunks[2].startswith("3. Conclusions"))

    def test_markdown_headings_respected(self):
        text = "## Methods\n\nBody A.\n\n## Results\n\nBody B."
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertEqual([c.split("\n", 1)[0] for c in chunks], ["## Methods", "## Results"])

    def test_overlong_section_splits_with_heading_prefix(self):
        heading = "2. Methods"
        paras = [f"Paragraph {i} " + "x" * 120 for i in range(8)]   # ~970 chars
        text = heading + "\n\n" + "\n\n".join(paras)
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 500)                     # 标题计入预算
            self.assertTrue(c.startswith(heading), "每个子块都必须带小节标题前缀")
        joined = "\n".join(chunks)
        for i in (0, 7):
            self.assertIn(f"Paragraph {i}", joined)               # 内容无丢失

    def test_page_marker_is_not_a_heading(self):
        text = "1. Introduction\n\nBody text here.\n--- Page 2 ---\nMore body text."
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertEqual(len(chunks), 1)
        self.assertIn("--- Page 2 ---", chunks[0])                # 页码标记留在正文里

    def test_long_list_item_is_not_a_heading(self):
        long_item = ("1. The magnetorheological effect describes the field-induced "
                     "yield stress increase of magnetorheological fluids under shear.")
        text = long_item + "\n\nFollow-up paragraph."
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertEqual(len(chunks), 1)                          # >80 字符的编号行是正文

    def test_no_headings_falls_back_to_paragraph_merge(self):
        text = "A B\n\nC D\n\nE F"
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertEqual(chunks, ["A B\nC D\nE F"])               # 旧回退行为不变

    def test_numeric_junk_lines_are_not_headings(self):
        """回归：PDF 提取的公式/页码碎片（"1 2"、"0. 8"、"0\\tH"）不得当标题，
        否则重建后出现一堆 3 字符的垃圾块（实测占 15%）。"""
        text = "1. Introduction\n\nReal body text.\n1 2\n0. 8\n0\tH\n47 4"
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertEqual(len(chunks), 1)              # 碎片只是正文，不另立小节
        self.assertTrue(chunks[0].startswith("1. Introduction"))

    def test_heading_only_section_keeps_the_heading(self):
        text = "1. Introduction\n\nBody.\n\n2. Appendix\n"
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertIn("2. Appendix", chunks)

    def test_hard_split_fragments_overlap(self):
        text = "".join(str(i % 10) for i in range(1300))
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertEqual(chunks[0][-50:], chunks[1][:50])


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


class _FakePage:
    """pymupdf Page 的最小替身：只需 get_text() 与 get_text("dict")。"""

    def __init__(self, text, blocks):
        self._text = text
        self._blocks = blocks

    def get_text(self, kind=None):
        return {"blocks": self._blocks} if kind == "dict" else self._text


class _FakeDoc:
    """pymupdf Document 的最小替身。"""

    def __init__(self, page_text, blocks=None, metadata=None):
        self.metadata = metadata or {}
        self._page = _FakePage(page_text, blocks or [])

    def load_page(self, _n):
        return self._page

    def close(self):
        pass


#: 真实下错的两对 (PDF 文件, 当时的请求标题)：DOI 有效但不是请求的那篇，
#: 见 run.out L7193 与 pdf_downloads/ 里 14:46 下载的这两份 PDF。
_REAL_MISMATCHED_PDFS = (
    ("pdf_downloads/10.10631.3480551.pdf",
     "Effect of particle shape in magnetorheology"),
    ("pdf_downloads/10.10880964-1726212025014.pdf",
     "Effect of particle aspect ratio in magnetorheology"),
)


class PaperTitleGuardTest(unittest.TestCase):
    """read_paper 必须校验"DOI 下到的 PDF 是不是请求的那篇"。

    历史缺陷：只按 DOI 取 PDF、从不看内容，于是 LLM 编造的 (标题, DOI) 配对会把
    完全无关的论文摘要当文献证据喂进上下文（实测 8073 字符的无关摘要，采样者随后
    写下 "tool outputs were irrelevant"）。
    """

    def test_mismatched_title_is_rejected_with_the_actual_title(self):
        doc = _FakeDoc("Fabry-Perot interferometer utilized for displacement measurement "
                       "in a large measuring range. Rev. Sci. Instrum. 81, 093102 (2010)")
        notice = rp._verify_pdf_title(
            doc, "Effect of particle shape in magnetorheology", "10.1063/1.3480551")
        self.assertIsNotNone(notice)
        self.assertIn("标题不符", notice)
        self.assertIn("Fabry-Perot", notice)      # 报错要带上"实际下到的是什么"
        self.assertIn("10.1063/1.3480551", notice)

    def test_matching_title_passes(self):
        doc = _FakeDoc("Effect of particle shape in magnetorheology. "
                       "Journal of Applied Physics 108, 093102 (2010)")
        self.assertIsNone(rp._verify_pdf_title(
            doc, "Effect of particle shape in magnetorheology", "10.1063/1.3480551"))

    def test_doi_passed_as_title_is_skipped(self):
        # 模型有时把 DOI 本身当标题传（实测 3 例）：无从校验，放行而不是误杀
        doc = _FakeDoc("Magnetorheology of suspensions")
        self.assertIsNone(rp._verify_pdf_title(doc, "10.11221/.3005402", "10.11221/.3005402"))

    def test_title_without_letters_is_skipped(self):
        doc = _FakeDoc("Some unrelated optics paper")
        self.assertIsNone(rp._verify_pdf_title(doc, "-----", "10.x"))

    def test_single_word_title_still_checked(self):
        # 单 token 标题也要判：命中通过、不命中拒绝（覆盖率阈值对短标题同样成立）
        self.assertIsNone(rp._verify_pdf_title(
            _FakeDoc("Magnetorheology of suspensions"), "Magnetorheology", "10.x"))
        self.assertIsNotNone(rp._verify_pdf_title(
            _FakeDoc("Fabry-Perot interferometer utilized for displacement"), "Magnetorheology", "10.x"))

    def test_largest_font_line_is_used_as_title_candidate(self):
        blocks = [{"lines": [
            {"spans": [{"text": "Real Paper Title", "size": 18.0}]},
            {"spans": [{"text": "some author name", "size": 9.0}]},
        ]}]
        doc = _FakeDoc("", blocks=blocks, metadata={"title": "Microsoft Word - draft.doc"})
        self.assertEqual(rp._first_page_title_candidate(doc), "Real Paper Title")

    def test_read_paper_returns_notice_instead_of_summary(self):
        """本地缓存命中路径：标题不符时不得调用摘要，且返回值与请求一一对齐。"""
        with tempfile.TemporaryDirectory() as tmp:
            doi = "10.1063/1.3480551"
            with open(os.path.join(tmp, rp._doi_filename(doi) + ".pdf"), "wb"):
                pass
            doc = _FakeDoc("Fabry-Perot interferometer utilized for displacement measurement")
            with mock.patch.object(rp.pymupdf, "open", return_value=doc), \
                    mock.patch.object(rp, "_summarize_text") as summarize:
                out = json.loads(rp.read_paper(
                    [["Effect of particle shape in magnetorheology", doi]], save_dir=tmp))
            self.assertEqual(len(out), 1)           # 数组与请求条目对齐
            self.assertIn("标题不符", out[0])
            summarize.assert_not_called()           # 无关 PDF 不进 LLM


class RealPdfTitleGuardTest(unittest.TestCase):
    """对真实下错的 PDF 做回归；文件不存在时跳过，不阻塞无文献环境。"""

    @unittest.skipUnless(os.path.exists(_REAL_MISMATCHED_PDFS[0][0]),
                         "缺少真实 PDF 夹具")
    def test_fabry_perot_pdf_rejected_for_shape_title(self):
        path, title = _REAL_MISMATCHED_PDFS[0]
        doc = rp.pymupdf.open(path)
        try:
            notice = rp._verify_pdf_title(doc, title, "10.1063/1.3480551")
        finally:
            doc.close()
        self.assertIsNotNone(notice)
        self.assertIn("Fabry", notice)

    @unittest.skipUnless(os.path.exists(_REAL_MISMATCHED_PDFS[1][0]),
                         "缺少真实 PDF 夹具")
    def test_damper_pdf_rejected_for_aspect_ratio_title(self):
        path, title = _REAL_MISMATCHED_PDFS[1]
        doc = rp.pymupdf.open(path)
        try:
            notice = rp._verify_pdf_title(doc, title, "10.1088/0964-1726/21/2/025014")
        finally:
            doc.close()
        self.assertIsNotNone(notice)
        self.assertIn("self-sensing", notice)


class SummarizeTextTest(unittest.TestCase):
    class _Client:
        def __init__(self, response):
            self.kwargs = {"temperature": 0.7, "max_tokens": 65536}
            self._response = response

        def chat(self, messages):
            return self._response

    class _StatsClient:
        """模拟 llm.client：每次 chat 都会 print 一段 token/耗时统计。"""

        def __init__(self):
            self.kwargs = {"max_tokens": 65536}

        def chat(self, messages):
            print("[glm][glm-5.3-flash] 第1次\n"
                  "本次 tokens：prompt=3903, thinking=344, content=239, total=4486")
            return {"content": "摘要文本"}

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


class StatsToStderrTest(unittest.TestCase):
    """MCP 子进程里 LLM 统计必须落在 stderr 上。

    stdio 传输下子进程的 ``sys.stdout`` 是块缓冲的协议管道（SDK 之后虽把 fd 1 指到
    stderr，Python 层缓冲模式不变），直接 print 的统计会滞留缓冲区、随强杀一起丢失。
    """

    def _call_in_child(self):
        """在"看起来像 MCP 子进程"的环境里跑一次摘要，返回 (stdout, stderr) 文本。"""
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {rp.MCP_SERVER_ENV: "1"}):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rp._summarize_text(SummarizeTextTest._StatsClient(),
                                   {"max_tokens": 65536}, "full text")
                # 改道只作用于本次调用：返回后 stdout 必须复原，不能把整个子进程的
                # print 永久改到 stderr
                self.assertIs(sys.stdout, out)
        return out.getvalue(), err.getvalue()

    def test_stats_go_to_stderr_in_child(self):
        out, err = self._call_in_child()
        self.assertIn("第1次", err)
        self.assertIn("本次 tokens", err)
        self.assertEqual(out, "")

    def test_stats_stay_on_stdout_outside_child(self):
        """本进程（CLI/测试）直接调用时仍走 stdout，好让 run.out 的 tee 收得到。"""
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(rp.MCP_SERVER_ENV, None)
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rp._summarize_text(SummarizeTextTest._StatsClient(),
                                   {"max_tokens": 65536}, "full text")
        self.assertIn("本次 tokens", out.getvalue())
        self.assertEqual(err.getvalue(), "")

    def test_mark_server_process_sets_the_flag_read_paper_reads(self):
        """mcp_server 启动时的标记必须正是 read_paper 判定的那个变量名。"""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(rp.MCP_SERVER_ENV, None)
            ms._mark_server_process()
            self.assertEqual(os.environ.get(rp.MCP_SERVER_ENV), "1")


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

    def test_results_root_forwarded(self):
        # 实验目录靠 DRSR_ 前缀透传：MCP 子进程据此把自己的 stderr 旁路进 run.err
        with mock.patch.dict(os.environ, {ms.RESULTS_ROOT_ENV: r"C:\tmp\run1"}):
            env = tr._server_env()
            c = tr.MCPStdioClient()
        self.assertEqual(env.get(ms.RESULTS_ROOT_ENV), r"C:\tmp\run1")
        self.assertEqual(c._params.env.get(ms.RESULTS_ROOT_ENV), r"C:\tmp\run1")


class ServerStderrTeeTest(unittest.TestCase):
    """MCP 子进程的 stderr 必须同时落进实验目录的 run.err。

    子进程的 fd 2 由父进程的 ``errlog`` 决定（只到控制台）；``run.err`` 是主进程在
    Python 层替换 ``sys.stderr`` 得到的，子进程看不到——文献总结的 token/耗时统计
    因此只闪在终端里，产物里查不到（"摘要用了哪个模型"无法事后核实）。
    """

    def setUp(self):
        self._orig = sys.stderr
        self.addCleanup(setattr, sys, "stderr", self._orig)

    def _attach(self, root):
        with mock.patch.dict(os.environ, {ms.RESULTS_ROOT_ENV: str(root)}):
            ms.attach_server_stderr()

    def test_stderr_mirrored_into_run_err(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            console = io.StringIO()
            sys.stderr = console
            self._attach(tmp)
            sys.stderr.write("[dashscope][qwen3.7-flash] 第1次\n")
            sys.stderr.flush()
            tee = sys.stderr
            body = open(os.path.join(tmp, "run.err"), encoding="utf-8").read()
            tee.close()
        self.assertIn("[dashscope][qwen3.7-flash] 第1次", body)   # 落进产物
        self.assertIn("[dashscope][qwen3.7-flash] 第1次", console.getvalue())  # 控制台不丢

    def test_attach_is_idempotent(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            sys.stderr = io.StringIO()
            self._attach(tmp)
            first = sys.stderr
            self._attach(tmp)                 # 第二次不得再套一层
            self.assertIs(sys.stderr, first)
            sys.stderr.close()

    def test_missing_env_is_noop(self):
        console = io.StringIO()
        sys.stderr = console
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ms.RESULTS_ROOT_ENV, None)
            ms.attach_server_stderr()
        self.assertIs(sys.stderr, console)

    def test_unusable_root_degrades_silently(self):
        console = io.StringIO()
        sys.stderr = console
        self._attach("\x00:/nowhere")         # 非法路径：open 抛 ValueError
        self.assertIs(sys.stderr, console)


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

    def test_api_backend_requires_a_complete_base_url(self):
        """空端点拼出的 URL 是 ``/embeddings``，报错会与"配置写错"毫无关系——在这里拦住。"""
        path = self._write({"backend": "api"})
        with self.assertRaises(ValueError) as ctx:
            rag_kb.load_config(path)
        self.assertIn("api_base_url", str(ctx.exception))

    def test_api_backend_rejects_bare_hostname(self):
        """与 LLM 侧同一条规则：端点必须是完整 URL，不写裸主机域名。"""
        path = self._write({"backend": "api",
                            "api_base_url": "api.siliconflow.cn/v1"})
        with self.assertRaises(ValueError) as ctx:
            rag_kb.load_config(path)
        self.assertIn("https://", str(ctx.exception))

    def test_local_backend_ignores_the_endpoint(self):
        """local 后端根本不用端点，缺它/写它都不该被拦。"""
        path = self._write({"backend": "local"})
        self.assertEqual(rag_kb.load_config(path)["backend"], "local")

    def test_env_key_inference_follows_the_renamed_field(self):
        with mock.patch.dict(os.environ, {"SILICONFLOW_API_KEY": "env-key"}):
            self.assertEqual(
                rag_kb._env_key_for_base_url("https://api.siliconflow.cn/v1"),
                "env-key")
        self.assertEqual(rag_kb._env_key_for_base_url("https://unknown.example/v1"), "")


class KnowledgeMetadataInferenceTest(unittest.TestCase):
    """入库元数据的推断链：不得把文件名或 DOI 冒充标题。

    实测 explain.md 的参考文献标题显示成 ``10.216561000-0887.380021.pdf``、
    ``10.11221.3005402.pdf``——那是历史实现 ``title = title or doi or stem`` 与
    "把去斜杠文件名当 DOI 恢复"共同造成的：文件名抹掉的是哪个 ``/`` 无从判断，
    恢复出来的是假 DOI。
    """

    def test_ambiguous_filename_doi_is_refused(self):
        # 后缀含 '-'：既可能是 DOI 自带的连字符、也可能是被抹掉的 '/'，无法唯一还原
        self.assertIsNone(rag_kb._recover_doi("10.216561000-0887.380021"))
        self.assertIsNone(rag_kb._recover_doi("10.10880964-17262412125005"))

    def test_unambiguous_filename_doi_is_recovered(self):
        self.assertEqual(rag_kb._recover_doi("10.1016j.jmmm.2020.166652"),
                         "10.1016/j.jmmm.2020.166652")
        self.assertEqual(rag_kb._recover_doi("10.1002smll.202410011"),
                         "10.1002/smll.202410011")

    def test_printed_doi_in_pdf_text_wins(self):
        text = "Journal of Magnetics 21(2)\nhttp://dx.doi.org/10.4283/JMAG.2016.21.2.244\n"
        self.assertEqual(rag_kb.doi_from_pdf_text(text), "10.4283/JMAG.2016.21.2.244")
        self.assertIsNone(rag_kb.doi_from_pdf_text("no doi in this text"))

    def test_title_never_falls_back_to_a_doi(self):
        self.assertEqual(rag_kb._resolve_title("", "", "", "some_paper_name"),
                         "some_paper_name")
        # 文件名是 DOI 形态时宁可留空，也不写成标题
        self.assertEqual(rag_kb._resolve_title("", "", "", "10.216561000-0887.380021"), "")

    def test_page_title_beats_placeholder_metadata(self):
        self.assertEqual(
            rag_kb._resolve_title("", "Microsoft Word - x.doc", "Real Paper Title", "10.1/2"),
            "Real Paper Title")
        self.assertEqual(rag_kb._resolve_title("Given", "meta", "page", "stem"), "Given")


if __name__ == "__main__":
    unittest.main()
