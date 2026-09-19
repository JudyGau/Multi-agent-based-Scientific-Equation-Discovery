"""explain.md 的两条硬性要求（用户指定）与剪枝量化评估的回归测试。

要求一：**必须列出参考文献**
    - 正文用 [n] 标注、文末清单由系统从"知识库命中 ∪ 解释过程中的工具命中"生成；
    - LLM 自己写的参考文献一节要被替换掉（避免两份清单，也避免编造条目）；
    - 一条都没检索到时写明"本次未获取到可引用的文献"，而不是交给模型自由发挥。

要求二：**必须解释剪枝后的表达式，讲清剪掉了哪些项、为什么合理**
    - 提示词里同时给出剪枝前/后表达式、被移除项及敏感度、剪枝前后拟合对比；
    - ``find_best_eq`` 必须先剪枝再解释（否则解释拿不到剪枝结果）。

另含 ``prune_report`` 的量化对比（MSE/NMSE/逐点一致性）与训练数据定位的测试。
"""
from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

import sympy as sp

_FUNC = ("Variables:\n"
         "- Independents: x1, x2\n"
         "- Dependent: y\n"
         "def equation(x1, x2, params):\n"
         "    return params[0]*x1 + params[1]*x2 + params[2]\n")

_EXP = {
    "sample_order": "1",
    "equation": "def equation(x1, x2, params):\n    return params[0]*x1 + params[1]*x2 + params[2]",
    "thinking_content": "先线性项、后非线性项的推导……\n最后一行会被截掉",
}

_PRUNING = {
    "dependent": "y",
    "sym_names": ["x1", "x2"],
    "threshold": 0.1,
    "sample_range": (1, 6),
    "substituted_expr": "2.0*x1 + 3.0*x2 + 1.0",
    "pruned_expr": "2.0*x1 + 3.0*x2",
    "nodes_visited": 4,
    "nodes_pruned": 1,
    "prune_rate": 0.25,
    "removed": [{"kind": "term_of_Add", "term": "1.0", "sensitivity": 3.2e-05, "depth": 0}],
    "fit": {"n_points": 5, "mse_before": 1e-30, "mse_after": 2e-30,
            "nmse_before": 1e-31, "nmse_after": 2e-31, "max_abs_diff": 1e-15,
            "identical": False},
}

_RAG_REFS = [
    {"title": "Magnetorheological fluid compress mode", "doi": "10.1000/mrf.1",
     "source_file": "mrf1.pdf", "text": "MRF 压缩模式下的磁致应力……"},
    {"title": "Particle shape effect", "doi": "10.1000/mrf.2",
     "source_file": "mrf2.pdf", "text": "颗粒形状对磁致应力的影响……"},
]


def _make_experiment(root: pathlib.Path) -> None:
    """最小实验目录：样本、config_snapshot、训练 CSV。"""
    (root / "samples").mkdir(parents=True, exist_ok=True)
    (root / "samples" / "top01_samples_1.json").write_text(
        json.dumps({"score": -0.5, "sample_order": 1, "function": _FUNC,
                    "params": [2.0, 3.0, 1.0]}), encoding="utf-8")
    (root / "config_snapshot.json").write_text(
        json.dumps({"data_csv": "data/tiny/train.csv"}), encoding="utf-8")
    data_dir = root / "data" / "tiny"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = ["x1,x2,y"]
    for x1 in (1.0, 2.0, 3.0, 4.0, 5.0):
        x2 = 6.0 - x1
        rows.append(f"{x1},{x2},{2.0 * x1 + 3.0 * x2 + 1.0}")
    (data_dir / "train.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


class PruningPromptTest(unittest.TestCase):
    """解释提示词必须覆盖剪枝后的表达式与剪枝过程。"""

    def _prompt(self, pruning=_PRUNING):
        from drsr_420.analysis import explain as explain_mod

        with mock.patch.object(explain_mod, "retrieve_rag", return_value=[]):
            return explain_mod.build_explain_content(_FUNC, _EXP, pruning=pruning)

    def test_prompt_contains_pruned_expression_and_removed_terms(self):
        prompt = self._prompt()
        self.assertIn("2.0*x1 + 3.0*x2", prompt, "必须给出剪枝后的表达式")
        self.assertIn("2.0*x1 + 3.0*x2 + 1.0", prompt, "必须给出剪枝前的表达式")
        self.assertIn("被移除的项", prompt)
        self.assertIn("3.200e-05", prompt, "被移除项的敏感度必须一并给出")
        self.assertIn("term_of_Add", prompt)

    def test_prompt_contains_fit_comparison(self):
        prompt = self._prompt()
        self.assertIn("剪枝前后的拟合对比", prompt)
        self.assertIn("5 个数据点", prompt)

    def test_prompt_requires_pruning_rationale_structure(self):
        prompt = self._prompt()
        self.assertIn("剪枝后表达式的逐项力学解释", prompt)
        self.assertIn("剪枝过程去掉了哪些项", prompt)
        self.assertIn("剪枝合理性的论证", prompt)

    def test_prompt_states_when_nothing_was_pruned(self):
        pruning = dict(_PRUNING, nodes_pruned=0, prune_rate=0.0, removed=[],
                       pruned_expr=_PRUNING["substituted_expr"])
        prompt = self._prompt(pruning)
        self.assertIn("没有移除任何项", prompt)

    def test_prompt_without_pruning_says_so(self):
        prompt = self._prompt(pruning=None)
        self.assertIn("本次没有得到剪枝结果", prompt)

    def test_prompt_contains_numbered_reference_list(self):
        from drsr_420.analysis import explain as explain_mod

        with mock.patch.object(explain_mod, "retrieve_rag", return_value=_RAG_REFS):
            prompt = explain_mod.build_explain_content(_FUNC, _EXP, pruning=_PRUNING)
        self.assertIn("[1] Magnetorheological fluid compress mode", prompt)
        self.assertIn("[2] Particle shape effect", prompt)

    def test_rag_context_is_numbered_like_the_reference_list(self):
        from drsr_420.analysis import explain as explain_mod

        with mock.patch.object(explain_mod, "retrieve_rag", return_value=_RAG_REFS):
            prompt = explain_mod.build_explain_content(_FUNC, _EXP, pruning=_PRUNING)
        self.assertIn("【文献 1】Magnetorheological fluid compress mode", prompt)
        self.assertIn("【文献 2】Particle shape effect", prompt)


class ReferenceSectionTest(unittest.TestCase):
    """文末参考文献清单：机器生成、去重、替换 LLM 自编清单。"""

    def test_render_lists_numbered_entries_with_doi(self):
        from drsr_420.analysis import explain as explain_mod

        section = explain_mod.render_reference_section(_RAG_REFS)
        self.assertIn("## 参考文献", section)
        self.assertIn("[1] Magnetorheological fluid compress mode", section)
        self.assertIn("DOI: 10.1000/mrf.2", section)
        self.assertIn("知识库来源: mrf1.pdf", section)

    def test_render_dedupes_repeated_chunks_of_one_paper(self):
        from drsr_420.analysis import explain as explain_mod

        chunks = [{"title": "T", "doi": "10.1/a", "text": f"片段{i}", "source_file": "a.pdf"}
                  for i in range(3)]
        section = explain_mod.render_reference_section(chunks)
        self.assertEqual(section.count("[1]"), 1)
        self.assertNotIn("[2]", section)

    def test_render_with_empty_refs_states_it_explicitly(self):
        from drsr_420.analysis import explain as explain_mod

        section = explain_mod.render_reference_section([])
        self.assertIn("## 参考文献", section)
        self.assertIn("本次未获取到可引用的文献", section)

    def test_entry_falls_back_to_source_file_when_title_is_a_guessed_doi(self):
        """知识库的 title 有时就是"从文件名回推的 DOI"，此时改用 PDF 名更好追溯。"""
        from drsr_420.analysis import explain as explain_mod

        entry = explain_mod.format_reference_entry({
            "title": "10.216561000/-0887.380021", "doi": "10.216561000/-0887.380021",
            "source_file": "10.216561000-0887.380021.pdf"})
        self.assertTrue(entry.startswith("10.216561000-0887.380021.pdf."), entry)
        self.assertIn("DOI: 10.216561000/-0887.380021", entry)

    def test_merge_dedupes_by_doi_then_title(self):
        from drsr_420.analysis import explain as explain_mod

        dup_doi = {"title": "另一个题名", "doi": "10.1000/MRF.1"}
        dup_title = {"title": "  particle SHAPE effect ", "doi": ""}
        merged = explain_mod.merge_references(_RAG_REFS, [dup_doi, dup_title, {}])
        self.assertEqual(len(merged), 2)

    def test_merge_combines_chunks_of_the_same_paper(self):
        """同一 PDF 的多个片段要合并（正文拼接、字段互补），编号才对得上清单。"""
        from drsr_420.analysis import explain as explain_mod

        chunks = [
            {"title": "T", "doi": "", "text": "第一段", "source_file": "a.pdf"},
            {"title": "", "doi": "10.1/a", "text": "第二段", "source_file": "a.pdf"},
        ]
        merged = explain_mod.merge_references(chunks)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["doi"], "10.1/a", "缺失字段应从后续片段补齐")
        self.assertIn("第一段", merged[0]["text"])
        self.assertIn("第二段", merged[0]["text"])

    def test_prompt_numbering_matches_the_reference_list(self):
        from drsr_420.analysis import explain as explain_mod

        chunks = [
            {"title": "P1", "doi": "10.1/a", "text": "片段甲", "source_file": "a.pdf"},
            {"title": "P1", "doi": "10.1/a", "text": "片段乙", "source_file": "a.pdf"},
            {"title": "P2", "doi": "10.1/b", "text": "片段丙", "source_file": "b.pdf"},
        ]
        with mock.patch.object(explain_mod, "retrieve_rag", return_value=chunks):
            prompt = explain_mod.build_explain_content(_FUNC, _EXP, pruning=_PRUNING)
        self.assertIn("【文献 1】P1", prompt)
        self.assertIn("【文献 2】P2", prompt)
        self.assertNotIn("【文献 3】", prompt)
        self.assertIn("[2] P2", prompt)
        self.assertNotIn("[3] P2", prompt)

    def test_strip_replaces_model_written_reference_section(self):
        from drsr_420.analysis import explain as explain_mod

        answer = ("正文提到参考文献中的结论一致。\n\n"
                  "## 一、材料体系\n磁流变液。\n\n"
                  "## 参考文献\n[1] 模型自己编的文献\n")
        body = explain_mod._strip_reference_section(answer)
        self.assertIn("正文提到参考文献中的结论一致", body)
        self.assertIn("## 一、材料体系", body)
        self.assertNotIn("模型自己编的文献", body)

    def test_collect_tool_refs_parses_search_paper_and_kb(self):
        from drsr_420.analysis import explain as explain_mod

        refs: list = []
        explain_mod._collect_tool_refs(refs, "search_paper", {}, json.dumps([
            {"title": "T1", "doi": "10.1/a", "journal": "J", "year": 2020, "authors": ["A B"]}]))
        explain_mod._collect_tool_refs(refs, "search_kb", {}, json.dumps([
            {"title": "T2", "doi": "10.1/b", "source_file": "b.pdf"}]))
        explain_mod._collect_tool_refs(refs, "read_paper",
                                       {"title_doi": [["T3", "10.1/c"]]}, "{}")
        self.assertEqual([r["doi"] for r in refs], ["10.1/a", "10.1/b", "10.1/c"])
        self.assertEqual(refs[0]["journal"], "J")
        explain_mod._collect_tool_refs(refs, "search_paper", {}, "not-json")  # 静默跳过
        self.assertEqual(len(refs), 3)

    def test_explain_best_sample_appends_authoritative_references(self):
        from drsr_420.analysis import explain as explain_mod

        def fake_re_act(client, content, tool_refs=None):
            if tool_refs is not None:
                tool_refs.append({"title": "工具检索到的文献", "doi": "10.1000/tool.1"})
            return "正文……\n\n## 参考文献\n[1] LLM 编造的条目"

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_experiment(root)
            (root / "experiences.json").write_text(
                json.dumps({"Good": [_EXP]}), encoding="utf-8")
            with mock.patch.object(explain_mod, "retrieve_rag", return_value=_RAG_REFS), \
                 mock.patch.object(explain_mod, "explain_re_act", fake_re_act), \
                 mock.patch("builtins.print"):
                explain_mod.explain_best_sample(
                    str(root), _FUNC, "1", role_clients=_role_clients())
            written = (root / "explain.md").read_text(encoding="utf-8")

        self.assertIn("正文……", written)
        self.assertNotIn("LLM 编造的条目", written, "LLM 自编的参考文献必须被替换")
        self.assertIn("[1] Magnetorheological fluid compress mode", written)
        self.assertIn("DOI: 10.1000/mrf.2", written)
        self.assertIn("[3] 工具检索到的文献", written, "解释过程中工具检索到的文献也要列出")

    def test_explain_best_sample_without_any_reference_writes_note(self):
        from drsr_420.analysis import explain as explain_mod

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_experiment(root)
            (root / "experiences.json").write_text(
                json.dumps({"Good": [_EXP]}), encoding="utf-8")
            with mock.patch.object(explain_mod, "retrieve_rag", return_value=[]), \
                 mock.patch.object(explain_mod, "explain_re_act",
                                   lambda client, content, tool_refs=None: "正文……"), \
                 mock.patch("builtins.print"):
                explain_mod.explain_best_sample(
                    str(root), _FUNC, "1", role_clients=_role_clients())
            written = (root / "explain.md").read_text(encoding="utf-8")

        self.assertIn("本次未获取到可引用的文献", written)

    def test_llm_failure_does_not_overwrite_existing_explain_md(self):
        """LLM 调用失败时保留既有 explain.md：覆盖成"只剩参考文献"的残件会掩盖故障。"""
        from drsr_420.analysis import explain as explain_mod

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_experiment(root)
            (root / "experiences.json").write_text(
                json.dumps({"Good": [_EXP]}), encoding="utf-8")
            (root / "explain.md").write_text("上一版解释", encoding="utf-8")
            with mock.patch.object(explain_mod, "retrieve_rag", return_value=_RAG_REFS), \
                 mock.patch.object(explain_mod, "explain_re_act",
                                   lambda client, content, tool_refs=None: None), \
                 mock.patch("builtins.print") as printer:
                explain_mod.explain_best_sample(
                    str(root), _FUNC, "1", role_clients=_role_clients())
            written = (root / "explain.md").read_text(encoding="utf-8")
            printed = "\n".join(str(c.args[0]) for c in printer.call_args_list if c.args)

        self.assertEqual(written, "上一版解释")
        self.assertIn("保留既有 explain.md 不覆盖", printed)


class PruneReportTest(unittest.TestCase):
    """剪枝量化评估：定位训练数据 + 剪枝前后拟合对比。"""

    def test_compare_fits_reports_identical_for_same_expression(self):
        from drsr_420.analysis import prune_report as pr

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_experiment(root)
            data = pr.load_training_data(str(root))
            expr = 2.0 * sp.Symbol("x1") + 3.0 * sp.Symbol("x2") + 1.0
            fit = pr.compare_fits("y", ["x1", "x2"], data, expr, expr)
            self.assertEqual(fit["n_points"], 5)
            self.assertLess(fit["mse_before"], 1e-20)
            self.assertTrue(fit["identical"])
            self.assertIn("逐点完全相同", pr.format_fit_summary(fit))

    def test_compare_fits_quantifies_change_after_pruning(self):
        from drsr_420.analysis import prune_report as pr

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_experiment(root)
            data = pr.load_training_data(str(root))
            x1, x2 = sp.symbols("x1 x2")
            fit = pr.compare_fits("y", ["x1", "x2"], data, 2.0 * x1 + 3.0 * x2 + 1.0,
                                  2.0 * x1 + 3.0 * x2)   # 去掉常数项 1.0
            self.assertGreater(fit["mse_after"], fit["mse_before"])
            self.assertAlmostEqual(fit["max_abs_diff"], 1.0, places=9)
            self.assertIn("相对变化", pr.format_fit_summary(fit))

    def test_load_training_data_missing_snapshot_returns_none(self):
        from drsr_420.analysis import prune_report as pr

        with tempfile.TemporaryDirectory() as tmp, mock.patch("builtins.print"):
            self.assertIsNone(pr.load_training_data(tmp))


class FindBestEqOrderTest(unittest.TestCase):
    """顺序契约：先剪枝、再解释，且剪枝摘要必须传到解释阶段。"""

    def test_prune_runs_before_explain_and_summary_is_passed(self):
        from drsr_420.analysis import find_best_eq as fbe

        calls: list = []
        pruning = {"nodes_pruned": 1, "removed": [{"kind": "term_of_Add", "term": "1.0"}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_experiment(root)
            with mock.patch.object(fbe, "prune_and_visualize",
                                   lambda *a, **k: calls.append("prune") or pruning), \
                 mock.patch.object(fbe, "explain_best_sample",
                                   lambda *a, **k: calls.append(("explain", k.get("pruning")))), \
                 mock.patch("builtins.print"):
                fbe.find_best_eq(str(root))

        self.assertEqual(calls[0], "prune", "必须先剪枝，解释才能覆盖剪枝结果")
        self.assertEqual(calls[1], ("explain", pruning))

    def test_prune_and_visualize_returns_summary_dict(self):
        from drsr_420.analysis.find_best_eq import prune_and_visualize

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_experiment(root)
            with mock.patch("builtins.print"):
                summary = prune_and_visualize(str(root), _FUNC, [2.0, 3.0, 1.0],
                                              threshold=0.1, sample_range=(1, 6))
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["dependent"], "y")
        self.assertEqual(summary["sym_names"], ["x1", "x2"])
        self.assertIn("substituted_expr", summary)
        self.assertIn("fit", summary)
        self.assertEqual(summary["prune_rate"], summary["nodes_pruned"] / summary["nodes_visited"])


def _role_clients():
    from drsr_420.llm.role_clients import RoleClients

    class _Client:
        model = "fake/model"

        def _provider_name(self):
            return "fake"

        kwargs = {}

    return RoleClients.single(_Client())


class ExplainReActToolCapTest(unittest.TestCase):
    """``explain_re_act`` 必须有工具轮次上限。

    此前是 ``while True`` 且没有任何上限：模型只要一直发起工具调用，收尾解释就永远
    不返回（采样侧的 ToolCallerAgent 一直有兜底，这里漏了）。
    """

    class _LoopingClient:
        """每一轮都发起工具调用，永不给出最终答复。"""

        def chat_stream(self, messages):
            yield {
                "tool_calls": [{"id": "c1",
                                "function": {"name": "search_kb", "arguments": "{}"}}],
                "content": "PARTIAL",
                "reasoning_content": "",
                "final": True,
            }

    def test_cap_breaks_the_tool_loop(self):
        from drsr_420.analysis import explain as explain_mod

        with mock.patch.object(explain_mod, "mcp_call_tool", return_value="R") as call:
            out = explain_mod.explain_re_act(self._LoopingClient(), "CONTENT",
                                             max_tool_rounds=2)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(out, "PARTIAL")


if __name__ == "__main__":
    unittest.main()
