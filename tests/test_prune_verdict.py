"""剪枝实质判定：真剪枝 / 只是 simplify 换了写法（通分/展开）/ 什么都没做。

用户提出的问题：剪枝模块有时只是把原表达式**分母通分**，公式数学上根本没变——这种情形
不该被当成"剪枝结果"发布出去。契约有三条：

1. ``SensitivityPruner.prune`` 在**没有真移除任何项**（``nodes_pruned == 0``）时返回
   **原表达式本身**（``stats.simplified_expr`` 只作诊断，用来说明形式重排的规模）；
2. ``prune_report.classify_pruning`` 给出判定与证据，且"公式到底变没变"用**有效定义域
   上的数值比较**判定：``sp.simplify(a - b) == 0`` 与 ``a.equals(b)`` 在含浮点指数的
   表达式上会给**假阴性**（实测某次 ``equals() == False``，而 8 个数据点 + 采样点上
   相对差恰为 0 —— 见下面的真实样本回归）；
3. ``prune_and_visualize`` 未实际剪枝时不产出重复的 ``prunedExpr.png`` /
   ``pruned_expr_tree``，曲线只画一条（``plot_data_curves(pruned=None)``）。
"""
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import sympy as sp

from drsr_420.analysis import prune_report as pr
from drsr_420.analysis.prune_stats import PruneStats
from drsr_420.analysis.sensitivity_prune import SensitivityPruner

x, y = sp.symbols("x y", real=True)
L12, L23 = sp.symbols("lambda12 lambda23", real=True)

#: 合成"通分"夹具：simplify 把它改写成一个分式（节点数 5 → 6），但两项都剪不掉。
COMMON_DENOM = 1 / (x + 1) + 1 / (y + 2)

#: 真实样本（experiments/MRFCompress-Cuboid/MRFCompress-Cuboid_20260918-211536 的最优
#: 公式）：0 项剪枝、simplify 改写了形式，且 sympy 的两套符号判定都判不出恒等。
REAL_FORM_ONLY = (50.4835 * L12**0.55027 + 38.6298 * L12**4.26619 / L23**5.78475
                  + 56.5146 * L23**0.567986
                  + 1.03519e-7 * L23**8.7605 * (L12 - 1.0)**2 + 47.4264)


def _make_experiment(root: pathlib.Path, func: str, params: list,
                     rows: list[str]) -> None:
    """搭一个最小实验目录：样本、config_snapshot、训练 CSV（表头固定 x1,x2,y）。"""
    (root / "samples").mkdir(parents=True, exist_ok=True)
    (root / "samples" / "top01_samples_1.json").write_text(
        json.dumps({"score": -0.5, "sample_order": 1,
                    "function": func, "params": params}), encoding="utf-8")
    (root / "config_snapshot.json").write_text(
        json.dumps({"data_csv": "data/tiny/train.csv"}), encoding="utf-8")
    data_dir = root / "data" / "tiny"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "\n".join(["x1,x2,y"] + rows) + "\n", encoding="utf-8")


#: 两项都剪不掉、但 simplify 会通分的公式（端到端夹具）
_FUNC_FORM_ONLY = ("Variables:\n"
                   "- Independents: x1, x2\n"
                   "- Dependent: y\n"
                   "def equation(x1, x2, params):\n"
                   "    return params[0]/(x1 + 1) + params[1]/(x2 + 2)\n")
_ROWS_FORM_ONLY = [f"{v1},{6.0 - v1},{2.0 / (v1 + 1) + 3.0 / (6.0 - v1 + 2)}"
                   for v1 in (1.0, 2.0, 3.0, 4.0, 5.0)]


class PruneReturnContractTest(unittest.TestCase):
    """没真剪掉项 → 返回原式；真剪掉项 → 才做 simplify 并返回剪枝结果。"""

    def test_zero_pruned_returns_the_original_object(self):
        pruner = SensitivityPruner([x, y], threshold=0.01, sample_range=(1, 6), seed=42)
        published = pruner.prune(COMMON_DENOM)
        self.assertEqual(pruner.stats.nodes_pruned, 0)
        self.assertIs(published, COMMON_DENOM, "0 项剪枝时必须原样返回原表达式")
        self.assertEqual(pruner.stats.simplify_applied, False)
        self.assertEqual(pruner.stats.ops_before, pruner.stats.ops_after)

    def test_zero_pruned_keeps_simplify_result_only_as_diagnostic(self):
        pruner = SensitivityPruner([x, y], threshold=0.01, sample_range=(1, 6), seed=42)
        pruner.prune(COMMON_DENOM)
        simplified = pruner.stats.simplified_expr
        self.assertIsNotNone(simplified)
        self.assertNotEqual(simplified, COMMON_DENOM)      # simplify 确实改了形式
        self.assertGreater(sp.count_ops(simplified), sp.count_ops(COMMON_DENOM))
        # 诊断值必须与原式数值等价（否则"只是换写法"的说法就不成立）
        rel = pr.max_relative_difference(COMMON_DENOM, simplified, ["x", "y"], (1, 6))
        self.assertLess(rel, pr.FORM_ONLY_RTOL)

    def test_real_pruning_still_simplifies(self):
        expr = x**2 + y**2 + sp.Rational(1, 1000) * x * y
        pruner = SensitivityPruner([x, y], threshold=0.02, sample_range=(1, 6), seed=42)
        published = pruner.prune(expr)
        self.assertGreater(pruner.stats.nodes_pruned, 0)
        self.assertTrue(pruner.stats.simplify_applied)
        self.assertIsNone(pruner.stats.simplified_expr)
        self.assertNotIn("x*y", str(published))
        self.assertLessEqual(pruner.stats.ops_after, pruner.stats.ops_before)


class ClassifyPruningTest(unittest.TestCase):
    def _classify(self, expr, syms, rng=(1, 6), threshold=0.01):
        pruner = SensitivityPruner(syms, threshold=threshold, sample_range=rng, seed=42)
        published = pruner.prune(expr)
        verdict = pr.classify_pruning(expr, published, pruner.stats,
                                      [str(s) for s in syms], rng)
        return published, verdict

    def test_form_only_for_common_denominator(self):
        published, verdict = self._classify(COMMON_DENOM, [x, y])
        self.assertEqual(verdict["kind"], "form_only")
        self.assertFalse(verdict["actually_pruned"])
        self.assertTrue(verdict["used_original"])
        self.assertTrue(verdict["form_rewritten"])
        self.assertFalse(verdict["form_changed"])          # 发布的公式仍是原式
        self.assertTrue(verdict["numerically_equivalent"])
        self.assertEqual(verdict["ops_published"], verdict["ops_before"])
        self.assertGreater(verdict["simplify_ops"], verdict["ops_before"])
        self.assertIn("未实际剪枝", verdict["summary"])
        self.assertIn("沿用剪枝前的形式", verdict["summary"])
        self.assertEqual(published, COMMON_DENOM)

    def test_none_when_nothing_changes_at_all(self):
        _published, verdict = self._classify(2 * x * y, [x, y])
        self.assertEqual(verdict["kind"], "none")
        self.assertFalse(verdict["form_rewritten"])
        self.assertIn("与剪枝前完全相同", verdict["summary"])

    def test_pruned_kind_when_a_term_is_really_removed(self):
        published, verdict = self._classify(
            x**2 + y**2 + sp.Rational(1, 1000) * x * y, [x, y], threshold=0.02)
        self.assertEqual(verdict["kind"], "pruned")
        self.assertTrue(verdict["actually_pruned"])
        self.assertFalse(verdict["used_original"])
        self.assertTrue(verdict["form_changed"])
        self.assertLess(verdict["ops_published"], verdict["ops_before"])
        self.assertIn("本次实际剪枝", verdict["summary"])
        self.assertNotEqual(published, x**2 + y**2 + sp.Rational(1, 1000) * x * y)

    def test_numeric_comparison_is_the_authority_for_float_exponents(self):
        """真实样本回归：sympy 的符号判定给假阴性，数值比较才是权威判据。"""
        pruner = SensitivityPruner([L12, L23], threshold=0.1,
                                   sample_range=(1, 14), seed=42)
        published = pruner.prune(REAL_FORM_ONLY)
        self.assertEqual(pruner.stats.nodes_pruned, 0)
        self.assertIs(published, REAL_FORM_ONLY)

        simplified = pruner.stats.simplified_expr
        self.assertNotEqual(simplified, REAL_FORM_ONLY)
        # 记录实测事实（不作为断言，避免 sympy 未来变聪明导致脆性失败）：
        #   sp.simplify(REAL_FORM_ONLY - simplified) == 0  →  False
        #   simplified.equals(REAL_FORM_ONLY)              →  False
        # 而数值比较给出"完全等价"，所以判定必须走数值这条路。
        rel = pr.max_relative_difference(REAL_FORM_ONLY, simplified,
                                        ["lambda12", "lambda23"], (1, 14))
        self.assertIsNotNone(rel)
        self.assertLess(rel, pr.FORM_ONLY_RTOL)

        verdict = pr.classify_pruning(REAL_FORM_ONLY, published, pruner.stats,
                                      ["lambda12", "lambda23"], (1, 14))
        self.assertEqual(verdict["kind"], "form_only")
        self.assertTrue(verdict["numerically_equivalent"])

    def test_unverified_branch_when_simplify_would_change_the_math(self):
        """simplify 若真改了数值：仍回退原式，但判定为"无法确认等价"并要求人工看。"""
        stats = PruneStats(nodes_visited=1, nodes_pruned=0, ops_before=sp.count_ops(x),
                           simplified_expr=x + 1)     # 伪造一个"改了数学"的 simplify 结果
        verdict = pr.classify_pruning(x, x, stats, ["x"], (1, 6))
        self.assertEqual(verdict["kind"], "form_only_unverified")
        self.assertFalse(verdict["numerically_equivalent"])
        self.assertIn("为稳妥起见仍沿用剪枝前的形式", verdict["summary"])

    def test_summary_distinguishes_actual_pruning(self):
        not_pruned = PruneStats(nodes_visited=13, nodes_pruned=0, ops_before=17,
                                simplified_expr=x**2 + x * y + y**2)
        self.assertIn("实际剪枝   : 否", not_pruned.summary())
        self.assertIn("沿用原式", not_pruned.summary())
        pruned = PruneStats(nodes_visited=4, nodes_pruned=1, ops_before=6, ops_after=3)
        self.assertIn("实际剪枝   : 是", pruned.summary())
        self.assertTrue(pruned.actually_pruned)
        self.assertFalse(not_pruned.actually_pruned)


class DegeneratePruningRejectedTest(unittest.TestCase):
    """剪枝结果不含任何自变量（公式塌缩成常数）→ 判为剪枝失败、回退原式。

    真实事故：MRFCompress-Cuboid_20260919-143926 的最优样本被剪成常数 193.054，
    NMSE 从 8.7e-07 退到 6.81（比"预测样本均值"的 1.0 还差 6.8 倍）。
    """

    def _stats(self, pruned: int, visited: int) -> PruneStats:
        st = PruneStats()
        st.nodes_visited = visited
        st.nodes_pruned = pruned
        st.ops_before = sp.count_ops(COMMON_DENOM)
        return st

    def test_constant_result_is_rejected(self):
        verdict = pr.classify_pruning(COMMON_DENOM, sp.Integer(3),
                                      self._stats(2, 2), ["x", "y"], (1, 6))
        self.assertEqual(verdict["kind"], "degenerate")
        self.assertFalse(verdict["actually_pruned"])
        self.assertTrue(verdict["used_original"])
        self.assertFalse(verdict["form_changed"])
        self.assertEqual(verdict["ops_published"], verdict["ops_before"])
        self.assertIn("退化剪枝", verdict["summary"])
        self.assertIn("沿用剪枝前", verdict["summary"])

    def test_result_keeping_one_independent_is_fine(self):
        verdict = pr.classify_pruning(COMMON_DENOM, 1 / (x + 1), self._stats(1, 2),
                                      ["x", "y"], (1, 6))
        self.assertEqual(verdict["kind"], "pruned")
        self.assertTrue(verdict["actually_pruned"])

    def test_rejection_is_wired_through_prune_and_visualize(self):
        """端到端：退化剪枝被拒后，对外发布的仍是原式（含自变量）。

        夹具：第二项在剪枝采样区间 (1,6) 上恒为 0（条件 x1 > 100 不成立），于是它被
        判为"零敏感"而剪掉，剪枝结果只剩常数——正是要被拒的那种退化。
        """
        from drsr_420.analysis.find_best_eq import prune_and_visualize

        func = ("Variables:\n"
                "- Independents: x1, x2\n"
                "- Dependent: y\n"
                "def equation(x1, x2, params):\n"
                "    return params[0] + params[1]*where(x1 > 100, x1, 0)\n")
        params = [3.0, 1.0]
        rows = [f"{v1},{6.0 - v1},3.0" for v1 in (1.0, 2.0, 3.0, 4.0, 5.0)]
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_experiment(root, func, params, rows)
            with mock.patch("builtins.print"):
                summary = prune_and_visualize(str(root), func, params,
                                              threshold=0.1, sample_range=(1, 6))
        self.assertIsNotNone(summary)
        self.assertEqual(summary["verdict"]["kind"], "degenerate")   # 退化剪枝被识别
        self.assertTrue(summary["used_original"])
        published = sp.sympify(summary["pruned_expr"])
        self.assertTrue({str(s) for s in published.free_symbols} & {"x1", "x2"},
                        "对外发布的公式必须仍含自变量（不得退化为常数）")


class ExplainBlockTest(unittest.TestCase):
    """解释提示词必须点明判定，并把 simplify 的改写标成"未采用"。"""

    def test_block_states_verdict_and_marks_simplify_as_unused(self):
        from drsr_420.analysis.explain import _format_pruning_block

        pruner = SensitivityPruner([x, y], threshold=0.01, sample_range=(1, 6), seed=42)
        published = pruner.prune(COMMON_DENOM)
        verdict = pr.classify_pruning(COMMON_DENOM, published, pruner.stats,
                                      ["x", "y"], (1, 6))
        text = _format_pruning_block({
            "dependent": "y",
            "sym_names": ["x", "y"],
            "threshold": 0.01,
            "sample_range": (1, 6),
            "substituted_expr": sp.sstr(COMMON_DENOM),
            "pruned_expr": sp.sstr(published),
            "simplify_expr": sp.sstr(pruner.stats.simplified_expr),
            "nodes_visited": pruner.stats.nodes_visited,
            "nodes_pruned": pruner.stats.nodes_pruned,
            "prune_rate": pruner.stats.prune_rate,
            "removed": [],
            "verdict": verdict,
            "fit": None,
        })

        self.assertIn("剪枝判定：", text)
        self.assertIn("未实际剪枝", text)
        self.assertIn("未采用", text)                       # simplify 形式被明确标注为未采用
        self.assertIn("与剪枝前完全相同（没有移除任何项）", text)
        self.assertIn("被移除的项：无", text)

    def test_required_structure_forbids_calling_a_rewrite_pruning(self):
        from drsr_420.analysis.explain import _REQUIRED_STRUCTURE

        self.assertIn("未实际剪枝", _REQUIRED_STRUCTURE)
        self.assertIn("不是**剪枝结果", _REQUIRED_STRUCTURE)
        self.assertIn("退化剪枝被拒", _REQUIRED_STRUCTURE)

    def test_required_structure_confronts_priors_with_measured_baselines(self):
        """解释必须把物理先验与代码实测的骨架基线对质，并交代可辨识性限制。"""
        from drsr_420.analysis.explain import _REQUIRED_STRUCTURE

        self.assertIn("候选骨架基线", _REQUIRED_STRUCTURE)
        self.assertIn("不得把先验写成已被数据证实的事实", _REQUIRED_STRUCTURE)
        self.assertIn("不可单独辨识", _REQUIRED_STRUCTURE)

    def test_facts_block_lists_baselines_and_identifiability(self):
        from drsr_420.analysis.explain import _format_facts_block

        text = _format_facts_block({
            "dependent": "sigma",
            "extremes": {"max": {"value": 352.1991,
                                 "at": {"lambda12": 1.0, "lambda23": 14.1245}}},
            "correlations": [{"a": "lambda23", "b": "sigma", "pearson": 0.8336,
                              "spearman": 0.8571, "log_pearson": 0.9651},
                             {"a": "lambda12", "b": "lambda23", "pearson": -0.376,
                              "spearman": -0.4192, "log_pearson": 0.0924}],
            "skeletons": [
                {"expression": "a*(lambda12*lambda23)^b + c", "nmse": 0.1698, "r2": 0.8302},
                {"expression": "a*lambda12^b*lambda23^c + d", "nmse": 0.0307, "r2": 0.9693},
            ],
            "identifiability": [{"a": "lambda12", "b": "lambda23",
                                 "message": "NOT separately identifiable"}],
            "monotonicity": [{"feature": "lambda23", "monotone": False, "reversals": 1,
                              "first_reversal": {"from": {"lambda23": 3.9174, "dependent": 306.577},
                                                 "to": {"lambda23": 4.8446, "dependent": 296.651}}}],
        })

        self.assertIn("352.1991", text)
        self.assertIn("lambda23 vs sigma", text)
        self.assertIn("NMSE=0.0307", text)
        self.assertIn("NOT separately identifiable", text)
        self.assertIn("必须显式报告冲突", text)
        # 单调性必须由代码给出结论，解释不得写成单调/饱和趋势
        self.assertIn("**不单调**", text)
        self.assertIn("306.577->296.651", text)
        # 只列与因变量相关的配对，不把自变量两两相关也当成解释依据
        self.assertNotIn("lambda12 vs lambda23：", text)

    def test_facts_block_degrades_when_file_missing(self):
        from drsr_420.analysis.explain import _format_facts_block

        text = _format_facts_block(None)
        self.assertIn("没有 data_facts.json", text)
        self.assertIn("不得声称某个先验或骨架形状已被数据支持", text)

    def test_block_explains_rejected_degenerate_pruning(self):
        """退化剪枝被拒时，提示词必须说清"最终还是剪枝前的公式"，不能写成已简化。"""
        from drsr_420.analysis.explain import _format_pruning_block

        stats = PruneStats(nodes_visited=2, nodes_pruned=2,
                           ops_before=sp.count_ops(COMMON_DENOM))
        verdict = pr.classify_pruning(COMMON_DENOM, sp.Integer(3), stats, ["x", "y"], (1, 6))
        text = _format_pruning_block({
            "dependent": "y",
            "sym_names": ["x", "y"],
            "threshold": 0.1,
            "sample_range": (1, 6),
            "substituted_expr": sp.sstr(COMMON_DENOM),
            "pruned_expr": sp.sstr(COMMON_DENOM),      # 回退后发布的仍是原式
            "nodes_visited": stats.nodes_visited,
            "nodes_pruned": stats.nodes_pruned,
            "prune_rate": stats.prune_rate,
            "removed": [],
            "verdict": verdict,
            "fit": None,
        })
        self.assertIn("退化剪枝", text)
        self.assertIn("沿用剪枝前的公式", text)
        self.assertNotIn("与剪枝前完全相同（没有移除任何项）", text)


class ArtifactSuppressionTest(unittest.TestCase):
    """未实际剪枝 → 不产出重复的"剪枝后"图件，曲线只画一条。"""

    def test_prune_and_visualize_keeps_original_and_skips_duplicate_artifacts(self):
        from drsr_420.analysis.find_best_eq import prune_and_visualize

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_experiment(root, _FUNC_FORM_ONLY, [2.0, 3.0], _ROWS_FORM_ONLY)
            with mock.patch("builtins.print"):
                summary = prune_and_visualize(str(root), _FUNC_FORM_ONLY, [2.0, 3.0],
                                              threshold=0.1, sample_range=(1, 6))

            self.assertTrue(summary["used_original"])
            self.assertEqual(summary["verdict"]["kind"], "form_only")
            self.assertEqual(summary["pruned_expr"], summary["substituted_expr"])
            self.assertEqual(summary["nodes_pruned"], 0)
            # 诊断字段：simplify 会给出什么形式（未采用），供 explain.md 说明"只是通分"
            self.assertIsNotNone(summary["simplify_expr"])
            self.assertNotEqual(summary["simplify_expr"], summary["substituted_expr"])
            # 不再产出与 expr.png 重复的"剪枝后"图件
            self.assertFalse((root / "prunedExpr.png").is_file())
            self.assertFalse((root / "pruned_expr_tree").is_file())
            self.assertFalse((root / "pruned_expr_tree.pdf").is_file())
            # 曲线图仍然产出（本次单条曲线，图注注明未剪枝）
            self.assertTrue((root / "expr_curve_x1.png").is_file())
            self.assertTrue((root / "expr_curve_x2.png").is_file())

    def test_plot_expr_curves_passes_none_when_nothing_pruned(self):
        from drsr_420.analysis import expr_curves as ec

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_experiment(root, _FUNC_FORM_ONLY, [2.0, 3.0], _ROWS_FORM_ONLY)
            captured = {}

            def _spy(*args, **kwargs):
                captured["args"] = args
                return []

            with mock.patch.object(ec, "plot_data_curves", _spy), \
                 mock.patch("builtins.print"):
                ec.plot_expr_curves(str(root), threshold=0.1, sample_range=(1, 6))

        self.assertIn("args", captured)
        self.assertIsNone(captured["args"][4],
                          "未实际剪枝时必须以 pruned=None 调用绘图（只画一条曲线）")

    def test_tree_rendering_skips_the_missing_side(self):
        from drsr_420.analysis import expr_viz as viz

        rendered: list = []

        class _FakeSource:
            def __init__(self, dot):
                self.dot = dot

            def render(self, base, view=False):     # noqa: ARG002 - 签名对齐 graphviz
                rendered.append(base)

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(viz, "Source", _FakeSource), \
             mock.patch("builtins.print"):
            viz.render_expr_trees(tmp, x + 1, None)

        self.assertEqual(rendered, [f"{tmp}/original_expr_tree"])

    def test_tree_rendering_keeps_both_sides_when_pruned(self):
        from drsr_420.analysis import expr_viz as viz

        rendered: list = []

        class _FakeSource:
            def __init__(self, dot):
                self.dot = dot

            def render(self, base, view=False):     # noqa: ARG002
                rendered.append(base)

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(viz, "Source", _FakeSource), \
             mock.patch("builtins.print"):
            viz.render_expr_trees(tmp, x + 1, x)

        self.assertEqual(rendered, [f"{tmp}/original_expr_tree",
                                    f"{tmp}/pruned_expr_tree"])


if __name__ == "__main__":
    unittest.main()
