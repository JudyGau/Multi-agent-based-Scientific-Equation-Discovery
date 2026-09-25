"""动态范围体检（角点钉扎/下溢尖峰类病理解治理）的单元与集成测试。

背景（实测 MRFCompress-Cuboid_20260921-161549 及 45 次历史实验普查）：
LLM 骨架爱用"只在个别数据点非零"的局部化器件（大负指数幂、窄高斯、下溢尖峰）
去消化个别难拟合的数据点——训练点 MSE 看不见这种行为。体检在训练数据包围盒
网格（含角点对数壳层）上评估拟合后的方程，**两条判据**超阈值即按超出量罚分
（评分层治理，不侵入拟合，残差分析回路仍看到真实残差）：

1. 输出跨度（放大/溢出型器件）；
2. 局部斜率（门控型器件：只在角点非零、域内其余位置下溢到 0，跨度判据看不见）。
   判据 2 来自实测 MRFCompress-Cuboid_20260925-112514 的发布最优解——它靠
   λ12^(−40.153) 门控（自身动态范围 1.16e28）把 λ12=1 的两个点单独分支出来，
   输出跨度只有 2.15 倍、判据 1 判"正常"，而它是该次 top-10 里 8 个门控解之一。
"""
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from drsr_420.evaluation import problems
from drsr_420.evaluation.problems import (
    RANGE_PENALTY_CAP,
    RANGE_SLOPE_LIMIT,
    RANGE_SPAN_RATIO_LIMIT,
    dynamic_range_check,
    evaluate,
    local_slope_check,
)

# MRFCompress-Cuboid 训练数据的自变量点与响应（λ12/λ23 反相关的一维山脊 + 角点）
_XY = [(1.0, 1.0), (1.0, 14.1245), (2.0, 8.9331), (2.5, 7.4185),
       (3.0, 6.3166), (3.5, 5.4875), (4.0, 4.8446), (5.0, 3.9174)]
_Y = [193.054, 352.199, 339.578, 317.233, 305.193, 299.014, 296.651, 306.577]
_DATA = {"inputs": [list(p) for p in _XY], "outputs": list(_Y)}

# 20260921-161549 最优样本的有效函数：平滑部分 + 角点钉扎下溢尖峰（p0 只在 (1,1) 非零）
def _spike_equation(x1, x2, params):
    return (params[0] / (x1 * x2) ** 126.08
            + 23.1034 * x1 + 70.1043 * x2
            + 1029.5085 * np.log(x1) + 841.7849 * np.log(x2)
            - 2890.0514)

# 同一平滑部分、无尖峰（在角点邻域外与上式几乎重合）
def _smooth_equation(x1, x2, params):
    return (23.1034 * x1 + 70.1043 * x2
            + 1029.5085 * np.log(x1) + 841.7849 * np.log(x2)
            - 2890.0514 + params[0])

#: 20260925-112514 的发布最优解（MSE 0.12679）：门控型角点钉扎，跨度判据看不见。
_GATE_PARAMS = [119.05442184012739, 35.151497878397585, 0.8105997307897103,
                49.688883077073406, -40.15300581395821, 3.243229057962613,
                -14.083735853522994]


def _gate_equation(x1, x2, params):
    """σ = p0 + p1·λ23^p2 + p3·λ12^p4 + p5·λ12² + p6·(λ12^p4)(λ23^p2)。

    p4 = −40.153 → λ12^(−40.153) 是"近乎 Heaviside 的门控"：λ12=1 时 1、
    1.017 时 0.5、2 时 8.2e-13、5 时 8.6e-29，把 λ12=1 的两个数据点单独分支。
    """
    gate = x1 ** params[4]
    return (params[0] + params[1] * x2 ** params[2] + params[3] * gate
            + params[5] * x1 ** 2 + params[6] * gate * x2 ** params[2])


class RangeGridTest(unittest.TestCase):
    def test_grid_contains_regular_and_corner_shell_points(self):
        pts = problems.range_check_points(np.array(_XY))
        # 2 维：24^2 均匀网格 + 4 角点 × 16 壳层
        self.assertEqual(pts.shape, (24 * 24 + 4 * 16, 2))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        self.assertTrue((lo >= np.array(_XY).min(axis=0) - 1e-12).all())
        self.assertTrue((hi <= np.array(_XY).max(axis=0) + 1e-12).all())
        # 角点壳层必须贴到 (1+δ, 1+δ)：δ 最小 1e-6×各维 range（角点本身由
        # 均匀网格端点覆盖，壳层是它的紧邻）
        d = np.linalg.norm(pts - np.array([1.0, 1.0]), axis=1)
        near = d[d > 0]
        self.assertLess(near.min(), 1e-4)

    def test_high_dimension_does_not_explode(self):
        rng = np.random.default_rng(0)
        pts5 = rng.uniform(0, 1, size=(12, 5))
        grid = problems.range_check_points(pts5)
        # d=5：每维 int(round(576^(1/5)))=4 → 4^5=1024 网格 + 2^5×16=512 壳层
        self.assertEqual(grid.shape, (4 ** 5 + 2 ** 5 * 16, 5))
        self.assertTrue((grid >= pts5.min(axis=0) - 1e-12).all())
        self.assertTrue((grid <= pts5.max(axis=0) + 1e-12).all())


class DynamicRangeCheckTest(unittest.TestCase):
    def test_benign_affine_has_zero_penalty(self):
        # 数据覆盖包围盒四角（数据跨度 = 网格跨度）→ 仿射模型 span_ratio = 1
        X = np.array([(1.0, 1.0), (1.0, 14.1245), (5.0, 1.0), (5.0, 14.1245)])
        y = 100.0 + 10.0 * X[:, 0] + 5.0 * X[:, 1]
        info = dynamic_range_check(X, y, lambda a, b: 100.0 + 10.0 * a + 5.0 * b)
        self.assertLessEqual(info["span_ratio"], 1.0 + 1e-9)
        self.assertEqual(info["penalty"], 0.0)
        # 两条判据都不许误报：仿射的归一化差商 ~0.1（斜率 10 / 跨度 105.6）
        self.assertLess(info["slope_max"], 1.0)

    def test_corner_spike_is_caught(self):
        X = np.array(_XY)
        y = np.array(_Y)
        info = dynamic_range_check(
            X, y, lambda a, b: _spike_equation(a, b, [2989.9]))
        self.assertGreater(info["span_ratio"], RANGE_SPAN_RATIO_LIMIT)
        self.assertGreater(info["penalty"], 0.0)
        self.assertLess(info["grid_min"], 0.0, "角点邻域深谷（平滑部分外推到 -2796）必须被撞见")

    def test_all_nonfinite_gets_capped_penalty(self):
        X = np.array(_XY)
        y = np.array(_Y)
        info = dynamic_range_check(X, y, lambda a, b: np.full_like(a, np.nan))
        self.assertEqual(info["penalty"], RANGE_PENALTY_CAP)
        self.assertEqual(info["span_ratio"], float("inf"))
        self.assertEqual(info["slope_max"], float("inf"))

    def test_realistic_smooth_only_model_also_flagged(self):
        """平滑部分自身在角点邻域外推到 -2796（共线大系数），即使无尖峰也是病理。"""
        X = np.array(_XY)
        y = np.array(_Y)
        info = dynamic_range_check(X, y, lambda a, b: _smooth_equation(a, b, [0.0]))
        self.assertGreater(info["penalty"], 0.0)

    def test_constant_equation_is_not_a_failure(self):
        """常数方程是合法解（输出跨度 0），不能与"求值失败"混为一谈。"""
        info = dynamic_range_check(np.array(_XY), np.array(_Y),
                                   lambda a, b: np.full_like(a, 250.0))
        self.assertEqual(info["span_ratio"], 0.0)
        self.assertEqual(info["slope_max"], 0.0)
        self.assertEqual(info["penalty"], 0.0)


class LocalSlopeCheckTest(unittest.TestCase):
    """判据二（局部斜率）：专抓只在角点非零、域内其余位置下溢到 0 的门控器件。"""

    def test_gate_has_huge_slope_but_normal_span(self):
        X, y = np.array(_XY), np.array(_Y)
        info = dynamic_range_check(X, y, lambda a, b: _gate_equation(a, b, _GATE_PARAMS))
        # 跨度判据看不见它：只在角点非零的门控不会放大包围盒上的输出跨度
        self.assertLess(info["span_ratio"], RANGE_SPAN_RATIO_LIMIT,
                        "门控不放大输出跨度——这正是判据一漏掉它的原因")
        self.assertEqual(info["span_penalty"], 0.0)
        # 判据二必须认出它（实测 16.5，健康参数化的上沿是 0.68）
        self.assertGreater(info["slope_max"], RANGE_SLOPE_LIMIT)
        self.assertGreater(info["slope_penalty"], 0.0)
        self.assertEqual(info["penalty"], info["slope_penalty"])

    def test_healthy_fits_stay_far_below_the_limit(self):
        """健康参数化（幂律/仿射/二次/饱和）不得被误报——留出量级余量。"""
        healthy = [
            lambda a, b, p: p[0] * a ** p[1] * b ** p[2] + p[3],
            lambda a, b, p: p[0] * (a * b) + p[1] * (a + b) + p[2],
            lambda a, b, p: p[0] * b / (p[1] + b) + p[2],
            lambda a, b, p: p[0] * (1.0 - np.exp(-p[1] * b)) + p[2],
        ]
        from drsr_420.evaluation.problems import evaluate as _fit
        for eq in healthy:
            with np.errstate(all='ignore'), mock.patch("builtins.print"):
                score, _, params = _fit(_DATA, eq, range_check=False, seed=0)
            if score is None:
                continue
            info = dynamic_range_check(
                np.array(_XY), np.array(_Y), lambda a, b, _e=eq, _p=params: _e(a, b, _p))
            with self.subTest(eq=eq):
                self.assertLess(info["slope_max"], RANGE_SLOPE_LIMIT)

    def test_slope_check_direct_call_and_degenerate_axis(self):
        """直接调用（不经过 dynamic_range_check）与退化维（某列无变化）都不崩。"""
        X = np.array([(1.0, 1.0), (1.0, 14.1245), (5.0, 1.0), (5.0, 14.1245)])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        out = local_slope_check(X, y, lambda a, b: 100.0 + 10.0 * a + 5.0 * b)
        self.assertIn("slope_max", out)
        single = np.array([(2.0, 1.0), (2.0, 9.0)])       # 第 0 列无变化
        out2 = local_slope_check(single, np.array([1.0, 2.0]),
                                 lambda a, b: 3.0 * b)
        self.assertTrue(np.isfinite(out2["slope_max"]))


class EvaluatePenaltyIntegrationTest(unittest.TestCase):
    def test_pathological_model_is_penalized_and_clean_is_not(self):
        with mock.patch("builtins.print"):
            score_patho, res_patho, params_patho = evaluate(_DATA, _spike_equation)
            score_patho_off, _, _ = evaluate(_DATA, _spike_equation, range_check=False)
        # 拟合本身不受影响：钉扎系数 p0 ≈ 2989.9，把角点残差清零
        self.assertAlmostEqual(float(params_patho[0]), 2989.9, delta=1.0)
        # 罚分只作用于评分：病理样本被压到远低于其训练 MSE 的水平
        self.assertLess(score_patho, score_patho_off - 10.0)
        self.assertGreater(score_patho_off, -1.0, "训练 MSE 本身应很小（尖峰把 8 点都拟合了）")
        # 残差列保持真实残差（罚分不进残差矩阵）：病理样本的残差仍是训练点残差
        self.assertIsNotNone(res_patho)
        self.assertLess(float(np.abs(res_patho[:, -1]).max()), 10.0)

    def test_clean_model_score_unchanged_by_check(self):
        # 数据覆盖包围盒四角的良性仿射：体检开关不改变评分（罚分为 0）
        X = [(1.0, 1.0), (1.0, 14.1245), (5.0, 1.0), (5.0, 14.1245)]
        data = {"inputs": [list(p) for p in X],
                "outputs": [100.0 + 10.0 * a + 5.0 * b for a, b in X]}

        def affine(a, b, params):
            return params[0] + params[1] * a + params[2] * b

        with mock.patch("builtins.print"):
            score_on, _, _ = evaluate(data, affine, seed=1)
            score_off, _, _ = evaluate(data, affine, seed=1, range_check=False)
        self.assertAlmostEqual(score_on, score_off, places=9)

    def test_gate_solution_loses_to_its_own_mse(self):
        """门控解在评分层被扣：score = −(MSE + 斜率罚分)，不再是 −MSE。

        这是 20260925-112514 的真实情形——该解 MSE 0.12679 曾是全局最优、零罚分；
        加上判据二后它必须被扣掉约 14.5 分，从而让位给无门控的结构。
        """
        with mock.patch("builtins.print"):
            score_on, _, _ = evaluate(_DATA, _gate_equation, x0=np.array(_GATE_PARAMS))
            score_off, _, _ = evaluate(_DATA, _gate_equation,
                                       x0=np.array(_GATE_PARAMS), range_check=False)
        self.assertLess(score_off, -0.1)                  # 训练 MSE 本身很小（≈0.127）
        self.assertLess(score_on, score_off - RANGE_SLOPE_LIMIT - 1.0)


class ExplainRangeSectionTest(unittest.TestCase):
    PRUNING = {"dependent": "sigma", "sym_names": ["lambda12", "lambda23"],
               "threshold": 0.1, "sample_range": (1, 14), "nodes_visited": 8,
               "nodes_pruned": 0, "prune_rate": 0.0, "removed": [],
               "substituted_expr": "23.1*lambda12", "pruned_expr": "23.1*lambda12",
               "verdict": {"kind": "none", "summary": "未实际剪枝"},
               "fit": {}, "range_check": {
                   "span_ratio": 38.3, "grid_min": -2746.0, "grid_max": 3353.3,
                   "span_penalty": 23.3, "slope_max": 1.0, "slope_limit": 2.0,
                   "slope_penalty": 0.0, "penalty": 23.3,
                   "limit": RANGE_SPAN_RATIO_LIMIT, "n_points": 640}}

    #: 只有判据二命中（判据一是 0）——这正是 20260925-112514 发布解的形态。
    GATE_ONLY = {"span_ratio": 2.15, "grid_min": 193.8, "grid_max": 501.0,
                 "span_penalty": 0.0, "slope_max": 16.51, "slope_limit": 2.0,
                 "slope_penalty": 14.51, "penalty": 14.51,
                 "limit": RANGE_SPAN_RATIO_LIMIT, "n_points": 640}

    def test_pruning_block_reports_pathology(self):
        from drsr_420.analysis.explain import _format_pruning_block
        text = _format_pruning_block(self.PRUNING)
        self.assertIn("动态范围体检", text)
        self.assertIn("病理性", text)
        self.assertIn("角点钉扎", text)
        self.assertIn("输出跨度", text)

    def test_pruning_block_reports_clean(self):
        """通过时只声明"未检出"并列出所检两项，不得写成"无角点钉扎类病理"。"""
        from drsr_420.analysis.explain import _format_pruning_block
        pruning = dict(self.PRUNING)
        pruning["range_check"] = {"span_ratio": 1.2, "grid_min": 193.0,
                                  "grid_max": 352.5, "span_penalty": 0.0,
                                  "slope_max": 0.5, "slope_limit": 2.0,
                                  "slope_penalty": 0.0, "penalty": 0.0,
                                  "limit": RANGE_SPAN_RATIO_LIMIT, "n_points": 640}
        text = _format_pruning_block(pruning)
        self.assertIn("未检出", text)
        self.assertIn("局部斜率", text)
        self.assertNotIn("病理性", text)
        self.assertNotIn("无角点钉扎", text)

    def test_gate_only_hit_is_reported_as_pathology(self):
        """判据一正常、判据二命中：判定必须是病理性，且点名的判据是局部斜率。"""
        from drsr_420.analysis.explain import _format_pruning_block, render_range_section
        pruning = dict(self.PRUNING)
        pruning["range_check"] = self.GATE_ONLY
        text = _format_pruning_block(pruning)
        self.assertIn("病理性", text)
        self.assertIn("局部斜率", text)
        section = render_range_section(self.GATE_ONLY)
        self.assertIn("病理性", section)
        self.assertIn("局部斜率", section)
        self.assertIn("未检出", render_range_section(dict(
            self.GATE_ONLY, slope_max=0.5, slope_penalty=0.0, penalty=0.0)))

    def test_assemble_replaces_llm_authored_range_section(self):
        from drsr_420.analysis.explain import _assemble_explain
        body = "正文\n\n## 动态范围体检\nLLM 编造的数字 12345\n\n## 其他\n内容"
        text = _assemble_explain(body, refs=[], range_check=self.PRUNING["range_check"])
        self.assertEqual(text.count("## 动态范围体检"), 1)
        self.assertNotIn("12345", text)
        self.assertIn("病理性", text)
        # 机器小节排在样本外验证之后、参考文献之前
        self.assertLess(text.index("## 样本外验证"), text.index("## 动态范围体检"))
        self.assertLess(text.index("## 动态范围体检"), text.index("## 参考文献"))


class PruneSummaryRangeCheckTest(unittest.TestCase):
    """收尾产物：prune_and_visualize 的剪枝摘要必须携带发布式的体检结果。"""

    FUNC = ("Variables:\n"
            "- Independents: lambda12, lambda23\n"
            "- Dependent: sigma\n"
            "def equation(lambda12, lambda23, params):\n"
            "    return params[0]/((lambda12*lambda23)**126.08)"
            " + 23.1034*lambda12 + 70.1043*lambda23"
            " + 1029.5085*np.log(lambda12) + 841.7849*np.log(lambda23)"
            " - 2890.0514\n")

    def test_summary_carries_range_check(self):
        from drsr_420.analysis.find_best_eq import prune_and_visualize
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "samples").mkdir(parents=True)
            (root / "samples" / "top01_samples_1.json").write_text(json.dumps(
                {"score": -0.25, "sample_order": 1, "function": self.FUNC,
                 "params": [2989.9]}), encoding="utf-8")
            (root / "config_snapshot.json").write_text(json.dumps(
                {"data_csv": "data/tiny/train.csv"}), encoding="utf-8")
            data_dir = root / "data" / "tiny"
            data_dir.mkdir(parents=True)
            rows = ["lambda12,lambda23,sigma"]
            for (a, b), y in zip(_XY, _Y):
                rows.append(f"{a},{b},{y}")
            (data_dir / "train.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            with mock.patch("builtins.print"):
                summary = prune_and_visualize(str(root), self.FUNC, [2989.9],
                                              threshold=0.1, sample_range=(1, 14),
                                              test_csv="none")
        rc = summary.get("range_check")
        self.assertIsInstance(rc, dict, "剪枝摘要必须携带体检结果")
        self.assertGreater(rc["penalty"], 0.0, "钉扎样本必须被判病理")
        self.assertGreater(rc["span_ratio"], RANGE_SPAN_RATIO_LIMIT)


if __name__ == "__main__":
    unittest.main()
