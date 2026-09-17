"""expr_curves：剪枝前后表达式曲线 + 数据点的核心契约。

三件事：
1. 曲线**沿训练数据路径求值**（按当前自变量排序、在完整数据坐标处代入模型）——
   数据自变量强耦合时（如 MRF 的 lambda12*lambda23 ≈ 常数），"其它变量固定在中
   位数"的切片不在数据流形上，曲线会与散点严重脱节（实测教训，见模块注释）；
2. 全精度求值：expr_substitution 已去掉 n(2) 有效数字舍入，曲线必须穿过精确
   拟合的数据点（NMSE 1e-29 的拟合画出来对不上 = 精度陷阱回归）；
3. 剪枝完成后自动触发：prune_and_visualize 末尾调用 plot_data_curves，
   失败只告警，不拖垮收尾流程。
"""
import json
import os
import pathlib
import tempfile
import unittest
import unittest.mock


def _make_experiment(root: pathlib.Path) -> None:
    """搭一个最小实验目录：样本、config_snapshot、训练 CSV。"""
    (root / "samples").mkdir(parents=True, exist_ok=True)
    func = (
        "Variables:\n"
        "- Independents: x1, x2\n"
        "- Dependent: y\n"
        "def equation(x1, x2, params):\n"
        "    return params[0]*x1 + params[1]*x2 + params[2]\n"
    )
    (root / "samples" / "top01_samples_1.json").write_text(
        json.dumps({"score": -0.5, "sample_order": 1,
                    "function": func, "params": [2.0, 3.0, 1.0]}),
        encoding="utf-8")
    (root / "config_snapshot.json").write_text(
        json.dumps({"data_csv": "data/tiny/train.csv"}), encoding="utf-8")
    # 训练数据：x1 与 x2 完全耦合（x2 = 6 - x1），模拟 MRF 那种一维流形
    data_dir = root / "data" / "tiny"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = ["x1,x2,y"]
    for x1 in (1.0, 2.0, 3.0, 4.0, 5.0):
        rows.append(f"{x1},{6.0 - x1},{2.0 * x1 + 3.0 * (6.0 - x1) + 1.0}")
    (data_dir / "train.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


class PlotDataCurvesTest(unittest.TestCase):
    def test_writes_one_png_per_independent(self):
        import matplotlib
        from drsr_420.analysis.expr_curves import plot_data_curves
        import sympy as sp

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_experiment(root)
            expr = 2.0 * sp.Symbol("x1") + 3.0 * sp.Symbol("x2") + 1.0
            written = plot_data_curves(str(root), "y", ["x1", "x2"], expr, expr)
            self.assertEqual(len(written), 2)
            for p in written:
                self.assertTrue(os.path.isfile(p))
                self.assertGreater(os.path.getsize(p), 0)
            self.assertTrue((root / "expr_curve_x1.png").is_file())
            self.assertTrue((root / "expr_curve_x2.png").is_file())
            del matplotlib

    def test_missing_snapshot_skips_quietly(self):
        from drsr_420.analysis.expr_curves import plot_data_curves
        import sympy as sp

        with tempfile.TemporaryDirectory() as tmp:
            written = plot_data_curves(tmp, "y", ["x1"], sp.Symbol("x1") + 1.0)
            self.assertEqual(written, [])


class PruneWiringTest(unittest.TestCase):
    def test_prune_and_visualize_also_writes_curves(self):
        """回归：剪枝完成后必须自动产出曲线图（此前只能手动补跑）。"""
        from drsr_420.analysis.find_best_eq import prune_and_visualize

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_experiment(root)
            func = (
                "Variables:\n"
                "- Independents: x1, x2\n"
                "- Dependent: y\n"
                "def equation(x1, x2, params):\n"
                "    return params[0]*x1 + params[1]*x2 + params[2]\n"
            )
            with unittest.mock.patch("builtins.print"):
                prune_and_visualize(str(root), func, [2.0, 3.0, 1.0],
                                    threshold=0.1, sample_range=(1, 6))
            self.assertTrue((root / "expr_curve_x1.png").is_file(),
                            "剪枝完成后应自动写出 expr_curve_x1.png")
            self.assertTrue((root / "expr_curve_x2.png").is_file())


if __name__ == "__main__":
    unittest.main()
