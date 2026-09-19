"""数据事实表（evaluation/data_facts.py）的行为测试。

覆盖：
- 逐列统计 / 相关结构（线性 + 秩 + 对数空间）：秩相关对单调非线性关系应为 1；
- 全局极值点：必须给出真实的最大值位置（实测模型曾把峰说在 lambda12=2，
  而真实最大在 lambda12=1）；
- 候选骨架基线：必须用评估器同口径拟合，且能把"乘积支配"这类先验反驳掉
  （真实 MRF 数据上分别幂律的 NMSE 明显小于乘积骨架）；
- 完备数据表的写入门槛（超过 MAX_TABLE_ROWS 不写全表，避免把抽样当全部）；
- 可辨识性告警：人为共线设计必须告警、真实 MRF 数据不得误报；
- render / load / extract_xy 的契约与降级行为。
"""
import json
import os
import tempfile
import unittest

import numpy as np

from drsr_420.evaluation import data_facts as df

#: 真实 MRF 压缩模式数据（8 行，lambda12/lambda23 -> sigma）。
_MRF_CSV = os.path.join("data", "MRFCompress-Cuboid", "train.csv")


def _load_mrf():
    """读真实数据，返回 (X, feature_names, y, dependent)。"""
    table = np.genfromtxt(_MRF_CSV, delimiter=",", names=True)
    names = list(table.dtype.names)
    X = np.column_stack([table[n] for n in names[:-1]])
    return X, names[:-1], np.asarray(table[names[-1]]), names[-1]


class _MRFFixtureTest(unittest.TestCase):
    """真实数据缺失时整体跳过，不阻塞无数据环境的测试。"""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(_MRF_CSV):
            raise unittest.SkipTest("缺少 MRF 训练数据夹具")


class RankAndCorrelationTest(unittest.TestCase):
    def test_spearman_is_one_for_monotone_nonlinear(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = x ** 3                       # 单调但非线性：秩相关=1，线性相关<1
        self.assertAlmostEqual(df._spearman(x, y), 1.0, places=9)
        self.assertLess(df._pearson(x, y), 1.0)

    def test_constant_column_gives_none(self):
        self.assertIsNone(df._pearson(np.ones(5), np.arange(5.0)))


class FactSheetTest(_MRFFixtureTest):
    def test_columns_and_row_count(self):
        X, names, y, dep = _load_mrf()
        facts = df.compute_facts(X, y, names, dep, with_skeletons=False)
        self.assertEqual(facts["n_rows"], 8)
        self.assertEqual(facts["features"], ["lambda12", "lambda23"])
        self.assertEqual(facts["dependent"], "sigma")
        self.assertAlmostEqual(facts["columns"]["sigma"]["max"], 352.1991, places=3)

    def test_global_maximum_location_is_the_real_one(self):
        """回归：模型曾断言峰在 lambda12=2，实测全局最大在 lambda12=1。"""
        X, names, y, dep = _load_mrf()
        facts = df.compute_facts(X, y, names, dep, with_skeletons=False)
        top = facts["extremes"]["max"]
        self.assertAlmostEqual(top["value"], 352.1991, places=3)
        self.assertAlmostEqual(top["at"]["lambda12"], 1.0, places=6)
        self.assertAlmostEqual(top["at"]["lambda23"], 14.1245, places=3)

    def test_dependent_correlation_matches_measured(self):
        X, names, y, dep = _load_mrf()
        facts = df.compute_facts(X, y, names, dep, with_skeletons=False)
        corr = {(c["a"], c["b"]): c for c in facts["correlations"]}
        # lambda23 才是本数据里的主驱动，lambda12 几乎没有独立贡献
        self.assertAlmostEqual(corr[("lambda23", "sigma")]["pearson"], 0.8336, places=3)
        self.assertAlmostEqual(corr[("lambda12", "sigma")]["pearson"], 0.1537, places=3)

    def test_complete_table_included_for_small_data(self):
        X, names, y, dep = _load_mrf()
        facts = df.compute_facts(X, y, names, dep, with_skeletons=False)
        self.assertTrue(facts["table_included"])
        self.assertEqual(len(facts["table_rows"]), 8)
        self.assertEqual(facts["table_columns"], ["lambda12", "lambda23", "sigma"])

    def test_table_omitted_when_too_many_rows(self):
        X, names, y, dep = _load_mrf()
        facts = df.compute_facts(X, y, names, dep, max_table_rows=5, with_skeletons=False)
        self.assertFalse(facts["table_included"])
        self.assertEqual(facts["table_rows"], [])


class SkeletonBaselineTest(_MRFFixtureTest):
    def test_separate_power_beats_product_skeleton(self):
        """核心回归：把"整体长细比支配"这一先验变成可被实测反驳的候选。

        实测中模型反复断言压缩模式应力由 lambda12*lambda23 支配，而真实数据上
        乘积骨架的 NMSE 远高于分别幂律。
        """
        X, names, y, dep = _load_mrf()
        rows = df.skeleton_baselines(X, y, names, dep)
        nmse = {r["expression"]: r["nmse"] for r in rows}
        separate = nmse["a*lambda12^b*lambda23^c + d"]
        product = nmse["a*(lambda12*lambda23)^b + c"]
        self.assertLess(separate, 0.05)
        self.assertGreater(product, 3 * separate)

    def test_rows_are_sorted_by_nmse_and_carry_r2(self):
        X, names, y, dep = _load_mrf()
        rows = df.skeleton_baselines(X, y, names, dep)
        values = [r["nmse"] for r in rows if r["nmse"] is not None]
        self.assertEqual(values, sorted(values))
        self.assertTrue(all("expression" in r and "r2" in r for r in rows))

    def test_two_feature_labels_are_used(self):
        X, names, y, dep = _load_mrf()
        labels = [r["expression"] for r in df.skeleton_baselines(X, y, names, dep)]
        self.assertTrue(any("lambda12/lambda23" in label for label in labels))
        self.assertTrue(any(label.startswith("a*lambda23^b") for label in labels))


class IdentifiabilityTest(unittest.TestCase):
    def test_collinear_design_is_flagged(self):
        x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x2 = 2.0 * x1
        facts = df.compute_facts(np.column_stack([x1, x2]), x1 * 3, ["a", "b"], "y",
                                 with_skeletons=False)
        self.assertEqual(len(facts["identifiability"]), 1)
        self.assertIn("NOT separately identifiable", facts["identifiability"][0]["message"])

    def test_independent_design_is_not_flagged(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(1.0, 5.0, size=(30, 2))
        facts = df.compute_facts(X, X[:, 0] + 2 * X[:, 1], ["a", "b"], "y",
                                 with_skeletons=False)
        self.assertEqual(facts["identifiability"], [])


class RenderFactsTest(_MRFFixtureTest):
    def test_render_contains_authority_rules_and_measured_numbers(self):
        X, names, y, dep = _load_mrf()
        text = df.render_facts(df.compute_facts(X, y, names, dep))
        self.assertIn("measured by code", text)
        self.assertIn("do NOT paraphrase", text)
        self.assertIn("352.1991", text)
        self.assertIn("NMSE=", text)
        self.assertIn("lower NMSE is better", text)

    def test_render_empty_facts_is_empty(self):
        self.assertEqual(df.render_facts({}), "")


class ExtractAndLoadTest(unittest.TestCase):
    def test_extract_xy_accepts_both_shapes(self):
        X = np.arange(6.0).reshape(3, 2)
        y = np.arange(3.0)
        wrapped = df.extract_xy({"data": {"inputs": X, "outputs": y}})
        flat = df.extract_xy({"inputs": X, "outputs": y})
        for got in (wrapped, flat):
            self.assertIsNotNone(got)
            np.testing.assert_allclose(got[0], X)
            np.testing.assert_allclose(got[1], y)

    def test_extract_xy_rejects_other_shapes(self):
        for bad in ([{"data": {}}], "path.csv", {}, {"inputs": [], "outputs": []}):
            self.assertIsNone(df.extract_xy(bad))

    def test_load_facts_degrades_to_empty_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(df.load_facts(tmp), {})        # 文件不存在
            with open(df.facts_path(tmp), "w", encoding="utf-8") as f:
                f.write("{ not json")
            self.assertEqual(df.load_facts(tmp), {})        # 损坏
            with open(df.facts_path(tmp), "w", encoding="utf-8") as f:
                json.dump({"n_rows": 8}, f)
            self.assertEqual(df.load_facts(tmp), {"n_rows": 8})


if __name__ == "__main__":
    unittest.main()