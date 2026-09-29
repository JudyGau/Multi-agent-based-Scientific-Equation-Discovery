"""expr_parse.expr_substitution 单元测试：参数代入、变量替换、中间变量消解与边界。"""
import json
import pathlib
import tempfile
import unittest

import sympy as sp

from drsr_420.analysis.expr_parse import (
    WhereArityError,
    audit_parse_failures,
    classify_sample,
    expr_substitution,
    find_matching_paren,
    fold_constant_comparisons,
    normalize_condition,
    rewrite_where_calls,
    split_top_level,
)

X1 = sp.Symbol("x1")
X2 = sp.Symbol("x2")


def _spec(body_lines, independents="x1", dependent="y", sig="equation(x1, params)"):
    return (
        "Variables:\n"
        f"- Independents: {independents}\n"
        f"- Dependent: {dependent}\n"
        f"def {sig}:\n"
        + "".join(f"    {ln}\n" for ln in body_lines)
    )


class _ExprTestCase(unittest.TestCase):
    """提供数值比较断言：绕过 SymPy Float/Integer 的符号不等价。"""

    def assert_expr_close(self, a, b, symbols=(X1,), places=6):
        for v in (0.3, 1.1, 2.7):
            subs = {s: v for s in symbols}
            self.assertAlmostEqual(float(a.subs(subs)), float(b.subs(subs)), places=places)


class ExprSubstitutionTest(_ExprTestCase):
    def test_linear_substitution(self):
        func = _spec(["return params[0]*x1 + params[1]"], sig="equation(x1, params)")
        expr = expr_substitution(func, [2.0, 3.0])
        self.assertIsNotNone(expr)
        self.assertEqual(sp.simplify(expr - (2 * X1 + 3)), 0)

    def test_multi_variable(self):
        func = (
            "Variables:\n"
            "- Independents: x1, x2\n"
            "- Dependent: y\n"
            "def equation(x1, x2, params):\n"
            "    return params[0]*x1 + params[1]*x2 + params[2]\n"
        )
        expr = expr_substitution(func, [2.0, 3.0, 1.0])
        self.assertEqual(sp.simplify(expr - (2 * X1 + 3 * X2 + 1)), 0)

    def test_params_keep_significant_digits_not_two_decimals(self):
        """回归：参数按**有效数字**舍入，而不是小数点后 2 位。

        实测 20260918-195057：params[3]=0.0074 被舍成 0.01，乘上量级 1e4 的项后
        MSE 从 2.9e-4 变成 186——收尾产物（剪枝/曲线/report.md）解释的将是另一个
        模型。定点 2 位小数对"小系数 × 巨量项"的骨架必须废弃。
        """
        func = _spec(["return params[0]*x1"])
        expr = expr_substitution(func, [2.3456])
        self.assertEqual(sp.simplify(expr - sp.Rational(23456, 10000) * X1), 0)

        func = _spec(["return params[0]*x1"])
        small = expr_substitution(func, [0.0074])
        self.assertEqual(sp.simplify(small - sp.Rational(74, 10000) * X1), 0,
                         "小系数必须保住自身量级（0.0074 不能变成 0.01）")

    def test_big_params_are_rounded_to_six_significant_digits(self):
        func = _spec(["return params[0]*x1"])
        expr = expr_substitution(func, [193.054296])
        self.assertAlmostEqual(float(expr.coeff(X1)), 193.054, places=6)

    def test_tuple_unpacking_of_params(self):
        """回归：p0, p1, ... = params[:n] 解包写法必须按位置代参
        （实测 194203：旧流程留下自由符号，表达式退化为裸 sigma）。"""
        func = _spec([
            "p0, p1, p2 = params[:3]",
            "return p0*x1 + p1*x1**2 + p2",
        ])
        expr = expr_substitution(func, [2.0, 3.0, 5.0])
        self.assertEqual(sp.simplify(expr - (2 * X1 + 3 * X1**2 + 5)), 0)

    def test_multiline_assignment_is_merged(self):
        """回归：sigma = ( 换行续写的多行赋值必须合并成单行再解析。"""
        func = _spec([
            "sigma = (",
            "    params[0]*x1",
            "    + params[1]",
            ")",
            "return sigma",
        ])
        expr = expr_substitution(func, [2.0, 3.0])
        self.assertEqual(sp.simplify(expr - (2 * X1 + 3)), 0)

    def test_194203_shape_tuple_unpack_plus_multiline_sigma(self):
        """194203 实际失败形态：解包 + 中间变量 + 多行赋值 + return sigma。"""
        func = _spec([
            "p0, p1, p2, p3 = params[:4]",
            "",
            "d12 = x1 - 1.0",
            "",
            "sigma = (",
            "    p0",
            "    + p1 * d12",
            "    + p2 * d12**2",
            "    + p3 / x1",
            ")",
            "return sigma",
        ], sig="equation(x1, params)")
        expr = expr_substitution(func, [193.05, 357.8, -78.38, 12.5])
        self.assertIsNotNone(expr)
        self.assertEqual(expr.free_symbols, {X1})   # 不再退化为裸符号
        v = float(expr.subs(X1, 2.0))
        self.assertAlmostEqual(v, 193.05 + 357.8 - 78.38 + 6.25, places=6)

    def test_tuple_unpacking_with_explicit_indices(self):
        """回归（20260921-134921）：``a, b, c = params[0], params[1], params[2]``
        逐项解包曾因 RHS 不是单一 params[...] 而整行漏掉，a..f 成为自由符号 →
        解析返回 None → 最优样本的剪枝与 held-out 验证被整体跳过。"""
        func = _spec([
            "a, b, c = params[0], params[1], params[2]",
            "return a*x1**b + c",
        ])
        expr = expr_substitution(func, [2.0, 3.0, 5.0])
        self.assertIsNotNone(expr)
        self.assertEqual(expr.free_symbols, {X1})
        self.assertAlmostEqual(float(expr.subs(X1, 2.0)), 2 * 8 + 5, places=9)

    def test_tuple_unpacking_itemwise_is_positional_not_index_bound(self):
        """逐项解包按位置对应（Python 语义）：即使项的书写顺序与下标交错也以
        位置为准——``a, b = params[1], params[0]`` 给 a=params[1]、b=params[0]。"""
        func = _spec([
            "a, b = params[1], params[0]",
            "return a*x1 + b",
        ])
        expr = expr_substitution(func, [10.0, 20.0])
        self.assertIsNotNone(expr)
        self.assertEqual(sp.simplify(expr - (20 * X1 + 10)), 0)

    def test_tuple_unpacking_with_backslash_continuation(self):
        """回归（20260921-161549 最优样本）：逐项解包写成反斜杠续行——

        ``p0, p1, p2 = params[0], params[1], \\n    params[2]`` 第一行以 ``\\``
        结尾，旧实现按单行匹配，``\\s*$`` 撞上行尾续行符必然失败，第二行又
        是裸参数行——p0..p7 全部沦为自由符号 → 解析返回 None → 剪枝与
        held-out 验证被整体跳过。续行必须先拼回单行再走逐项映射。
        """
        func = _spec([
            "p0, p1, p2 = params[0], params[1], \\",
            "            params[2]",
            "product = x1 * p1",
            "return p0 + product**p2",
        ])
        expr = expr_substitution(func, [2.0, 3.0, 2.0])
        self.assertIsNotNone(expr, "反斜杠续行的逐项解包未能解析")
        self.assertEqual(expr.free_symbols, {X1})
        self.assertAlmostEqual(float(expr.subs(X1, 1.5)), 2.0 + (1.5 * 3.0) ** 2,
                               places=9)

    def test_tuple_unpacking_parenthesized_multiline(self):
        """回归（20260921-161549 order75 最优样本）：RHS 是跨行括号元组——

        ``p0, ..., p3 = (params[0], params[1], params[2],\\n params[3])``
        首行括号未闭合，旧流程既不在合并后的行上重试逐项解包，也没有
        "剥掉 RHS 包裹括号"的兜底——p0..p6 成为自由符号 → 剪枝与
        held-out 再次被跳过。"""
        func = _spec([
            "p0, p1, p2, p3 = (params[0], params[1], params[2],",
            "                  params[3])",
            "aspect = x1 * params[4]",
            "return (p0",
            "        + p1 * aspect ** p2",
            "        + p3)",
        ])
        expr = expr_substitution(func, [2.0, 3.0, 2.0, 10.0, 1.5])
        self.assertIsNotNone(expr, "跨行括号元组解包未能解析")
        self.assertEqual(expr.free_symbols, {X1})
        self.assertAlmostEqual(float(expr.subs(X1, 2.0)),
                               2.0 + 3.0 * (1.5 * 2.0) ** 2 + 10.0, places=9)

    def test_numpy_prefix_removed(self):
        func = _spec(["return np.sin(params[0]*x1)"])
        expr = expr_substitution(func, [2.0])
        self.assertIsNotNone(expr)
        self.assert_expr_close(expr, sp.sin(2 * X1))

    def test_intermediate_variable_substituted(self):
        func = _spec([
            "a = params[0]*x1",
            "return a + params[1]",
        ])
        expr = expr_substitution(func, [2.0, 5.0])
        self.assertEqual(sp.simplify(expr - (2 * X1 + 5)), 0)

    def test_maximum_alias(self):
        func = _spec(["return maximum(params[0]*x1, params[1])"])
        expr = expr_substitution(func, [2.0, 1.0])
        self.assertIsNotNone(expr)
        self.assert_expr_close(expr, sp.Max(2 * X1, 1))

    def test_no_independents_returns_none(self):
        func = "def equation(x1, params):\n    return params[0]*x1\n"
        self.assertIsNone(expr_substitution(func, [1.0]))

    def test_no_return_returns_none(self):
        func = "- Independents: x1\ndef equation(x1, params):\n    x = params[0]*x1\n"
        self.assertIsNone(expr_substitution(func, [1.0]))

    def test_comment_stripped(self):
        func = _spec([
            "return params[0]*x1  # 这里注释掉的内容不应参与解析",
        ])
        expr = expr_substitution(func, [4.0])
        self.assertEqual(sp.simplify(expr - 4 * X1), 0)

    def test_empty_params_returns_none(self):
        # 无参数代入时 params[i] 仍留在表达式中，SymPy 无法解析 -> None
        func = _spec(["return params[0]*x1"])
        self.assertIsNone(expr_substitution(func, []))

    def test_where_alias_piecewise(self):
        # where(cond, a, b) -> Piecewise((a, cond), (b, True))
        func = _spec(["return where(params[0]*x1 > 0, params[1]*x1, params[2])"])
        expr = expr_substitution(func, [1.0, 2.0, 3.0])
        self.assertIsNotNone(expr)
        self.assertAlmostEqual(float(expr.subs(X1, 1.0)), 2.0)
        self.assertAlmostEqual(float(expr.subs(X1, -1.0)), 3.0)

    def test_high_index_not_corrupted_by_prefix(self):
        """回归：params[1] 不应作为 params[10] 的前缀被误替换。

        原实现逐个 `func.replace(f"params[{i}]", ...)`，i=1 时会命中
        `params[10]` 内部，导致 index>=10 的参数被腐蚀。此处固化修复后的正确行为。
        """
        func = _spec(["return params[1]*x1 + params[10]"])
        params = [float(i) for i in range(11)]  # params[1]=1.0, params[10]=10.0
        expr = expr_substitution(func, params)
        self.assertIsNotNone(expr, "高索引参数被前缀腐蚀导致解析失败")
        self.assertEqual(sp.simplify(expr - (1 * X1 + 10)), 0)


class TypeCastWrapperTest(_ExprTestCase):
    """numpy 纯类型转换包装必须去掉（缺陷 2 的根因，2026-09-29 确证）。

    最小复现：``asarray(lambda12, dtype=float)`` 让 sympy 1.14 抛
    ``ValueError: Unknown options: {'dtype': float}`` → 中间变量行被跳过 → 符号成孤儿 →
    ``return`` 无法求值 → 样本返回 None → ``find_best_eq`` 拿不到选解信息 → 报告缺
    「发布解选择」小节（实测 ``ab-iso6-no6/..._20260929-091844`` 的 best 样本）。
    实测分布：四臂 1024 条样本里 **28** 条含这类包装、**全部带 dtype=**。
    """

    def test_the_failing_form_now_substitutes(self):
        func = _spec(["l12 = np.asarray(x1, dtype=float)",
                      "l23 = np.asarray(x2, dtype=float)",
                      "return params[0]*l12 + params[1]*l23 + params[2]"],
                     independents="x1, x2", sig="equation(x1, x2, params)")
        expr = expr_substitution(func, [2.0, 3.0, 1.0])
        self.assertIsNotNone(expr)                      # 修复前恒为 None
        self.assertEqual(sp.simplify(expr - (2 * X1 + 3 * X2 + 1)), 0)

    def test_astype_and_array_wrappers_too(self):
        func = _spec(["a = x1.astype(float)",
                      "return params[0]*np.array(a) + params[1]"])
        expr = expr_substitution(func, [2.0, 1.0])
        self.assertEqual(sp.simplify(expr - (2 * X1 + 1)), 0)

    def test_value_changing_calls_are_left_alone(self):
        """只删**纯类型转换**：``maximum`` / ``where`` 会改变值，必须原样保留。"""
        from drsr_420.analysis.expr_parse import strip_type_cast_wrappers

        for text in ("maximum(x1, x2)", "where(x1 >= 0, p0, p1)", "power(x1, x2)"):
            with self.subTest(text=text):
                self.assertEqual(strip_type_cast_wrappers(text), text)


class ParameterArrayAliasTest(_ExprTestCase):
    """``p = params`` 是**整体别名**、不是解包（2026-09-29 确证）。

    旧逻辑把它按"名字按位置对应 params 列表"处理 → ``p`` 被绑成 ``params[0]`` 的**值** →
    后续 ``p[3]`` 经名字替换变成 ``193.05[3]``（数字被索引）→ sympy 只报内部错误
    ``Integer.__new__() missing 1 required positional argument: 'i'``，没有任何可读线索。
    实测 ``ab-fix6-control/..._20260928-154613`` 的 best 样本正是这种写法（``p = params``
    + ``return p[0] + p[1]*a12 + …``），后果是 contains：该样本解析不出 → find_best_eq
    拿不到选解信息 → 报告缺「发布解选择」小节。
    """

    def test_whole_array_alias_indexes_by_position(self):
        func = _spec(["p = params", "return p[1]*x1 + p[0]"])
        expr = expr_substitution(func, [10.0, 20.0])
        self.assertIsNotNone(expr)                      # 修复前恒为 None
        self.assertEqual(sp.simplify(expr - (20 * X1 + 10)), 0)

    def test_slice_alias_too(self):
        func = _spec(["p = params[:2]", "return p[1]*x1 + p[0]"])
        expr = expr_substitution(func, [10.0, 20.0])
        self.assertEqual(sp.simplify(expr - (20 * X1 + 10)), 0)

    def test_a_scalar_binding_is_still_supported(self):
        """``q = params[0]``（标量、无冒号）必须仍走逐项分支绑成参数值。"""
        func = _spec(["q = params[0]", "return q*x1"])
        expr = expr_substitution(func, [7.0])
        self.assertEqual(sp.simplify(expr - 7 * X1), 0)

    def test_unpacking_is_still_supported(self):
        """真解包（左侧多个名字）必须不受影响。"""
        func = _spec(["p0, p1 = params[:2]", "return p1*x1 + p0"])
        expr = expr_substitution(func, [10.0, 20.0])
        self.assertEqual(sp.simplify(expr - (20 * X1 + 10)), 0)


class ComprehensionUnpackTest(_ExprTestCase):
    """``p0, p1, ... = (params[i] for i in range(n))``：与 ``params[:n]`` 等价的解包。

    实测 ``ab-fix6-treatment/..._20260928-154609`` samples_42 的 best 样本用了生成器
    表达式，RHS 既非单一 ``params[...]``、也非逐项逗号列表，两套既有识别全部落空 →
    p0..p5 沦为自由符号 → 整条样本返回 None（find_best_eq 拿不到选解信息）。
    """

    def test_generator_expression_unpack(self):
        func = _spec([
            "p0, p1, p2 = (params[i] for i in range(3))",
            "return p0 + p1*x1 + p2*x1**2",
        ])
        expr = expr_substitution(func, [5.0, 3.0, 2.0])
        self.assertIsNotNone(expr)                      # 修复前恒为 None
        self.assertEqual(sp.simplify(expr - (5 + 3 * X1 + 2 * X1**2)), 0)

    def test_list_comprehension_unpack(self):
        func = _spec([
            "p0, p1 = [params[i] for i in range(2)]",
            "return p0 + p1*x1",
        ])
        expr = expr_substitution(func, [7.0, 4.0])
        self.assertEqual(sp.simplify(expr - (7 + 4 * X1)), 0)

    def test_range_len_params(self):
        func = _spec([
            "p0, p1 = (params[i] for i in range(len(params)))",
            "return p1*x1 + p0",
        ])
        expr = expr_substitution(func, [10.0, 20.0])
        self.assertEqual(sp.simplify(expr - (20 * X1 + 10)), 0)

    def test_count_mismatch_is_safely_rejected(self):
        """名字数多于 range 上界（运行时会 ValueError）：安全失败，不猜。"""
        func = _spec([
            "p0, p1, p2 = (params[i] for i in range(2))",
            "return p0 + p1*x1 + p2",
        ])
        self.assertIsNone(expr_substitution(func, [1.0, 2.0, 3.0]))


class NumpyPowerCollisionTest(_ExprTestCase):
    """``power`` 既是 sympy 全局函数、又被模型用作中间变量名：显式 ``np.power`` 必须
    仍解析成幂，而不是那个**符号**。

    实测 ``ab-iso6-head/..._20260929-091848`` samples_12：先 ``power = np.power(aspect, p1)``
    定义了中间变量 ``power``，后又写 ``np.power(lambda23, p4)``。剥掉 ``np.`` 后裸
    ``power(...)`` 命中 local_dict 里的符号 → ``'Symbol' object is not callable`` →
    该行被跳过 → ``l23_term`` 成孤儿 → 整条样本返回 None。
    """

    def test_power_variable_then_np_power_call(self):
        func = _spec([
            "power = np.power(x1, params[1])",
            "saturating = params[0] * power / (1.0 + params[2] * power)",
            "return saturating + params[3] * np.power(x1, params[4])",
        ])
        expr = expr_substitution(func, [2.0, 3.0, 0.5, 4.0, 2.0])
        self.assertIsNotNone(expr)                      # 修复前恒为 None
        self.assertEqual(expr.free_symbols, {X1})
        pw = X1**3
        expected = 2.0 * pw / (1.0 + 0.5 * pw) + 4.0 * X1**2
        self.assert_expr_close(expr, expected)

    def test_nested_np_power_is_rewritten(self):
        func = _spec(["return params[0] * np.power(np.power(x1, params[1]), params[2])"])
        expr = expr_substitution(func, [5.0, 2.0, 3.0])
        self.assertIsNotNone(expr)
        self.assert_expr_close(expr, 5 * X1**6)     # (x1**2)**3 == x1**6


class WherePiecewiseHelpersTest(_ExprTestCase):
    """`where(...)` -> `Piecewise((a, cond), (b, True))` 改写所用的括号/切分工具。"""

    def test_split_top_level_ignores_nested_commas(self):
        self.assertEqual(
            split_top_level("a, Max(b, c), d"),
            ["a", " Max(b, c)", " d"],
        )

    def test_split_top_level_ignores_deeply_nested(self):
        self.assertEqual(
            split_top_level("Max(a, Min(b, c)), d"),
            ["Max(a, Min(b, c))", " d"],
        )

    def test_split_top_level_ignores_brackets_and_strings(self):
        self.assertEqual(split_top_level("a, [1, 2], 'x,y'"), ["a", " [1, 2]", " 'x,y'"])

    def test_split_top_level_empty(self):
        self.assertEqual(split_top_level(""), [""])

    def test_find_matching_paren_skips_inner_parens(self):
        self.assertEqual(find_matching_paren("f(a, g(b), c)", 1), 12)

    def test_find_matching_paren_unbalanced_returns_minus_one(self):
        self.assertEqual(find_matching_paren("f(a, g(b)", 1), -1)

    def test_rewrite_simple_call(self):
        self.assertEqual(
            rewrite_where_calls("where(c, a, b)"),
            "Piecewise((a, c), (b, True))",
        )

    def test_rewrite_keeps_nested_commas_intact(self):
        self.assertEqual(
            rewrite_where_calls("where(c, Max(a, b), d)"),
            "Piecewise((Max(a, b), c), (d, True))",
        )

    def test_rewrite_recurses_into_nested_where(self):
        self.assertEqual(
            rewrite_where_calls("where(c1, where(c2, a, b), d)"),
            "Piecewise((Piecewise((a, c2), (b, True)), c1), (d, True))",
        )

    def test_rewrite_does_not_touch_longer_identifier(self):
        """`somewhere(...)` 不是 where 调用，不应被改名（旧实现是全局 str.replace）。"""
        self.assertEqual(rewrite_where_calls("somewhere(x, y)"), "somewhere(x, y)")

    def test_rewrite_does_not_touch_prefixed_call(self):
        """`np.where(...)` 由上游先剥掉 np. 前缀再进入本函数，此处不应命中。"""
        self.assertEqual(rewrite_where_calls("np.where(a, b, c)"), "np.where(a, b, c)")

    def test_rewrite_wrong_arity_raises(self):
        """参数个数不是 3 时显式报错；`Piecewise(x1 > 0)` 会被 SymPy 静默求成 nan。"""
        with self.assertRaises(WhereArityError):
            rewrite_where_calls("where(c, a)")
        with self.assertRaises(WhereArityError):
            rewrite_where_calls("where(c)")

    def test_rewrite_tolerates_trailing_comma(self):
        self.assertEqual(
            rewrite_where_calls("where(c, a, b,)"),
            "Piecewise((a, c), (b, True))",
        )

    def test_rewrite_unbalanced_keeps_text(self):
        self.assertEqual(rewrite_where_calls("where(c, a"), "where(c, a")


class WherePiecewiseTest(_ExprTestCase):
    """回归：嵌套逗号 / 嵌套调用 / 非行尾写法等旧实现下失效的场景。"""

    def test_comma_inside_argument(self):
        """where(c, maximum(a, b), d)：旧实现按 split(',') 切分会错位并丢弃分支。"""
        func = _spec([
            "return where(params[0]*x1 > 0, maximum(params[1]*x1, params[2]), params[3])",
        ])
        expr = expr_substitution(func, [1.0, 2.0, 3.0, 4.0])
        self.assertIsNotNone(expr, "参数内含逗号的 where(...) 未能解析")
        # x1 > 0 分支取 Max(2*x1, 3)，否则取 4
        self.assertAlmostEqual(float(expr.subs(X1, 1.0)), 3.0)
        self.assertAlmostEqual(float(expr.subs(X1, -1.0)), 4.0)

    def test_nested_where_keeps_all_branches(self):
        func = _spec([
            "return where(x1 > 0, where(x1 > 1, params[0], params[1]), params[2])",
        ])
        expr = expr_substitution(func, [10.0, 20.0, 30.0])
        self.assertIsNotNone(expr, "嵌套 where 未能解析（旧实现会丢分支）")
        self.assertAlmostEqual(float(expr.subs(X1, 2.0)), 10.0)
        self.assertAlmostEqual(float(expr.subs(X1, 0.5)), 20.0)
        self.assertAlmostEqual(float(expr.subs(X1, -0.5)), 30.0)

    def test_where_not_at_end_of_line(self):
        """旧正则要求闭括号后紧跟换行，`where(...) + x` 会被整条漏改。"""
        func = _spec(["return where(x1 > 0, params[0], params[1]) + params[2]"])
        expr = expr_substitution(func, [1.0, 2.0, 3.0])
        self.assertIsNotNone(expr, "非行尾的 where(...) 未能解析")
        self.assertAlmostEqual(float(expr.subs(X1, 1.0)), 4.0)
        self.assertAlmostEqual(float(expr.subs(X1, -1.0)), 5.0)

    def test_where_without_trailing_newline(self):
        func = _spec(["return where(x1 > 0, params[0], params[1])"]).rstrip("\n")
        expr = expr_substitution(func, [1.0, 2.0])
        self.assertIsNotNone(expr, "函数串末尾无换行时 where(...) 未能解析")
        self.assertAlmostEqual(float(expr.subs(X1, 1.0)), 1.0)
        self.assertAlmostEqual(float(expr.subs(X1, -1.0)), 2.0)

    def test_multiline_where(self):
        func = _spec([
            "return where(x1 > 0,",
            "             params[0],",
            "             params[1])",
        ])
        expr = expr_substitution(func, [7.0, 8.0])
        self.assertIsNotNone(expr)
        self.assertAlmostEqual(float(expr.subs(X1, 1.0)), 7.0)
        self.assertAlmostEqual(float(expr.subs(X1, -1.0)), 8.0)

    def test_where_in_intermediate_variable(self):
        func = _spec([
            "a = where(x1 > 0, params[0], params[1])",
            "return a * params[2]",
        ])
        expr = expr_substitution(func, [1.0, 2.0, 5.0])
        self.assertIsNotNone(expr)
        self.assertAlmostEqual(float(expr.subs(X1, 1.0)), 5.0)
        self.assertAlmostEqual(float(expr.subs(X1, -1.0)), 10.0)

    def test_numpy_where_prefix(self):
        func = _spec(["return np.where(x1 > 0, params[0], params[1])"])
        expr = expr_substitution(func, [1.0, 2.0])
        self.assertIsNotNone(expr)
        self.assertAlmostEqual(float(expr.subs(X1, 1.0)), 1.0)

    def test_sympy_style_piecewise_passes_through(self):
        """模型直接写 SymPy 风格的 Piecewise 时不应被改写坏。"""
        func = _spec(["return Piecewise((params[0], x1 > 0), (params[1], True))"])
        expr = expr_substitution(func, [1.0, 2.0])
        self.assertIsNotNone(expr)
        self.assertAlmostEqual(float(expr.subs(X1, 1.0)), 1.0)
        self.assertAlmostEqual(float(expr.subs(X1, -1.0)), 2.0)

    def test_where_with_wrong_arity_returns_none_not_crash(self):
        """旧实现在参数不足时抛 IndexError 打断整条流程；现在应安静返回 None。"""
        func = _spec(["return where(x1 > 0, params[0])"])
        self.assertIsNone(expr_substitution(func, [1.0]))

    def test_where_with_one_argument_returns_none(self):
        func = _spec(["return where(x1 > 0)"])
        self.assertIsNone(expr_substitution(func, []))

    def test_aliases_respect_identifier_boundaries(self):
        """回归：旧实现 `str.replace("maximum", "Max")` 会把 maximum_likelihood 改成 Max_likelihood。"""
        func = _spec(["return maximum_likelihood * x1"],
                     independents="x1, maximum_likelihood")
        expr = expr_substitution(func, [1.0])
        self.assertIsNotNone(expr)
        names = {s.name for s in expr.free_symbols}
        self.assertIn("maximum_likelihood", names)
        self.assertNotIn("Max_likelihood", names)

    def test_maximum_alias_still_applied(self):
        for name in ("maximum", "np.maximum", "numpy.maximum"):
            with self.subTest(name=name):
                func = _spec([f"return {name}(params[0]*x1, params[1])"])
                expr = expr_substitution(func, [2.0, 1.0])
                self.assertIsNotNone(expr)
                self.assert_expr_close(expr, sp.Max(2 * X1, 1))


class ConditionNormalizationTest(_ExprTestCase):
    """`==` / `!=` -> Eq / Ne：SymPy 不重载 ==，会退化成 Python 的 False。"""

    def test_eq_becomes_eq(self):
        self.assertEqual(normalize_condition("x1 == 0"), "Eq(x1, 0)")

    def test_ne_becomes_ne(self):
        self.assertEqual(normalize_condition("x1 != 0"), "Ne(x1, 0)")

    def test_plain_comparison_untouched(self):
        for cond in ("x1 > 0", "x1 >= 0", "x1 < 1", "x1 <= 1"):
            with self.subTest(cond=cond):
                self.assertEqual(normalize_condition(cond), cond)

    def test_eq_inside_parens_is_not_top_level(self):
        # 顶层没有 == 时不动括号里的内容
        self.assertEqual(normalize_condition("(x1 == 0)"), "(x1 == 0)")

    def test_chained_eq(self):
        self.assertEqual(normalize_condition("x1 == x2 == 0"), "Eq(x1, Eq(x2, 0))")

    def test_where_with_eq_condition_keeps_true_branch(self):
        """回归：`where(x1 == 0, a, b)` 曾被 SymPy 静默求成 b（条件塌缩）。"""
        func = _spec(["return where(x1 == 0, params[0], params[1])"])
        expr = expr_substitution(func, [11.0, 22.0])
        self.assertIsNotNone(expr, "== 条件导致整条表达式塌缩/解析失败")
        self.assertAlmostEqual(float(expr.subs(X1, 0.0)), 11.0)
        self.assertAlmostEqual(float(expr.subs(X1, 1.0)), 22.0)

    def test_where_with_ne_condition(self):
        func = _spec(["return where(x1 != 0, params[0], params[1])"])
        expr = expr_substitution(func, [11.0, 22.0])
        self.assertIsNotNone(expr)
        self.assertAlmostEqual(float(expr.subs(X1, 0.0)), 22.0)
        self.assertAlmostEqual(float(expr.subs(X1, 1.0)), 11.0)

    def test_where_with_ge_condition(self):
        func = _spec(["return where(x1 >= 0, params[0], params[1])"])
        expr = expr_substitution(func, [11.0, 22.0])
        self.assertIsNotNone(expr)
        self.assertAlmostEqual(float(expr.subs(X1, 5.0)), 11.0)
        self.assertAlmostEqual(float(expr.subs(X1, -5.0)), 22.0)


class IntermediateVariableLineTest(_ExprTestCase):
    """中间变量赋值行的识别：旧实现用 `"=" in line` + `split("=")`，被 `>=` 击穿。"""

    def test_assignment_line_with_ge_condition(self):
        """回归：`a = where(x1 >= 0, p0, p1)` 曾因 split("=") 解析失败而**静默返回裸符号 a**。"""
        func = _spec([
            "a = where(x1 >= 0, params[0], params[1])",
            "return a",
        ])
        expr = expr_substitution(func, [11.0, 22.0])
        self.assertIsNotNone(expr)
        self.assertNotEqual(expr, sp.Symbol("a"), "未替换中间变量，静默返回了裸符号 a")
        self.assertAlmostEqual(float(expr.subs(X1, 5.0)), 11.0)
        self.assertAlmostEqual(float(expr.subs(X1, -5.0)), 22.0)

    def test_assignment_line_with_eq_condition(self):
        func = _spec([
            "a = where(x1 == 0, params[0], params[1])",
            "return a",
        ])
        expr = expr_substitution(func, [11.0, 22.0])
        self.assertAlmostEqual(float(expr.subs(X1, 0.0)), 11.0)
        self.assertAlmostEqual(float(expr.subs(X1, 5.0)), 22.0)

    def test_assignment_chain_of_intermediates(self):
        func = _spec([
            "a = params[0]*x1",
            "b = a + params[1]",
            "return b",
        ])
        expr = expr_substitution(func, [2.0, 3.0])
        self.assertEqual(sp.simplify(expr - (2 * X1 + 3)), 0)

    def test_unparsable_assignment_line_is_skipped(self):
        # 解析失败的行应被跳过而不是崩溃；缺失的符号保持原样
        func = _spec([
            "a = params[0]*((",
            "return params[0]*x1",
        ])
        expr = expr_substitution(func, [4.0])
        self.assertIsNotNone(expr)
        self.assertEqual(sp.simplify(expr - 4 * X1), 0)


def _guard_spec(params_index: int) -> str:
    """模型手写的防除零骨架：`(params[k] == 0) * 1e-12`。

    真实事故（MRFCompress-Cuboid_20260919-143926 的最优样本）就是这种写法：参数代入后
    比较两侧都成了常量，``parse_expr`` 把它求值成 Python ``bool``，``bool * Float`` 抛
    TypeError → 中间变量行被跳过 → return 里的名字成了**孤儿符号** → 求值全 NaN →
    敏感度被当成 0 → 剪枝把公式剪成常数 193.054（NMSE 8.7e-07 → 6.81）。
    """
    k = params_index
    return _spec([
        "e12 = x1 - 1.0",
        f"sat12 = params[0] * e12 / (params[{k}] + abs(e12) + (params[{k}] == 0) * 1e-12)",
        "return sat12 + params[1]",
    ])


class ConstantComparisonFoldingTest(_ExprTestCase):
    """``(params[k] == 0) * 1e-12`` 代入参数后是**常量比较**：必须折叠成 0/1。"""

    def test_constant_comparison_folded_and_line_parses(self):
        expr = expr_substitution(_guard_spec(1), [2.0, 200.0, 0.0])
        self.assertIsNotNone(expr, "防除零写法不得让整条中间变量行解析失败")
        self.assertEqual({str(s) for s in expr.free_symbols}, {"x1"})
        # 守卫条件为假 → 分母里加的是 0，与手工代入逐点一致
        expected = 2.0 * (X1 - 1.0) / (200.0 + sp.Abs(X1 - 1.0)) + 200.0
        self.assert_expr_close(expr, expected)

    def test_folds_to_zero_when_guard_is_inactive(self):
        # 守卫条件为假 → 加 0，与手工代入完全一致
        func = _spec(["return params[0] + (params[0] == 0) * 1e-12"])
        expr = expr_substitution(func, [7.0])
        self.assertEqual(expr, sp.Float(7.0))

    def test_symbolic_comparison_left_alone(self):
        # 含符号的比较不属于折叠范围：在这里折叠会静默丢掉分支（那是 normalize_condition 的事）
        self.assertEqual(fold_constant_comparisons("x1 == 0"), "x1 == 0")
        self.assertEqual(fold_constant_comparisons("where(x1 == 0, 1, 2)").count("x1 == 0"), 1)

    def test_non_python_text_passes_through(self):
        self.assertEqual(fold_constant_comparisons("Piecewise((a, x1 >= 0), (b, True))"),
                         "Piecewise((a, x1 >= 0), (b, True))")
        self.assertEqual(fold_constant_comparisons("2 x"), "2 x")


class OrphanSymbolRejectionTest(_ExprTestCase):
    """return 用到未定义符号（中间变量行解析失败的后果）时必须拒绝解析。

    这类表达式求值必然全非有限，交给下游只会静默产出错误结论——真实事故里剪枝据此
    把公式剪成了常数。
    """

    def test_orphan_intermediate_symbol_is_rejected(self):
        func = _spec([
            "a = params[0]*((",
            "return a + x1",             # a 未定义：孤儿符号
        ])
        self.assertIsNone(expr_substitution(func, [1.0]))

    def test_undefined_symbol_in_expression_is_rejected(self):
        func = _spec(["b = x1 + params[0]", "return b + c"])
        self.assertIsNone(expr_substitution(func, [1.0]))

    def test_sample_count_symbol_n_is_still_allowed(self):
        # N（样本数）按历史约定始终保留为符号，不得被当成未定义符号拒绝
        func = _spec(["return params[0]*x1/N"])
        expr = expr_substitution(func, [6.0])
        self.assertIsNotNone(expr)
        self.assertIn(sp.Symbol("N"), expr.free_symbols)


class SympyNameCollisionTest(_ExprTestCase):
    """中间变量/自变量与 sympy 全局名撞名：必须解析成那个符号，而不是 sympy 的同名对象。

    回归自一次真实运行：骨架里写了 ``poly = params[5]*a12 + ...``，而 ``poly`` 在 sympy
    里是函数，于是 ``poly * f23`` 抛
    ``TypeError: unsupported operand type(s) for *: 'function' and 'Symbol'``，
    整个收尾分析（物理解释 / 剪枝 / 预览图）被跳过。``E`` / ``pi`` / ``gamma``
    这类更危险：旧实现**不报错**，静默把中间变量换成常量，产出错误的表达式。
    """

    def test_intermediate_named_poly_is_not_the_sympy_function(self):
        func = _spec([
            "poly = params[0]*x1",
            "return poly",
        ])
        expr = expr_substitution(func, [2.0])
        self.assertIsNotNone(expr)
        self.assertEqual(sp.simplify(expr - 2 * X1), 0)

    def test_poly_multiplied_by_another_intermediate(self):
        """`poly * f` 就是抛 TypeError 的那一行。"""
        func = _spec([
            "poly = params[0]*x1",
            "f = params[1] + x1",
            "return poly*f",
        ])
        expr = expr_substitution(func, [2.0, 3.0])
        self.assertIsNotNone(expr)
        self.assertEqual(sp.simplify(expr - 2 * X1 * (X1 + 3)), 0)

    def test_intermediate_named_E_is_not_eulers_number(self):
        """`E` 是最隐蔽的一类：旧实现不报错，静默换成自然常数 e。"""
        func = _spec([
            "E = params[0]*x1",
            "return E",
        ])
        expr = expr_substitution(func, [2.0])
        self.assertIsNotNone(expr)
        self.assertFalse(expr.has(sp.E), "中间变量 E 被静默替换成了自然常数 e")
        self.assertEqual(sp.simplify(expr - 2 * X1), 0)

    def test_independent_variable_named_like_a_sympy_function(self):
        """自变量同样可能撞名：`beta` 在 sympy 里是函数。"""
        func = _spec(["return params[0]*beta"], independents="beta")
        expr = expr_substitution(func, [2.0])
        self.assertIsNotNone(expr)
        self.assertEqual(expr.free_symbols, {sp.Symbol("beta")})
        self.assertAlmostEqual(float(expr.subs(sp.Symbol("beta"), 3.0)), 6.0)


class ParseAuditTest(unittest.TestCase):
    """收尾自检：解析失败率必须分「截断样本」与「解析器不支持的写法」两类计数。

    否则一个总百分比读不出该修采样侧（max_tokens 截断）还是解析器侧（写法不支持）。
    """

    _OK = _spec(["return params[0]*x1 + params[1]"])
    #: 正文无 return —— 模拟 max_tokens 截断的不完整样本（评估器未打分，score 为 None）
    _TRUNCATED = "def equation(x1, params):\n    \"\"\"只有 docstring，正文被截断\"\"\"\n"
    #: 有 return 但引用了未定义符号 —— 解析器不支持的写法
    _UNSUPPORTED = _spec(["return params[0]*unknown_symbol"])

    def _write(self, root: pathlib.Path, order, func, score, name=None):
        samples = root / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        (samples / (name or f"samples_{order}.json")).write_text(
            json.dumps({"sample_order": order, "score": score,
                        "function": func, "params": [1.0, 2.0]}), encoding="utf-8")

    def test_classify_three_kinds(self):
        self.assertEqual(classify_sample(self._OK, [1.0, 2.0])[0], "ok")
        self.assertEqual(classify_sample(self._TRUNCATED, [1.0])[0], "truncated")
        kind, warn = classify_sample(self._UNSUPPORTED, [1.0, 2.0])
        self.assertEqual(kind, "unsupported")
        self.assertIn("未定义符号", warn)

    def test_audit_splits_two_failure_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self._write(root, 1, self._OK, -1.0)
            self._write(root, 2, self._TRUNCATED, None)
            self._write(root, 3, self._UNSUPPORTED, -5.0)
            audit = audit_parse_failures(str(root))
            self.assertEqual(audit["n_total"], 3)
            self.assertEqual(audit["n_ok"], 1)
            self.assertEqual(audit["n_failed"], 2)
            self.assertEqual(audit["n_truncated"], 1)
            self.assertEqual(audit["n_unsupported"], 1)
            self.assertAlmostEqual(audit["failure_rate"], 2 / 3)
            self.assertEqual(audit["truncated"][0]["sample_order"], 2)
            self.assertEqual(audit["unsupported"][0]["sample_order"], 3)

    def test_audit_of_empty_dir_has_zero_and_none_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = audit_parse_failures(tmp)
            self.assertEqual(audit["n_total"], 0)
            self.assertIsNone(audit["failure_rate"])

    def test_top_k_copy_is_deduplicated_preferring_full_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self._write(root, 5, self._TRUNCATED, None, name="top3_samples_5.json")
            self._write(root, 5, self._OK, -1.0, name="samples_5.json")
            audit = audit_parse_failures(str(root))
            self.assertEqual(audit["n_total"], 1)
            self.assertEqual(audit["n_ok"], 1)      # 全量文件优先，截断副本被替换


if __name__ == "__main__":
    unittest.main()
