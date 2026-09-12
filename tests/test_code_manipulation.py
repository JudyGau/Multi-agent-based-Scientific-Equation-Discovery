"""code_manipulation 单元测试：AST 解析、函数/程序构造、调用重命名、装饰器识别。"""
import unittest

from drsr_420 import code_manipulation as cm


SPEC = (
    '"""spec docstring"""\n'
    "import numpy as np\n"
    "\n"
    "MAX_NPARAMS = 10\n"
    "\n"
    "@evaluate.run\n"
    "def evaluate(data: dict) -> float:\n"
    "    return 1.0\n"
    "\n"
    "@equation.evolve\n"
    "def equation(x1: np.ndarray, params: np.ndarray) -> np.ndarray:\n"
    '    """Equation to be evolved."""\n'
    "    return params[0] * x1\n"
)


class SanitizeTest(unittest.TestCase):
    def test_removes_null_bytes(self):
        self.assertEqual(cm.sanitize_code_text("a\x00b"), "ab")

    def test_non_str_passthrough(self):
        self.assertIsNone(cm.sanitize_code_text(None))
        self.assertEqual(cm.sanitize_code_text(123), 123)


class TextToProgramTest(unittest.TestCase):
    def test_parses_two_functions_and_preface(self):
        program = cm.text_to_program(SPEC)
        names = [f.name for f in program.functions]
        self.assertEqual(names, ["evaluate", "equation"])
        # preface 应含 import 与模块级赋值，且不含函数体
        self.assertIn("import numpy as np", program.preface)
        self.assertIn("MAX_NPARAMS = 10", program.preface)
        self.assertNotIn("def evaluate", program.preface)

    def test_function_args_and_return_type(self):
        program = cm.text_to_program(SPEC)
        fn = program.get_function("equation")
        self.assertIn("x1", fn.args)
        self.assertIn("params", fn.args)
        self.assertEqual(fn.return_type, "np.ndarray")

    def test_docstring_extracted(self):
        program = cm.text_to_program(SPEC)
        fn = program.get_function("equation")
        self.assertIsNotNone(fn.docstring)
        self.assertIn("Equation to be evolved.", fn.docstring)

    def test_find_function_index_errors(self):
        program = cm.text_to_program(SPEC)
        self.assertEqual(program.find_function_index("evaluate"), 0)
        with self.assertRaises(ValueError):
            program.find_function_index("nonexistent")

    def test_duplicate_function_raises(self):
        dup = "def f():\n    return 1\n\ndef f():\n    return 2\n"
        program = cm.text_to_program(dup)
        with self.assertRaises(ValueError):
            program.find_function_index("f")


class TextToFunctionTest(unittest.TestCase):
    def test_returns_single_function(self):
        fn = cm.text_to_function("def f(x):\n    return x\n")
        self.assertEqual(fn.name, "f")

    def test_multiple_functions_raise(self):
        with self.assertRaises(ValueError):
            cm.text_to_function("def f():\n    return 1\n\ndef g():\n    return 2\n")

    def test_body_setattr_strips_newlines(self):
        fn = cm.text_to_function("def f(x):\n    return x\n")
        self.assertFalse(fn.body.startswith("\n"))
        self.assertFalse(fn.body.endswith("\n"))

    def test_str_roundtrip_is_parseable(self):
        fn = cm.text_to_function("def f(x):\n    return x\n")
        reparsed = cm.text_to_function(str(fn))
        self.assertEqual(reparsed.name, "f")


class RenameFunctionCallsTest(unittest.TestCase):
    def test_renames_def_and_call(self):
        # rename_function_calls 重命名所有 "name(" 形式的 NAME token，
        # 包括函数定义名 —— 这正是 Island._generate_prompt 生成
        # `def equation_v0(...)` 并改写递归自调用的需要。
        code = "def equation(x):\n    return equation(x)\n"
        out = cm.rename_function_calls(code, "equation", "equation_v1")
        self.assertIn("def equation_v1(", out)
        self.assertIn("return equation_v1(x)", out)

    def test_body_only_renames_recursive_call(self):
        # 传入纯函数体（无 def）时仅重命名调用（_sample_to_program 的用法）
        body = "    return equation_v1(x) + 1\n"
        out = cm.rename_function_calls(body, "equation_v1", "equation")
        self.assertIn("equation(x)", out)

    def test_attribute_access_not_renamed(self):
        code = "y = obj.equation(1)\n"
        out = cm.rename_function_calls(code, "equation", "equation_v1")
        self.assertIn("obj.equation(1)", out)
        self.assertNotIn("equation_v1", out)

    def test_no_match_returns_unchanged(self):
        code = "y = other(1)\n"
        self.assertEqual(cm.rename_function_calls(code, "equation", "equation_v1"), code)


class GetFunctionsCalledTest(unittest.TestCase):
    def test_collects_direct_calls_excludes_attribute_access(self):
        code = "y = np.sin(x) + helper(x)\n"
        called = cm.get_functions_called(code)
        self.assertIn("helper", called)
        # 属性访问 np.sin 不是"函数调用"（is_attribute_access），应被排除
        self.assertNotIn("sin", called)

    def test_collects_attribute_access(self):
        code = "y = np.sin(x)\n"
        self.assertEqual(cm.get_functions_called(code), set())


class YieldDecoratedTest(unittest.TestCase):
    def test_yields_decorated_function_name(self):
        names = list(cm.yield_decorated(SPEC, "equation", "evolve"))
        self.assertEqual(names, ["equation"])

    def test_yields_evaluate_run(self):
        names = list(cm.yield_decorated(SPEC, "evaluate", "run"))
        self.assertEqual(names, ["evaluate"])

    def test_no_match_empty(self):
        self.assertEqual(list(cm.yield_decorated(SPEC, "nope", "nope")), [])


if __name__ == "__main__":
    unittest.main()
