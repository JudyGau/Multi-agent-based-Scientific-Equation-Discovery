"""evaluator_accelerate 单元测试：numba 装饰器注入与编译失败降级。"""
import ast
import unittest

import numpy as np

from drsr_420 import evaluator_accelerate as ea

PROGRAM = (
    "import numpy as np\n"
    "def func(x):\n"
    "    return x * 2\n"
)

PROGRAM_LIN = (
    "import numpy as np\n"
    "def equation(x1, x2, params):\n"
    "    return params[0] * x1 + params[1] * x2 + params[2]\n"
)

SAMPLE_ARGS = (
    np.array([1.0, 2.0]),
    np.array([3.0, 4.0]),
    np.array([0.5, -1.0, 0.25]),  # 与方程 params[2] 索引匹配（长度 >= 3）
)


class AddNumbaDecoratorTest(unittest.TestCase):
    def test_adds_decorator_and_import(self):
        out = ea.add_numba_decorator(PROGRAM, 'func')
        self.assertIn('import numba', out)
        self.assertIn('@numba.jit', out)
        self.assertIn('nopython', out)
        ast.parse(out)  # 结果仍是合法 Python

    def test_does_not_duplicate_import(self):
        out = ea.add_numba_decorator("import numba\n" + PROGRAM, 'func')
        self.assertEqual(out.count('import numba'), 1)

    def test_only_decorates_target_function(self):
        program = (
            "import numpy as np\n"
            "def other():\n"
            "    return 1\n"
            "def func():\n"
            "    return 2\n"
        )
        out = ea.add_numba_decorator(program, 'func')
        tree = ast.parse(out)
        decorators = {
            n.name: bool(n.decorator_list)
            for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        }
        self.assertTrue(decorators['func'])
        self.assertFalse(decorators['other'])


class TryAddNumbaDecoratorTest(unittest.TestCase):
    def setUp(self):
        try:
            import numba  # noqa: F401
        except Exception:
            self.skipTest('numba 未安装，跳过编译降级测试')

    def test_compatible_program_returns_accelerated(self):
        out = ea.try_add_numba_decorator(PROGRAM_LIN, 'equation', SAMPLE_ARGS)
        self.assertIn('@numba.jit', out)

        # 编译后的函数输出应与纯 Python 版本一致
        namespace = {}
        exec(out, namespace)
        expected = {}
        exec(PROGRAM_LIN, expected)
        np.testing.assert_allclose(
            namespace['equation'](*SAMPLE_ARGS),
            expected['equation'](*SAMPLE_ARGS),
        )

    def test_incompatible_program_falls_back(self):
        program_poly = (
            "import numpy as np\n"
            "def equation(x1, x2, params):\n"
            "    return params[0] * x1 + np.polyfit(x1, x2, 1)[0] * 0.0\n"
        )
        out = ea.try_add_numba_decorator(program_poly, 'equation', SAMPLE_ARGS)
        self.assertEqual(out, program_poly)  # 原样返回，未加速


if __name__ == '__main__':
    unittest.main()
