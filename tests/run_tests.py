"""统一测试入口。

用法（项目根目录）：
    python tests/run_tests.py

注意：LocalSandbox 在 Windows 下使用 multiprocessing spawn 启动常驻 worker，
子进程会重新导入 __main__ 模块。若用 `python -m unittest`，__main__ 是
unittest/__main__.py（无 `if __name__ == '__main__':` 保护），会导致子进程
递归重跑测试。因此统一从这里启动，用本文件的 __main__ 保护避免该问题。
"""
import sys
import unittest

if __name__ == '__main__':
    sys.path.insert(0, '.')  # 保证 import drsr_420 可用
    suite = unittest.defaultTestLoader.discover('tests', pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
