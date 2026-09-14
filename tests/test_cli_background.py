"""--background_file：背景词以 backgrounds/*.txt 为规范源的 CLI 契约。

==============================================
为什么要有这个参数
==============================================

背景提示词曾经在内联在 4 组 MRF 的 .bat/.sh/IDE XML/example 脚本里、共 6 处副本；
backgrounds/*.txt 只是没人消费的死副本——改 txt 不改脚本，下一次实验会**静默
用回旧提示词**（用户改了提示词却跑不出效果，且无任何报错）。现在脚本只传
``--background_file <路径>``，txt 成为唯一来源，改动即刻生效，不存在同步问题。

本文件守住 resolve_background 的四条契约：
1. 两个来源互斥——都给直接报错，绝不允许"到底用了哪个"变得不可追溯；
2. 文件不存在 / 内容为空 → 明确终止，绝不静默开跑；
3. 文件内容去首尾空白（尾随换行不算正文）；
4. 只给 --background 时行为与历史完全一致（内联文本透传）。
"""
from __future__ import annotations

import tempfile
import unittest

from drsr_420.cli.main import build_parser, resolve_background


class ResolveBackgroundTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.parser = build_parser()

    def _args(self, *argv):
        # --data_csv 是必填参数：与本测试无关，统一填占位值
        return self.parser.parse_args(['--data_csv', 'placeholder.csv', *argv])

    def _write(self, name: str, content: str) -> str:
        import os
        path = os.path.join(self._tmp.name, name)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def test_reads_file_and_strips_trailing_whitespace(self):
        path = self._write('bg.txt', 'Hello background.\n')
        args = self._args('--background_file', path)
        self.assertEqual(resolve_background(self.parser, args), 'Hello background.')

    def test_inline_background_passthrough(self):
        args = self._args('--background', 'inline text')
        self.assertEqual(resolve_background(self.parser, args), 'inline text')

    def test_neither_given_returns_none(self):
        args = self._args()
        self.assertIsNone(resolve_background(self.parser, args))

    def test_both_sources_rejected(self):
        path = self._write('bg.txt', 'x')
        args = self._args('--background', 'a', '--background_file', path)
        with self.assertRaises(SystemExit):
            resolve_background(self.parser, args)

    def test_missing_file_terminates_loudly(self):
        args = self._args('--background_file', 'nope/not-exist.txt')
        with self.assertRaises(SystemExit):
            resolve_background(self.parser, args)

    def test_empty_file_terminates_loudly(self):
        path = self._write('empty.txt', '   \n')
        args = self._args('--background_file', path)
        with self.assertRaises(SystemExit):
            resolve_background(self.parser, args)


if __name__ == '__main__':
    unittest.main()
