"""启动脚本一致性护栏：`example.bat` / `MRFCompress-3.bat` 必须与同名 `.sh` 等价。

为什么需要这组测试：Windows 批处理的 `call :label 参数` 会对参数做**第二轮解析**，
把参数里的脱字符（caret）**翻倍**——1 个变 2 个、2 个变 4 个；而 `echo` 又会把翻倍
后的结果显示回 1 个，于是写坏的值只在真正传给 Python 时才暴露（背景知识会带着
`^^{` 而不是 `^{` 进提示词）。`MRFCompress-3` 的 background 含 LaTeX 的 `^{...}`，
正是踩这个坑的样本：`example.bat` 因此把 background 放在 `BACKGROUND` 变量里传，
**不**放进 `call` 参数。下面的测试把这条结论固化成护栏，并守住两份脚本的逐一对应。

只做文本解析、不调用 cmd，所以在 Linux 上也能跑（`.bat` 的 CRLF 由 `.gitattributes`
的 `*.bat text eol=crlf` 保证）。
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLE_SH = _REPO_ROOT / "example.sh"
_EXAMPLE_BAT = _REPO_ROOT / "example.bat"
_MRF_SH = _REPO_ROOT / "MRFCompress-3.sh"
_MRF_BAT = _REPO_ROOT / "MRFCompress-3.bat"

#: bash 侧一行调用：run_problem <问题名> <csv> '<或">background<引号>
_PROBLEM_RE = re.compile(r"^\s*run_problem\s+(\S+)\s+(\S+)\s+(['\"])(.*)\3\s*$")
#: 批处理侧：set "BACKGROUND=..." 紧跟着 call :run_problem <问题名> "<csv>"
_SET_BG_RE = re.compile(r'^set "BACKGROUND=(.*)"$')
_CALL_RE = re.compile(r"^call :run_problem\s+(\S+)\s+(\S+)\s*$")
#: --flag value（value 可能是带引号的一整段）
_FLAG_RE = re.compile(r"(--\w+)\s+(\"[^\"]*\"|'[^']*'|\S+)")


def _sh_text(path: Path) -> str:
    """读 shell 脚本，并把反斜杠续行折成空格，使其可以按行解析。"""
    return path.read_text(encoding="utf-8").replace("\\\n", " ")


def _bat_text(path: Path) -> str:
    """读批处理脚本（Path.read_text 做全域换行归一，CRLF 会变成 LF）。"""
    return path.read_text(encoding="utf-8")


def _unquote(token: str) -> str:
    token = token.strip()
    return token[1:-1] if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'" else token


def _norm_csv(path: str) -> str:
    """`.sh` 写 `./data/x.csv`，`.bat` 写 `data/x.csv`，同一路径的两种写法。"""
    return path[2:] if path.startswith("./") else path


def _sh_problems() -> list[tuple[str, str, str]]:
    out = []
    for line in _sh_text(_EXAMPLE_SH).splitlines():
        m = _PROBLEM_RE.match(line)
        if m:
            out.append((m.group(1), m.group(2), m.group(4)))
    return out


def _bat_problems() -> list[tuple[str, str, str]]:
    """example.bat 的 (问题名, csv, background)；background 取自紧邻上一行的 set。"""
    out: list[tuple[str, str, str]] = []
    background: str | None = None
    for line in _bat_text(_EXAMPLE_BAT).splitlines():
        m = _SET_BG_RE.match(line)
        if m:
            background = m.group(1)
            continue
        m = _CALL_RE.match(line)
        if m:
            assert background is not None, f"call 前没有 set BACKGROUND：{line}"
            out.append((m.group(1), _unquote(m.group(2)), background))
            background = None
    return out


def _flags(text: str, llm_config: str) -> dict[str, str]:
    return {
        name.lstrip("-"): (llm_config if raw == '"%LLM_CONFIG%"' else _unquote(raw))
        for name, raw in _FLAG_RE.findall(text)
    }


def _default_llm_config(text: str) -> str:
    """从脚本里取出 LLM_CONFIG 的默认值（shell 用 ${VAR:-x}，批处理用 if not defined）。"""
    for pattern in (r"LLM_CONFIG:-([^}\"]+)", r'if not defined LLM_CONFIG set "LLM_CONFIG=([^"]+)"'):
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    raise AssertionError("脚本里找不到 LLM_CONFIG 默认值")


class BatchScriptParityTest(unittest.TestCase):
    """两个 .bat 与对应 .sh 逐项等价：问题名、数据路径、background、命令行参数。"""

    def test_shell_counterparts_still_exist(self):
        for path in (_EXAMPLE_SH, _MRF_SH):
            self.assertTrue(path.is_file(), f"{path.name} 被删了，但对应的 .bat 还在")

    def test_example_bat_declares_the_same_problems(self):
        sh, bat = _sh_problems(), _bat_problems()
        self.assertEqual(len(sh), 18, "example.sh 的问题数变了，example.bat 需要同步")
        self.assertEqual(len(bat), len(sh), "example.bat 与 example.sh 的问题数不一致")
        self.assertEqual([p[0] for p in sh], [p[0] for p in bat], "问题名/顺序不一致")
        self.assertEqual([_norm_csv(p[1]) for p in sh], [_norm_csv(p[1]) for p in bat], "csv 路径不一致")
        self.assertEqual([p[2] for p in sh], [p[2] for p in bat], "background 必须逐字节一致")
        # MRFCompress-3 的 background 带 LaTeX 的 ^{...}（3 个 ^）：这正是「call 会翻倍
        # 脱字符」的样本，上面那条逐字节比较一旦失效，这里先失败并提示护栏已过期。
        latex = {name: bg for name, _, bg in sh}["MRFCompress-3"]
        self.assertEqual(latex.count("^"), 3, "LaTeX 的 ^ 不再出现，脱字符护栏需要重估")

    def test_mrf_bat_passes_the_same_flags(self):
        # MRFCompress-3.sh 直接写死 --llm_config（没有环境变量默认值这一层），
        # 所以它的档案取值从命令行里取；.bat 那边用 %LLM_CONFIG% 的默认值。
        m = re.search(r"--llm_config\s+(\S+)", _sh_text(_MRF_SH))
        self.assertIsNotNone(m, "MRFCompress-3.sh 里没解析到 --llm_config")
        sh = _flags(_sh_text(_MRF_SH), _unquote(m.group(1)))
        self.assertTrue(sh.get("background"), "MRFCompress-3.sh 里没解析到 --background")
        bat = _flags(_bat_text(_MRF_BAT), _default_llm_config(_bat_text(_MRF_BAT)))
        for flags in (sh, bat):  # .sh 写 ./data/…，.bat 写 data/…，同一路径的两种写法
            flags["data_csv"] = _norm_csv(flags["data_csv"])
        self.assertEqual(sh, bat, "MRFCompress-3.bat 的参数与 MRFCompress-3.sh 不一致")

    def test_llm_config_defaults_match(self):
        sh = _default_llm_config(_sh_text(_EXAMPLE_SH))
        for path, text in ((_EXAMPLE_BAT, _bat_text(_EXAMPLE_BAT)),
                           (_MRF_BAT, _bat_text(_MRF_BAT))):
            self.assertEqual(_default_llm_config(text), sh, f"{path.name} 的默认档案与 example.sh 不一致")


class BatchScriptHygieneTest(unittest.TestCase):
    """批处理本身的硬性要求：CRLF + 无 BOM + UTF-8 + 双击可用 + 走模块入口。"""

    def test_files_are_utf8_crlf_without_bom(self):
        for path in (_EXAMPLE_BAT, _MRF_BAT):
            raw = path.read_bytes()
            with self.subTest(bat=path.name):
                self.assertFalse(raw.startswith(b"\xef\xbb\xbf"),
                                 "UTF-8 BOM 会让 cmd 把首行 @echo off 读成乱码命令")
                raw.decode("utf-8")  # 必须是 UTF-8，脚本内有 chcp 65001
                self.assertIn(b"chcp 65001", raw, "缺 chcp 65001，脚本里的中文会显示成乱码")
                self.assertEqual(raw.count(b"\n"), raw.count(b"\r\n"),
                                 "必须全是 CRLF：cmd 的 call/goto 按字节定位标签，LF-only 会找错标签")
                self.assertTrue(raw.endswith(b"\r\n"), "结尾缺少换行")

    def test_double_click_works_from_any_directory(self):
        for path in (_EXAMPLE_BAT, _MRF_BAT):
            with self.subTest(bat=path.name):
                self.assertIn('cd /d "%~dp0"', _bat_text(path),
                              "缺 cd /d \"%~dp0\"，双击时工作目录不对，data/ 与 config/ 都会找不到")

    def test_background_is_never_passed_as_call_argument(self):
        """call 会把参数里的脱字符翻倍，含 ^ 的 background 必须走变量传递。"""
        for line in _bat_text(_EXAMPLE_BAT).splitlines():
            if line.strip().startswith("call :run_problem"):
                self.assertNotIn("^", line, f"call 参数里的 ^ 会被翻倍，background 必须走变量：{line}")
        # 上面「走变量」的写法必须真的把 background 传下去：每个 call 前都有 set BACKGROUND
        self.assertEqual(len(_bat_problems()), 18)

    def test_module_entry_point_only(self):
        for path in (_EXAMPLE_BAT, _MRF_BAT):
            text = _bat_text(path)
            with self.subTest(bat=path.name):
                self.assertIn("-m drsr_420.cli.main", text)
                self.assertNotIn("main.py", text, "仓库根已无 main.py，入口必须是 python -m")
                self.assertNotIn("python3", text, "Windows 上没有 python3，只有 python")


if __name__ == "__main__":
    unittest.main()
