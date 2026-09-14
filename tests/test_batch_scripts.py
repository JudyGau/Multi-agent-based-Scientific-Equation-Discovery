"""启动脚本一致性护栏：4 组 MRF 运行配置各有三份等价物，必须逐项一致。

每一组配置（`MRFShear-Cuboid` / `MRFShear-Ellipsoid` / `MRFCompress-Cuboid` /
`MRFCompress-Ellipsoid`）都有三份：

    .idea/runConfigurations/<名>.xml   IDE 运行配置（入库，参数的源头）
    <名>.sh                             Linux/macOS 一行命令
    <名>.bat                            Windows 批处理（可直接双击）

外加 `example.sh` / `example.bat`（批量跑 18 个数据集，两行一组）。测试守住三件事：
① 三份等价物的参数逐项一致；② `background` 与数据列相符（列名以 `data/<名>/train.csv`
的表头为准）；③ 批处理本身的硬性要求（UTF-8 无 BOM、全 CRLF、`chcp 65001`、
`cd /d "%~dp0"`、走 `python -m` 入口）。

为什么盯得这么细：Windows 批处理的 `call :label 参数` 会对参数做**第二轮解析**，把参数
里的脱字符（caret）**翻倍**——1 个变 2 个、2 个变 4 个；而 `echo` 又会把翻倍后的结果
显示回 1 个，于是写坏的值只在真正传给 Python 时才暴露。`example.bat` 里 MRFCompress-3
的 background 含 LaTeX 的 `^{...}`，正是踩这个坑的样本，因此那份 background 走
`BACKGROUND` 变量传递、**不**放进 `call` 参数。

只做文本解析、不调用 cmd，所以在 Linux 上也能跑（`.bat` 的 CRLF 由 `.gitattributes`
的 `*.bat text eol=crlf` 保证）。
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_XML_DIR = _REPO_ROOT / ".idea" / "runConfigurations"

_EXAMPLE_SH = _REPO_ROOT / "example.sh"
_EXAMPLE_BAT = _REPO_ROOT / "example.bat"

#: 仓库根的所有批处理与 shell 脚本（example 的 + 4 组 MRF 的）
_ALL_BATS = sorted(_REPO_ROOT.glob("*.bat"))
_ALL_SHS = sorted(_REPO_ROOT.glob("*.sh"))
_MRF_BATS = [p for p in _ALL_BATS if p.stem.startswith("MRF")]
_MRF_SHS = [p for p in _ALL_SHS if p.stem.startswith("MRF")]
_XMLS = sorted(_XML_DIR.glob("*.xml"))

#: bash 侧一行调用：run_problem <问题名> <csv> '<或">background<引号>
_PROBLEM_RE = re.compile(r"^\s*run_problem\s+(\S+)\s+(\S+)\s+(['\"])(.*)\3\s*$")
#: bash 侧文件形态：run_problem_file <问题名> <csv> <backgrounds/*.txt 路径>
_PROBLEM_FILE_RE = re.compile(r"^\s*run_problem_file\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
#: 批处理侧：set "BACKGROUND=..." 紧跟着 call :run_problem <问题名> "<csv>"
_SET_BG_RE = re.compile(r'^set "BACKGROUND=(.*)"$')
_CALL_RE = re.compile(r"^call :run_problem\s+(\S+)\s+(\S+)\s*$")
#: 批处理侧文件形态：call :run_problem_file <问题名> "<csv>" "<backgrounds/*.txt>"
_CALL_FILE_RE = re.compile(r'^call :run_problem_file\s+(\S+)\s+"([^"]+)"\s+"([^"]+)"\s*$')
#: --flag value（value 可能是带引号的一整段）
_FLAG_RE = re.compile(r"(--\w+)\s+(\"[^\"]*\"|'[^']*'|\S+)")


def _read(path: Path) -> str:
    """读脚本（Path.read_text 做全域换行归一，CRLF 会变成 LF）。"""
    return path.read_text(encoding="utf-8")


def _sh_text(path: Path) -> str:
    """读 shell 脚本，并把反斜杠续行折成空格，使其可以按行解析。"""
    return _read(path).replace("\\\n", " ")


def _code_only(text: str) -> str:
    """去掉 rem/注释行，避免注释里的 `--词` 被当成参数。"""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().lower().startswith("rem"))


def _unquote(token: str) -> str:
    token = token.strip()
    return token[1:-1] if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'" else token


def _norm_csv(path: str) -> str:
    """`.sh`/XML 写 `./data/x.csv`，`.bat` 写 `data/x.csv`，同一路径的两种写法。"""
    return path[2:] if path.startswith("./") else path


def _flags(text: str, llm_config: str = "") -> dict[str, str]:
    """把命令行拆成 {参数名: 值}；`%LLM_CONFIG%` 用脚本里声明的默认值代入。"""
    return {
        name.lstrip("-"): (llm_config if raw == '"%LLM_CONFIG%"' else _unquote(raw))
        for name, raw in _FLAG_RE.findall(_code_only(text))
    }


def _default_llm_config(text: str) -> str:
    """脚本里 LLM_CONFIG 的默认值（shell 是 ${VAR:-x}，批处理是 if not defined）。"""
    for pattern in (r"LLM_CONFIG:-([^}\"]+)", r'if not defined LLM_CONFIG set "LLM_CONFIG=([^"]+)"'):
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    raise AssertionError("脚本里找不到 LLM_CONFIG 默认值")


def _xml_option(path: Path, name: str) -> str:
    m = re.search(r'<option name="%s" value="([^"]*)"' % name, _read(path))
    if m is None:
        raise AssertionError(f"{path.name} 里没有 option {name}")
    return m.group(1).replace("&quot;", '"').replace("&amp;", "&")


def _xml_name(path: Path) -> str:
    m = re.search(r'<configuration[^>]*\bname="([^"]+)"', _read(path))
    if m is None:
        raise AssertionError(f"{path.name} 里没有 configuration name")
    return m.group(1)


def _csv_columns(problem: str) -> set[str]:
    header = (_REPO_ROOT / "data" / problem / "train.csv").read_text(encoding="utf-8").splitlines()[0]
    return {c.strip() for c in header.split(",")}


def _sh_problems() -> list[tuple[str, str, str]]:
    """example.sh 的 (问题名, csv, 背景词)。

    background 两种形态：run_problem 的内联文本原样返回；
    run_problem_file 的文件引用返回 ``file:<路径>`` 标记（与 bat 侧逐字节比对用）。
    """
    out = []
    for line in _sh_text(_EXAMPLE_SH).splitlines():
        m = _PROBLEM_FILE_RE.match(line)
        if m:
            out.append((m.group(1), m.group(2), "file:" + m.group(3)))
            continue
        m = _PROBLEM_RE.match(line)
        if m:
            out.append((m.group(1), m.group(2), m.group(4)))
    return out


def _bat_problems() -> list[tuple[str, str, str]]:
    """example.bat 的 (问题名, csv, 背景词)。

    background 两种形态：``set BACKGROUND`` + ``call :run_problem``（内联文本，
    取自紧邻上一行的 set）；``call :run_problem_file ... <路径>``（文件引用，
    返回 ``file:<路径>`` 标记）。
    """
    out: list[tuple[str, str, str]] = []
    background: str | None = None
    for line in _read(_EXAMPLE_BAT).splitlines():
        m = _SET_BG_RE.match(line)
        if m:
            background = m.group(1)
            continue
        m = _CALL_FILE_RE.match(line)
        if m:
            out.append((m.group(1), _unquote(m.group(2)), "file:" + _unquote(m.group(3))))
            continue
        m = _CALL_RE.match(line)
        if m:
            assert background is not None, f"call 前没有 set BACKGROUND：{line}"
            out.append((m.group(1), _unquote(m.group(2)), background))
            background = None
    return out


class RunConfigurationParityTest(unittest.TestCase):
    """4 组 MRF 配置：XML（源头）、.sh、.bat 三份参数逐项一致。"""

    def test_the_four_configurations_all_exist_in_all_three_forms(self):
        names = {_xml_name(p) for p in _XMLS}
        self.assertEqual(len(names), 4, "入库的 IDE 运行配置应为 4 个（.idea/runConfigurations/*.xml）")
        self.assertEqual(names, {p.stem for p in _MRF_BATS}, "有配置缺 .bat，或有多余的 .bat")
        self.assertEqual(names, {p.stem for p in _MRF_SHS}, "有配置缺 .sh，或有多余的 .sh")

    def test_ide_configs_use_module_entry_point(self):
        for path in _XMLS:
            with self.subTest(config=path.name):
                self.assertEqual(_xml_option(path, "SCRIPT_NAME"), "drsr_420.cli.main")
                self.assertEqual(_xml_option(path, "MODULE_MODE"), "true")
                self.assertEqual(_xml_option(path, "WORKING_DIRECTORY"), "$PROJECT_DIR$")

    def test_bats_match_their_ide_run_configurations(self):
        for path in _XMLS:
            name = _xml_name(path)
            bat = _REPO_ROOT / f"{name}.bat"
            expected = _flags(_xml_option(path, "PARAMETERS"))
            expected["data_csv"] = _norm_csv(expected["data_csv"])
            actual = _flags(_read(bat), _default_llm_config(_read(bat)))
            actual["data_csv"] = _norm_csv(actual["data_csv"])
            with self.subTest(config=name):
                self.assertEqual(expected, actual, f"{bat.name} 的参数与 {path.name} 不一致")

    def test_bats_match_their_shell_scripts(self):
        for bat in _MRF_BATS:
            sh = _REPO_ROOT / f"{bat.stem}.sh"
            expected = _flags(_sh_text(sh))
            expected["data_csv"] = _norm_csv(expected["data_csv"])
            actual = _flags(_read(bat), _default_llm_config(_read(bat)))
            actual["data_csv"] = _norm_csv(actual["data_csv"])
            with self.subTest(config=bat.stem):
                self.assertEqual(expected, actual, f"{bat.name} 的参数与 {sh.name} 不一致")

    def _canonical_background_text(self, bat: Path) -> str:
        """取某 .bat 实际生效的背景词文本：必须走 --background_file 读规范源。"""
        flags = _flags(_read(bat), _default_llm_config(_read(bat)))
        self.assertNotIn("background", flags,
                         f"{bat.name} 还在用内联 background——背景词应以 "
                         f"backgrounds/*.txt 为规范源，脚本只传 --background_file 路径")
        ref = flags.get("background_file")
        self.assertIsNotNone(ref, f"{bat.name} 缺 --background_file")
        path = _REPO_ROOT / ref
        self.assertTrue(path.is_file(), f"{bat.name} 指向的背景词文件不存在：{ref}")
        return path.read_text(encoding="utf-8").strip()

    def test_background_matches_the_dataset(self):
        """background 描述的列必须真的存在——XML 里曾把 ellipsoid 写成 lambda12+lambda23。"""
        for bat in _MRF_BATS:
            name = bat.stem
            bg = self._canonical_background_text(bat)
            columns = _csv_columns(name)
            with self.subTest(config=name):
                for var in ("lambda12", "lambda23"):
                    self.assertEqual(var in bg, var in columns,
                                     f"{name} 的 background 与数据列不符（实际列：{sorted(columns)}）")
                mode = "shear" if "Shear" in name else "compress"
                self.assertIn(f"{mode} mode", bg, f"{name} 的 background 没写对模式")
                shape = name.rsplit("-", 1)[-1].lower()  # cuboid / ellipsoid
                self.assertIn(shape, bg.lower(), f"{name} 的 background 把粒子形状写错了")

    def test_scripts_point_at_the_canonical_backgrounds_txt(self):
        """--background_file 必须指向 backgrounds/<名>.txt（规范源），三份等价物一致。"""
        for bat in _MRF_BATS:
            name = bat.stem
            expected = f"backgrounds/{name}.txt"
            with self.subTest(config=name):
                self.assertEqual(
                    _flags(_read(bat), _default_llm_config(_read(bat))).get("background_file"),
                    expected, f"{bat.name} 的 --background_file 路径不对")
                self.assertEqual(
                    _flags(_sh_text(_REPO_ROOT / f"{name}.sh")).get("background_file"),
                    expected, f"{name}.sh 的 --background_file 路径不对")
                xml = _REPO_ROOT / ".idea" / "runConfigurations" / f"{name.replace('-', '_')}.xml"
                self.assertEqual(
                    _flags(_xml_option(xml, "PARAMETERS")).get("background_file"),
                    expected, f"{xml.name} 的 --background_file 路径不对")

    def test_example_scripts_reference_the_canonical_backgrounds_too(self):
        """example.sh/bat 里 4 个 MRF 问题必须走 run_problem_file + backgrounds/*.txt。"""
        expected = {b.stem: f"backgrounds/{b.stem}.txt" for b in _MRF_BATS}
        sh, bat = _sh_problems(), _bat_problems()
        sh_by_name = {p[0]: p[2] for p in sh}
        bat_by_name = {p[0]: p[2] for p in bat}
        for name, ref in expected.items():
            with self.subTest(problem=name):
                self.assertEqual(sh_by_name.get(name), "file:" + ref,
                                 f"{name}: example.sh 应以 run_problem_file 引用 {ref}")
                self.assertEqual(bat_by_name.get(name), "file:" + ref,
                                 f"{name}: example.bat 应以 run_problem_file 引用 {ref}")

    def test_llm_config_defaults_match(self):
        sh = _default_llm_config(_sh_text(_EXAMPLE_SH))
        for path in _ALL_BATS:
            with self.subTest(bat=path.name):
                self.assertEqual(_default_llm_config(_read(path)), sh,
                                 f"{path.name} 的默认档案与 example.sh 不一致")


class ExampleScriptParityTest(unittest.TestCase):
    """example.bat 的 18 条调用与 example.sh 逐字节一致。"""

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


class BatchHygieneTest(unittest.TestCase):
    """批处理本身的硬性要求：CRLF + 无 BOM + UTF-8 + 双击可用 + 走模块入口。"""

    def test_files_are_utf8_crlf_without_bom(self):
        for path in _ALL_BATS:
            raw = path.read_bytes()
            with self.subTest(bat=path.name):
                self.assertFalse(raw.startswith(b"\xef\xbb\xbf"),
                                 "UTF-8 BOM 会让 cmd 把首行 @echo off 读成乱码命令")
                raw.decode("utf-8")  # 必须是 UTF-8，脚本内有 chcp 65001
                self.assertIn(b"chcp 65001", raw, "缺 chcp 65001，脚本里的中文会显示成乱码")
                self.assertEqual(raw.count(b"\n"), raw.count(b"\r\n"),
                                 "必须全是 CRLF：cmd 的 call/goto 按字节定位标签，LF-only 会找错标签")
                self.assertTrue(raw.endswith(b"\r\n"), "结尾缺少换行")

    def test_shell_scripts_are_lf(self):
        for path in _ALL_SHS:
            raw = path.read_bytes()
            with self.subTest(sh=path.name):
                self.assertNotIn(b"\r\n", raw, "shell 脚本必须是 LF：CRLF 会让 shebang 与续行失效")
                self.assertTrue(raw.endswith(b"\n"), "结尾缺少换行")

    def test_double_click_works_from_any_directory(self):
        for path in _ALL_BATS:
            with self.subTest(bat=path.name):
                self.assertIn('cd /d "%~dp0"', _read(path),
                              '缺 cd /d "%~dp0"，双击时工作目录不对，data/ 与 config/ 都会找不到')

    def test_no_caret_inside_call_arguments(self):
        """call 会把参数里的脱字符翻倍，含 ^ 的 background 必须走变量传递。"""
        for path in _ALL_BATS:
            for line in _read(path).splitlines():
                if line.lstrip().startswith("call"):
                    self.assertNotIn("^", line,
                                     f"{path.name}: call 参数里的 ^ 会被翻倍：{line}")
        # example.bat 的「走变量」写法必须真的把 background 传下去：每个 call 前都有 set BACKGROUND
        self.assertEqual(len(_bat_problems()), 18)

    def test_module_entry_point_only(self):
        for path in _ALL_BATS:
            text = _read(path)
            with self.subTest(bat=path.name):
                self.assertIn("-m drsr_420.cli.main", text)
                self.assertNotIn("main.py", text, "仓库根已无 main.py，入口必须是 python -m")
                self.assertNotIn("python3", text, "Windows 上没有 python3，只有 python")


if __name__ == "__main__":
    unittest.main()
