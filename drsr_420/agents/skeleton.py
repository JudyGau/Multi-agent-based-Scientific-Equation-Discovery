"""方程骨架提取：把 LLM 的原始回复文本切出可直接执行的函数体。

角色归属
--------
SamplerAgent 的**纯文本预处理**环节：无状态、不调用 LLM、不读配置。

为什么单独一个模块
------------------
模型输出的形态很不稳定（正文里夹解释、用 ``` 代码块、漏写 ``return``、缩进多一层），
"从混合文本里切出可执行体"是采样链路上最容易出错的边界，值得独立测试与演进；
切出来是空串时由 SamplerAgent 决定重采样或丢弃（见 ``MAX_BODY_RETRIES``）。

对外契约
--------
``extract_body(text)`` 恒返回字符串（失败时为空串，绝不抛异常）；
``extract_code_fragment(text)`` 抽不到时返回 ``None``，供上层区分"整段无代码"与"代码为空"。
"""
from __future__ import annotations

import re

#: 骨架提取不到可执行代码时的最大重采样次数（避免无效骨架占用评估与经验配额）。
MAX_BODY_RETRIES = 3


def extract_code_fragment(text: str) -> str | None:
    """从混合文本中抽取可执行代码片段；抽不到时返回 None。

    抽取策略（按优先级）：
    1. ``def`` 开头的行：取其后的连续缩进代码行（函数体，保留缩进）；
    2. ``return`` 开头的行：从该行起收拢后续缩进行/return 行；
    3. 含 ``params[`` 的独立表达式行：补上 ``return`` 前缀（LLM 可能漏写）。
    """
    lines = text.splitlines()

    # 策略 1：def 函数体
    for i, line in enumerate(lines):
        if line.lstrip().startswith('def '):
            kept = []
            for ln in lines[i + 1:]:
                if ln.startswith((' ', '\t')):
                    if ln.strip():
                        kept.append(ln)
                elif not ln.strip():
                    continue  # 空行跳过
                else:
                    break  # 遇到顶层语句（如后续说明文字）停止
            return '\n'.join(kept) if kept else None

    # 策略 2：return 表达式（从第一个 return 行开始）
    for i, line in enumerate(lines):
        if line.lstrip().startswith('return'):
            code_lines = [line]
            for ln in lines[i + 1:]:
                if ln.startswith((' ', '\t')) or not ln.strip():
                    code_lines.append(ln)
                else:
                    break
            return '\n'.join(code_lines)

    # 策略 3：含 params[ 的独立表达式行（LLM 可能漏写 return）
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and 'params[' in stripped:
            return stripped if stripped.startswith('return') else f'return {stripped}'

    return None


#: 只有注释/docstring、没有可执行语句的"骨架"不算骨架（理由见 :func:`has_executable_statement`）。
_DOCSTRING_RE = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')
_COMMENT_RE = re.compile(r"#[^\n]*")
_RETURN_LINE_RE = re.compile(r"(?m)^[ \t]*return\b")


def has_executable_statement(body: str) -> bool:
    """骨架里有没有**可执行语句**——至少要有一个 ``return``。

    为什么单看"非空"不够：实测 ``MRFCompress-Cuboid_20260928-092926`` 的
    ``samples_92/93.json``——那次输出撞上 ``max_tokens`` 上限（thinking=65483 + content=53
    正好 65536），正文只剩提示模板里那段 docstring，``return`` 根本没写出来。原逻辑看到
    ``def`` 之后"有带缩进的内容"就把这几行 docstring 当骨架返回（**非空**）→ 上游
    ``MAX_BODY_RETRIES`` 的重采样**没被触发** → 样本一路走到评估，留下一条
    ``mse/penalty/score`` 全 null 的记录。

    判据只看 ``return``：本项目的骨架不返回预测就没有评估价值，而"有没有 return"是唯一
    不依赖具体写法就能判准的量。注释与 docstring 先剥掉，免得正文里那句
    "return the stress"被当成语句（截断时 docstring 可能没有闭合的三引号、剥离不生效，
    但判据要求 ``return`` 出现在**行首**，误判概率很低）。
    """
    code = _COMMENT_RE.sub(" ", _DOCSTRING_RE.sub(" ", str(body or "")))
    return bool(_RETURN_LINE_RE.search(code))


def extract_body(sample: str) -> str:
    """切出可直接执行的函数体；**没有可执行语句时返回空串**。

    空串让 SamplerAgent 走既有的重采样（``MAX_BODY_RETRIES``），而不是把一段只有 docstring
    的"骨架"送去评估（见 :func:`has_executable_statement` 的实测）。剪切规则本身见
    :func:`_extract_body_raw`。
    """
    body = _extract_body_raw(sample)
    if body and not has_executable_statement(body):
        print("[Skeleton] 骨架里没有可执行语句（只有 docstring/注释，常见于输出被截断）"
              " → 视为空骨架，交由上层重采样")
        return ''
    return body


def _extract_body_raw(sample: str) -> str:
    """
    Extract the function body from a response sample, removing any preceding descriptions
    and the function signature. Preserves indentation.
    ------------------------------------------------------------------------------------------------------------------
    Input example:
    ```
    This is a description...
    def function_name(...):
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    Output example:
    ```
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    增强逻辑：
    - 优先提取 ``` 代码块；
    - 无代码块时，从混合文本中抽取可执行代码片段（def 函数体 / return 表达式 / 含 params 的表达式），
      不再整段丢弃"文字+代码"混合输出；
    - 完全抽不到可执行代码时返回空字符串，由上游决定重采样。
    """
    # 提取 python 代码
    match = re.search(r'```([\s\S]*?)```', sample)
    if match:
        sample = match.group(1).strip()
    else:
        # 无代码块：尝试从混合文本中抽取代码片段
        extracted = extract_code_fragment(sample)
        if extracted is None:
            print("No executable code found in response, returning empty skeleton for resampling.")
            return ''
        sample = extracted

    # 去除LLM回复中的python
    sample = sample.replace('python', '')

    # 检测缺少缩进的return语句，并加上缩进'    '
    if (sample[:6] == 'return'):
        sample = '    ' + sample
        return sample

    # 检测多一个缩进的return语句，并改成一个缩进'    '
    if (sample[:14] == '        return'):
        sample = sample.replace('        ', '    ')
        return sample

    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False

    for lineno, line in enumerate(lines):
        # find the first 'def' program statement in the response
        if (line[:3] == 'def'):
            func_body_lineno = lineno
            find_def_declaration = True
            break

    if find_def_declaration:
        # 统一处理：直接保留函数定义后的原始缩进与内容
        code = ''
        for line in lines[func_body_lineno + 1:]:
            code += line + '\n'
        return code

    return sample
