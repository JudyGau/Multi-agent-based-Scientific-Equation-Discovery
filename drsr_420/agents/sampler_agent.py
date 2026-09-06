"""采样 Agent：调用 LLM 生成方程程序骨架（含提示词构造与空骨架重采样）。

角色：SamplerAgent 是"采样者"，负责把任务头 + 历史经验/残差注入拼成提示词，
通过 ToolCallerAgent 与 LLM 多轮对话拿到候选方程骨架，并过滤无效骨架。

协作：
- 上游：CoordinatorAgent（通过 draw_samples() 获取一批骨架）；
- 下游：ToolCallerAgent（多轮工具调用）→ LLMClient。
"""
from __future__ import annotations

import re

from abc import ABC, abstractmethod

from typing import Collection, Type, Any
import random

from drsr_420.console import LineStreamPrinter, print_block
from drsr_420 import config as config_lib
import json
import os
import traceback
from drsr_420 import prompt_config as pc
from llm import LLMClient

from drsr_420.agents.tool_caller_agent import ToolCallerAgent

# 骨架提取不到可执行代码时的最大重采样次数（避免无效骨架占用评估与经验配额）
_MAX_BODY_RETRIES = 3


class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]
    # self._samples_per_prompt = 4 每一次prompt都生成四个相互独立的回答


class SamplerAgent(LLM):
    """采样 Agent：调用 LLM 生成方程程序骨架。

    提示词构造（指令、任务头、历史经验/残差注入）与 MCP 工具循环都封装在此，
    工具循环委托给 ToolCallerAgent。
    """

    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True,
                 prompt_ctx: pc.PromptContext | None = None,
                 llm_client: LLMClient | None = None) -> None:
        """
        Args:
            batch_inference: Use batch inference when sample equation program skeletons. The batch size equals to the samples_per_prompt.
        """
        super().__init__(samples_per_prompt)

        self._prompt_ctx = prompt_ctx
        self._llm_client = llm_client
        instruction_prompt = (self._prompt_ctx.render_instruction() if self._prompt_ctx else pc.instruction_prompt)
        self._batch_inference = batch_inference
        self._instruction_prompt = instruction_prompt
        self._trim = trim
        # MCP 工具循环组件
        self._tool_caller = ToolCallerAgent(llm_client)
        # 本地文件目录（用于加载经验/残差），由 draw_samples 时设置
        self._base_dir = "."

        ####################################
        # 添加会话ID存储
        self._conversation_ids = {}  # 用于存储每个样本的对话ID

    def draw_samples(self, prompt: str, config: config_lib.Config) -> tuple[list[Any] | list[str], list[Any]] | None:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        # 记录统一结果目录供本地路径引用
        try:
            self._base_dir = config.results_root or "."
        except Exception:
            self._base_dir = "."

        # 统一走本地批量请求（已使用注入的 llm_client）
        return self._draw_samples_local(prompt, config)

    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> tuple[list[Any] | list[str], list[
        Any]] | None:
        # instruction
        prompt = '\n'.join([self._instruction_prompt, prompt])
        while True:
            try:
                all_samples = []
                all_thinking_contents = []
                # response from llm server
                if self._batch_inference:
                    print("运行了_draw_samples_local的_batch_inference分支")
                    content = self._build_request_content(prompt, config)
                    first_responses, thinking_contents = self._tool_caller.complete(
                        content, self._samples_per_prompt)
                    print("成功运行first_responses = ToolCaller.complete")
                    print_block(first_responses)
                    all_samples = list(first_responses)
                    all_thinking_contents = list(thinking_contents)
                else:
                    for _ in range(self._samples_per_prompt):
                        content = self._build_request_content(prompt, config)
                        first_responses, _second_responses = self._tool_caller.complete(content, 1)
                        all_samples.append(first_responses)

                # trim equation program skeleton body from samples
                if self._trim:
                    trimmed_samples = []
                    trimmed_thinking = []
                    dropped = 0
                    for idx, sample in enumerate(all_samples):
                        think = all_thinking_contents[idx] if idx < len(all_thinking_contents) else ''
                        body = _extract_body(sample, config)
                        # 抽不到可执行代码（空骨架）时重采样，避免无效样本占用评估与经验配额
                        for retry in range(1, _MAX_BODY_RETRIES + 1):
                            if body:
                                break
                            print(f"[Sampler] 第 {idx + 1} 个样本骨架为空，重采样（第 {retry}/{_MAX_BODY_RETRIES} 次）")
                            content = self._build_request_content(prompt, config)
                            resp, resp_think = self._tool_caller.complete(content, 1)
                            print_block(f"[Sampler] 重采样原始响应: {resp}")
                            body = _extract_body(resp, config)
                            think = resp_think
                        if not body:
                            # 重采样后仍无有效代码：直接丢弃该样本，不进入评估队列
                            dropped += 1
                            print(f"[Sampler] 第 {idx + 1} 个样本重采样后骨架仍为空，丢弃（不进入评估）")
                            continue
                        trimmed_samples.append(body)
                        trimmed_thinking.append(think)
                    if dropped:
                        print(f"[Sampler] 本轮共丢弃 {dropped} 个无效骨架样本")
                    all_samples = trimmed_samples
                    all_thinking_contents = trimmed_thinking

                return all_samples, all_thinking_contents
            except Exception:
                print(Exception)
                continue

    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        all_samples = []
        prompt = '\n'.join([self._instruction_prompt, prompt])
        for _ in range(self._samples_per_prompt):
            try:
                stream = LineStreamPrinter()
                shown = 0
                think_label_printed = False
                content_label_printed = False

                def _on_delta(chunk):
                    nonlocal shown, think_label_printed, content_label_printed
                    reasoning = chunk.get('reasoning_content') or ''
                    content = chunk.get('content') or ''
                    text = reasoning + content
                    if len(text) > shown:
                        if shown < len(reasoning) and not think_label_printed:
                            stream.write("[思考]\n")
                            think_label_printed = True
                        elif not content_label_printed:
                            stream.write_line("[正文]")
                            content_label_printed = True
                        stream.write(text[shown:])
                        shown = len(text)

                resp = self._llm_client.chat([{"role": "user", "content": prompt}], on_delta=_on_delta)
                stream.flush()
                print("\n====================================================\n")
                # 兜底：content 为空时回退到 reasoning，避免模型只思考不输出正文时骨架丢失
                response = resp.get('content', '') or resp.get('reasoning_content', '')
                if self._trim:
                    response = _extract_body(response, config)
                all_samples.append(response)
            except Exception:
                all_samples.append("")
        return all_samples

    def _build_request_content(self, content: str, config: config_lib.Config | None = None) -> str:
        """构造最终发送给 LLM 的内容：任务头 + 历史经验/残差注入。

        经验注入规则（超参数可由 Config.experience_injection 覆盖，缺省用默认值）：
        - None（失败教训）：始终注入，最多 max_per_category['None'] 条（默认 3）；
        - Good / Bad：各自以 optional_category_probability 概率参与，最多 max_per_category 条（默认 2）；
        - 样本进度超过 freshness_threshold 后，只注入 sample_order 在
          [current*freshness_window_ratio, current] 范围内的经验（新鲜度窗口）；
        - Good 按 score 降序（最成功优先），Bad 按 score 升序（最差教训优先），None 按时间序。
        """
        content = content.strip('\n').strip()

        # 经验注入超参数（Config.experience_injection 可覆盖，缺省用默认值）
        exp_cfg = getattr(config, "experience_injection", None)
        optional_category_probability = exp_cfg.optional_category_probability if exp_cfg else 0.5
        category_max_samples = exp_cfg.max_per_category if exp_cfg else {"None": 3, "Good": 2, "Bad": 2}
        freshness_threshold = exp_cfg.freshness_threshold if exp_cfg else 50
        freshness_ratio = exp_cfg.freshness_window_ratio if exp_cfg else 0.7
        max_analysis_chars = exp_cfg.max_analysis_chars if exp_cfg else 500
        inject_residual_probability = exp_cfg.inject_residual_probability if exp_cfg else 0.5

        # 尝试加载经验数据
        try:
            # 当前样本进度 = 各类别中最大的 sample_order（而不是各类别条数之和；
            # 多轮累计后条数总和会远大于真实样本序号，导致新鲜度窗口把所有经验过滤掉）。
            current_sample_order = 0

            experience_file = os.path.join(getattr(self, "_base_dir", "."), "experiences.json")

            if os.path.exists(experience_file):
                with open(experience_file, "r", encoding="utf-8") as f:
                    experiences = json.load(f)

                for category in ("None", "Good", "Bad"):
                    for exp in experiences.get(category, []):
                        order = exp.get("sample_order", 0)
                        if isinstance(order, (int, float)):
                            current_sample_order = max(current_sample_order, int(order))

                # 按类别筛选 + 排序 + 截断。
                filtered_experiences = {"None": [], "Good": [], "Bad": []}
                for category in ("None", "Good", "Bad"):
                    category_exps = experiences.get(category) or []
                    if not category_exps:
                        continue

                    # None 类经验始终注入；Good / Bad 先按概率决定是否注入。
                    if category != "None" and random.random() >= optional_category_probability:
                        continue

                    # 新鲜度窗口：样本进度超过阈值后，只保留近期经验。
                    if current_sample_order > freshness_threshold:
                        min_order = current_sample_order * freshness_ratio
                        category_exps = [
                            exp for exp in category_exps
                            if isinstance(exp.get("sample_order"), (int, float))
                            and min_order <= exp["sample_order"] <= current_sample_order
                        ]
                    if not category_exps:
                        continue

                    # 排序：Good 取最成功（score 降序），Bad 取最值得借鉴（score 升序），None 保持时间序。
                    if category == "Good":
                        category_exps = sorted(
                            category_exps,
                            key=lambda e: e.get("score") if isinstance(e.get("score"), (int, float)) else float('-inf'),
                            reverse=True,
                        )
                    elif category == "Bad":
                        category_exps = sorted(
                            category_exps,
                            key=lambda e: e.get("score") if isinstance(e.get("score"), (int, float)) else float('inf'),
                        )

                    # 截断到每类条数上限（原先 category_max_samples 定义了却未使用，导致 None 类被全部注入）。
                    filtered_experiences[category] = category_exps[:category_max_samples.get(category, 2)]

                # 合并所有类别的经验
                all_selected_experiences = []
                for category, exps in filtered_experiences.items():
                    for exp in exps:
                        experience_entry = {
                            "type": category,
                            "analysis": exp.get("analysis", ""),
                            "sample_order": exp.get("sample_order", "unknown"),
                        }

                        # 对于 None 类别，添加错误信息（如果有）
                        if category == "None" and "error" in exp:
                            error_msg = exp["error"]
                            # 移除特定错误信息（如果需要）
                            if error_msg == "Execution Error: too many values to unpack (expected 5)":
                                error_msg = ""

                            if error_msg:
                                experience_entry["error"] = error_msg

                        all_selected_experiences.append(experience_entry)

                # 如果有经验可用，构建经验提示
                if all_selected_experiences:
                    experience_prompt = pc.ideas_block_title

                    # 为每个经验分配编号，并标注类别（成功经验/待改进/失败教训），
                    # 帮助模型区分“要复制的成功因子”与“要避免的失败”。
                    label_map = {"Good": "successful experience", "Bad": "needs improvement", "None": "failure lesson"}
                    for i, exp in enumerate(all_selected_experiences, 1):
                        label = label_map.get(exp["type"], exp["type"])
                        experience_prompt += pc.idea_item_prefix.format(index=i, label=label)
                        print("=================================sample_order: ==================================\n", exp['sample_order'])

                        # 限制经验分析文本的最大字符数
                        analysis_text = exp["analysis"] if exp.get("analysis") else ""
                        if len(analysis_text) > max_analysis_chars:
                            analysis_text = analysis_text[:max_analysis_chars] + "..."
                        experience_prompt += analysis_text

                        experience_prompt += "\n---\n\n"

                    # 若包含失败经验，追加参数预算提示，避免模型为修复越界而要求更多参数
                    if any(exp.get("type") == "None" for exp in all_selected_experiences):
                        max_params = (
                            self._prompt_ctx.max_param_count
                            if hasattr(self, "_prompt_ctx") and self._prompt_ctx is not None else None
                        )
                        if max_params is not None:
                            experience_prompt += (
                                f"Note: the evaluator passes exactly {max_params} trainable parameters "
                                f"(params[0]..params[{max_params - 1}]). "
                                "Keep every equation within this budget; do not request more parameters.\n"
                            )

                    # 将经验添加到原始内容中
                    content = experience_prompt + "\n\n" + content

            # 有 p 的几率进入以下代码（注入最新残差分析）：
            p = inject_residual_probability  # 残差分析注入概率（Config.experience_injection.inject_residual_probability 可覆盖）

            if random.random() < p and os.path.exists(experience_file):
                print("use residual_analyze: True")

                residual_file = os.path.join(getattr(self, "_base_dir", "."), "residual_analyze.json")
                if os.path.exists(residual_file):
                    with open(residual_file, "r", encoding="utf-8") as f:
                        experiences = json.load(f)

                    # 提取最后一条信息
                    if experiences:
                        last_experience = experiences[-1]
                        last_analysis = last_experience.get("analysis", "")
                        last_sample_order = last_experience.get("sample_order", "unknown")
                        last_equation = last_experience.get("equation", "")
                        if last_equation is not None:
                            # 构建提示
                            experience_prompt = (
                                self._prompt_ctx.render_residual_block_title()
                                if hasattr(self, "_prompt_ctx") and self._prompt_ctx is not None
                                else pc.residual_block_title.format(problem=pc.problem_name_in_prompt)
                            )
                            if len(last_analysis) > 2000:
                                last_analysis = last_analysis[:2000] + "..."
                            experience_prompt += last_analysis
                            print("=================================sample_order: ==================================\n", last_sample_order)
                            # 将经验添加到原始内容中
                            content = experience_prompt + "\n\n" + content
                        else:
                            # 构建提示
                            experience_prompt = (
                                self._prompt_ctx.render_residual_block_title()
                                if hasattr(self, "_prompt_ctx") and self._prompt_ctx is not None
                                else pc.residual_block_title.format(problem=pc.problem_name_in_prompt)
                            )
                            if isinstance(last_analysis, list):
                                last_analysis = last_analysis[0] if last_analysis else ""
                            if len(last_analysis) > 2000:
                                last_analysis = last_analysis[:2000] + "..."
                            experience_prompt += last_analysis

                            # 将经验添加到原始内容中
                            content = experience_prompt + "\n\n" + content

        except Exception as e:
            print(f"加载经验数据时出错: {str(e)}")
            print("Error details:")
            traceback.print_exc()  # 输出详细的错误堆栈信息

        # 添加任务头
        if hasattr(self, "_prompt_ctx") and self._prompt_ctx is not None:
            head = self._prompt_ctx.render_head()
        else:
            head = pc.head_template.format(
                dependent=pc.dependent_name_in_prompt,
                problem=pc.problem_name_in_prompt,
                independent=pc.independent_name_in_prompt,
            )
        content = head + '\n' + content
        print_block("========================最终输入给大模型的content========================\n")
        print_block(content)
        return content


def _extract_code_fragment(text: str) -> str | None:
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


def _extract_body(sample: str, config: config_lib.Config) -> str:
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
      不再整段丢弃“文字+代码”混合输出；
    - 完全抽不到可执行代码时返回空字符串，由上游决定重采样。
    """
    # 提取 python 代码
    match = re.search(r'```([\s\S]*?)```', sample)
    if match:
        sample = match.group(1).strip()
    else:
        # 无代码块：尝试从混合文本中抽取代码片段
        extracted = _extract_code_fragment(sample)
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


# 兼容别名：旧模块名 drsr_420.sampler.Sampler 指向本类
Sampler = SamplerAgent
