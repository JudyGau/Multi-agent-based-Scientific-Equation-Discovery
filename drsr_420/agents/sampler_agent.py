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

from drsr_420.console import StreamDeltaPrinter, print_block
from drsr_420 import config as config_lib
import json
import os
import traceback
from drsr_420 import prompt_config as pc
from llm import LLMClient

from drsr_420.agents.tool_caller_agent import ToolCallerAgent
from drsr_420.agents.base import THREAD_PER_SAMPLER, AgentSpec, BaseAgent

# 骨架提取不到可执行代码时的最大重采样次数（避免无效骨架占用评估与经验配额）
_MAX_BODY_RETRIES = 3
# 单次 draw_samples 采样的最大尝试次数（异常时有界重试，避免 while True 死循环）
_MAX_SAMPLE_ATTEMPTS = 5


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


class SamplerAgent(LLM, BaseAgent):
    """采样 Agent：调用 LLM 生成方程程序骨架。

    提示词构造（指令、任务头、历史经验/残差注入）与 MCP 工具循环都封装在此，
    工具循环委托给 ToolCallerAgent。
    """

    SPEC = AgentSpec(
        key="sampler",
        role="采样者",
        mission="拼提示词（指令+任务头+经验/残差注入）生成方程骨架，空骨架自动重采样",
        entrypoints=("draw_samples",),
        upstream=("coordinator",),
        downstream=("tool_caller",),
        consumes=("prompt.code: str", "config: config_lib.Config"),
        produces=("samples: list[str]", "thinking_contents: list[str]"),
        artifacts=(),
        thread_model=THREAD_PER_SAMPLER,
        llm_task="sampling",
        notes="继承 LLM 抽象基类；抽不到可执行代码时最多重采样 _MAX_BODY_RETRIES 次。",
    )

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
        # 经验/残差 JSON 内存缓存：{path: ..., mtime: ..., data: ...}
        # 跨轮采样复用，避免每次构造提示词都重读磁盘（大批量时是 IO 热点）。
        # CoordinatorAgent 以原子写（os.replace）更新这些文件，mtime 变化即失效重读。
        self._json_cache: dict = {}

        ####################################
        # 添加会话ID存储
        self._conversation_ids = {}  # 用于存储每个样本的对话ID

    def _load_json_cached(self, path: str):
        """读取 JSON 并按 mtime 缓存；文件不存在或解析失败返回 None。

        跨轮采样复用同一份内存副本，仅在文件被原子替换（mtime 变化）时重新读盘，
        兼顾正确性与 IO 开销。SamplerAgent 实例随 CoordinatorAgent 生命周期复用，
        因此缓存覆盖整个采样线程的连续轮次。
        """
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            # 文件不存在或不可访问：清掉可能的旧缓存并返回 None
            if self._json_cache.get('path') == path:
                self._json_cache.clear()
            return None
        cache = self._json_cache
        if cache.get('path') == path and cache.get('mtime') == mtime:
            return cache.get('data')
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        self._json_cache = {'path': path, 'mtime': mtime, 'data': data}
        return data

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
        # 有界重试：原先 `while True: ... except: continue` 在异常时会无限重试，
        # 且 `print(Exception)` 打印的是类对象而非异常实例，无调试价值。
        # 改为最多 _MAX_SAMPLE_ATTEMPTS 次尝试，耗尽后返回空列表交由上层处理，
        # 避免单次采样故障拖死整个采样线程。
        for _attempt in range(1, _MAX_SAMPLE_ATTEMPTS + 1):
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
                        responses, _thinks = self._tool_caller.complete(content, 1)
                        all_samples.append(responses[0])

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
                            resp_list, think_list = self._tool_caller.complete(content, 1)
                            resp, resp_think = resp_list[0], think_list[0]
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
            except Exception as e:
                # 打印真实异常实例与堆栈，便于定位；有界重试避免死循环
                print(f"[Sampler] 采样第 {_attempt}/{_MAX_SAMPLE_ATTEMPTS} 次失败: {e!r}")
                traceback.print_exc()
                continue
        print(f"[Sampler] 采样连续 {_MAX_SAMPLE_ATTEMPTS} 次失败，返回空结果（本轮跳过）")
        return [], []

    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        all_samples = []
        prompt = '\n'.join([self._instruction_prompt, prompt])
        for _ in range(self._samples_per_prompt):
            try:
                printer = StreamDeltaPrinter()
                resp = self._llm_client.chat([{"role": "user", "content": prompt}], on_delta=printer.on_delta)
                printer.flush()
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

        编排顺序（前置注入，最终顺序为 head + residual + experience + 原始 content）：
        1. _inject_experiences：从 experiences.json 选经验条目拼块前置注入；
        2. _inject_residual：按概率注入最近一条残差分析；
        3. 任务头前置。

        经验注入规则（超参数由 Config.experience_injection 覆盖，缺省用默认值）：
        - None（失败教训）：始终注入，最多 max_per_category['None'] 条（默认 3）；
        - Good / Bad：各自以 optional_category_probability 概率参与，最多 max_per_category 条（默认 2）；
        - 样本进度超过 freshness_threshold 后，只注入 sample_order 在
          [current*freshness_window_ratio, current] 范围内的经验（新鲜度窗口）；
        - Good 按 score 降序（最成功优先），Bad 按 score 升序（最差教训优先），None 按时间序。
        """
        content = content.strip('\n').strip()
        exp_cfg = getattr(config, "experience_injection", None)
        try:
            content = self._inject_experiences(content, exp_cfg)
            inject_residual_probability = (
                exp_cfg.inject_residual_probability if exp_cfg else 0.5)
            content = self._inject_residual(content, inject_residual_probability)
        except Exception as e:
            print(f"加载经验数据时出错: {str(e)}")
            print("Error details:")
            traceback.print_exc()

        content = self._render_head() + '\n' + content
        print_block("========================最终输入给大模型的content========================\n")
        print_block(content)
        return content

    def _render_head(self) -> str:
        """任务头：有 PromptContext 时用动态渲染，否则用默认模板。"""
        if self._prompt_ctx is not None:
            return self._prompt_ctx.render_head()
        return pc.head_template.format(
            dependent=pc.dependent_name_in_prompt,
            problem=pc.problem_name_in_prompt,
            independent=pc.independent_name_in_prompt,
        )

    def _inject_experiences(self, content: str, exp_cfg) -> str:
        """从 experiences.json 选经验条目拼块，前置注入 content；无经验文件则原样返回。"""
        experience_file = os.path.join(getattr(self, "_base_dir", "."), "experiences.json")
        experiences = self._load_json_cached(experience_file)
        if experiences is None:
            return content

        # 经验注入超参数（Config.experience_injection 可覆盖，缺省用默认值）
        optional_category_probability = exp_cfg.optional_category_probability if exp_cfg else 0.5
        category_max_samples = exp_cfg.max_per_category if exp_cfg else {"None": 3, "Good": 2, "Bad": 2}
        freshness_threshold = exp_cfg.freshness_threshold if exp_cfg else 50
        freshness_ratio = exp_cfg.freshness_window_ratio if exp_cfg else 0.7
        max_analysis_chars = exp_cfg.max_analysis_chars if exp_cfg else 500

        # 当前样本进度 = 各类别中最大的 sample_order（而非条数之和；
        # 多轮累计后条数总和远大于真实样本序号，会使新鲜度窗口把所有经验过滤掉）
        current_sample_order = 0
        for category in ("None", "Good", "Bad"):
            for exp in experiences.get(category, []):
                order = exp.get("sample_order", 0)
                if isinstance(order, (int, float)):
                    current_sample_order = max(current_sample_order, int(order))

        selected = self._select_experiences(
            experiences, current_sample_order,
            optional_category_probability, category_max_samples,
            freshness_threshold, freshness_ratio)
        if not selected:
            return content

        experience_prompt = self._build_experience_prompt(selected, max_analysis_chars)
        if experience_prompt:
            print_block(f"[经验] 注入 {len(selected)} 条经验（进度 sample_order={current_sample_order}）")
            content = experience_prompt + "\n\n" + content
        return content

    def _select_experiences(self, experiences: dict, current_sample_order: int,
                            optional_category_probability: float,
                            category_max_samples: dict,
                            freshness_threshold: int,
                            freshness_ratio: float) -> list:
        """按类别筛选 + 排序 + 截断，返回扁平化经验条目列表。

        每条含 type/analysis/sample_order；None 类额外附 error（过滤掉指定噪声错误）。
        """
        selected = []
        for category in ("None", "Good", "Bad"):
            category_exps = experiences.get(category) or []
            if not category_exps:
                continue
            # None 类始终注入；Good / Bad 先按概率决定是否注入
            if category != "None" and random.random() >= optional_category_probability:
                continue
            # 新鲜度窗口：样本进度超过阈值后，只保留近期经验
            if current_sample_order > freshness_threshold:
                min_order = current_sample_order * freshness_ratio
                category_exps = [
                    exp for exp in category_exps
                    if isinstance(exp.get("sample_order"), (int, float))
                    and min_order <= exp["sample_order"] <= current_sample_order
                ]
            if not category_exps:
                continue
            # 排序：Good 取最成功（score 降序），Bad 取最值得借鉴（score 升序），None 保持时间序
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
            # 截断到每类条数上限
            category_exps = category_exps[:category_max_samples.get(category, 2)]

            for exp in category_exps:
                entry = {
                    "type": category,
                    "analysis": exp.get("analysis", ""),
                    "sample_order": exp.get("sample_order", "unknown"),
                }
                # None 类附加错误信息（过滤特定噪声错误）
                if category == "None" and "error" in exp:
                    error_msg = exp["error"]
                    if error_msg == "Execution Error: too many values to unpack (expected 5)":
                        error_msg = ""
                    if error_msg:
                        entry["error"] = error_msg
                selected.append(entry)
        return selected

    def _build_experience_prompt(self, selected: list, max_analysis_chars: int) -> str:
        """把选中的经验条目拼成提示词块（含编号、类别标签、参数预算提示）。"""
        prompt = pc.ideas_block_title
        # 为每个经验分配编号并标注类别（成功经验/待改进/失败教训），
        # 帮助模型区分"要复制的成功因子"与"要避免的失败"
        label_map = {"Good": "successful experience", "Bad": "needs improvement", "None": "failure lesson"}
        for i, exp in enumerate(selected, 1):
            label = label_map.get(exp["type"], exp["type"])
            prompt += pc.idea_item_prefix.format(index=i, label=label)
            analysis_text = exp["analysis"] if exp.get("analysis") else ""
            if len(analysis_text) > max_analysis_chars:
                analysis_text = analysis_text[:max_analysis_chars] + "..."
            prompt += analysis_text
            prompt += "\n---\n\n"
        # 若包含失败经验，追加参数预算提示，避免模型为修复越界而要求更多参数
        if any(exp.get("type") == "None" for exp in selected):
            max_params = (
                self._prompt_ctx.max_param_count
                if self._prompt_ctx is not None else None
            )
            if max_params is not None:
                prompt += (
                    f"Note: the evaluator passes exactly {max_params} trainable parameters "
                    f"(params[0]..params[{max_params - 1}]). "
                    "Keep every equation within this budget; do not request more parameters.\n"
                )
        return prompt

    def _inject_residual(self, content: str, inject_residual_probability: float) -> str:
        """以 inject_residual_probability 概率注入最近一条残差分析，前置 content。

        合并原先 if last_equation is not None / else 两个几乎重复的分支：统一处理
        analysis 为 list 的历史格式（取首条），并截断到 2000 字符。
        """
        if random.random() >= inject_residual_probability:
            return content
        # 仅当经验文件存在（已进入采样循环）时才注入残差，避免首轮空注入
        experience_file = os.path.join(getattr(self, "_base_dir", "."), "experiences.json")
        if not os.path.exists(experience_file):
            return content

        residual_file = os.path.join(getattr(self, "_base_dir", "."), "residual_analyze.json")
        residual_data = self._load_json_cached(residual_file)
        if not residual_data:
            return content

        last = residual_data[-1]
        last_analysis = last.get("analysis", "")
        # analysis 可能是 list（初始数据记录的历史格式），取首条
        if isinstance(last_analysis, list):
            last_analysis = last_analysis[0] if last_analysis else ""
        if len(last_analysis) > 2000:
            last_analysis = last_analysis[:2000] + "..."

        block_title = (
            self._prompt_ctx.render_residual_block_title()
            if self._prompt_ctx is not None
            else pc.residual_block_title.format(problem=pc.problem_name_in_prompt)
        )
        print_block(f"[残差] 注入最近残差分析（sample_order={last.get('sample_order', 'unknown')}）")
        return block_title + last_analysis + "\n\n" + content


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
