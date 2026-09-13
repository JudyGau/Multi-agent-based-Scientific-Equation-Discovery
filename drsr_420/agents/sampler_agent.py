"""采样 Agent：调用 LLM 生成方程程序骨架（含提示词构造与空骨架重采样）。

角色：SamplerAgent 是"采样者"，负责把任务头 + 历史经验/残差注入拼成提示词，
通过 ToolCallerAgent 与 LLM 多轮对话拿到候选方程骨架，并过滤无效骨架。

协作
----
* 上游：CoordinatorAgent（通过 ``draw_samples()`` 获取一批骨架）；
* 下游：ToolCallerAgent（多轮工具调用）→ LLMClient；
* 内部部件（本层内聚，不是 Agent）：

  - :mod:`drsr_420.agents.prompt_injection`：提示词装配（经验/残差注入策略）；
  - :mod:`drsr_420.agents.skeleton`：骨架提取（从混合文本切出可执行函数体）。

因此本模块只保留"编排 + 有界重试"：构造 content → 采样 → 校验骨架 → 必要时重采样。
"""
from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from typing import Any, Collection

from drsr_420.core.console import print_block
from drsr_420.core import config as config_lib
from drsr_420.core import prompt_config as pc
from drsr_420.llm import LLMClient

from drsr_420.agents.prompt_injection import PromptInjector
from drsr_420.agents.skeleton import MAX_BODY_RETRIES, extract_body
from drsr_420.agents.tool_caller_agent import ToolCallerAgent
from drsr_420.agents.base import THREAD_PER_SAMPLER, AgentSpec, BaseAgent

# 单次 draw_samples 采样的最大尝试次数（异常时有界重试，避免 while True 死循环）
_MAX_SAMPLE_ATTEMPTS = 5


class LLM(ABC):
    """采样接口（FunSearch 遗留抽象基类）：一次提示词 → 多条候选续写。"""

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

    提示词构造（指令、任务头、历史经验/残差注入）委托 :class:`PromptInjector`，
    MCP 工具循环委托 :class:`ToolCallerAgent`，骨架校验用 :func:`extract_body`。
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
        notes="继承 LLM 抽象基类；抽不到可执行代码时最多重采样 MAX_BODY_RETRIES 次。",
    )

    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True,
                 prompt_ctx: pc.PromptContext | None = None,
                 llm_client: LLMClient | None = None) -> None:
        """
        Args:
            batch_inference: Use batch inference when sample equation program skeletons.
                The batch size equals to the samples_per_prompt.
        """
        super().__init__(samples_per_prompt)

        instruction_prompt = (prompt_ctx.render_instruction() if prompt_ctx else pc.instruction_prompt)
        self._prompt_ctx = prompt_ctx
        self._llm_client = llm_client
        self._batch_inference = batch_inference
        self._instruction_prompt = instruction_prompt
        self._trim = trim
        # MCP 工具循环组件
        self._tool_caller = ToolCallerAgent(llm_client)
        # 提示词装配部件：经验/残差注入（base_dir 在每次 draw_samples 时同步为 results_root）
        self._prompt_injector = PromptInjector(prompt_ctx)

    # ------------------------------------------------------------------
    # 采样入口
    # ------------------------------------------------------------------
    def draw_samples(self, prompt: str, config: config_lib.Config) -> tuple[list[Any] | list[str], list[Any]] | None:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        # 记录统一结果目录供经验/残差注入引用
        try:
            self._prompt_injector.base_dir = config.results_root or "."
        except Exception:
            self._prompt_injector.base_dir = "."

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

                if self._trim:
                    all_samples, all_thinking_contents = self._trim_samples(
                        all_samples, all_thinking_contents, prompt, config)

                return all_samples, all_thinking_contents
            except Exception as e:
                # 打印真实异常实例与堆栈，便于定位；有界重试避免死循环
                print(f"[Sampler] 采样第 {_attempt}/{_MAX_SAMPLE_ATTEMPTS} 次失败: {e!r}")
                traceback.print_exc()
                continue
        print(f"[Sampler] 采样连续 {_MAX_SAMPLE_ATTEMPTS} 次失败，返回空结果（本轮跳过）")
        return [], []

    # ------------------------------------------------------------------
    # 骨架校验与重采样
    # ------------------------------------------------------------------
    def _trim_samples(self, all_samples: list, all_thinking_contents: list,
                      prompt: str, config: config_lib.Config) -> tuple[list, list]:
        """裁出每个样本的可执行骨架；空骨架先重采样，仍为空则整条丢弃。

        丢弃而不是留空串：空骨架进入评估只会白白消耗评估与经验配额，
        并在 ``experiences.json`` 里留下没有任何信息量的失败记录。
        """
        trimmed_samples = []
        trimmed_thinking = []
        dropped = 0
        for idx, sample in enumerate(all_samples):
            think = all_thinking_contents[idx] if idx < len(all_thinking_contents) else ''
            body = extract_body(sample)
            for retry in range(1, MAX_BODY_RETRIES + 1):
                if body:
                    break
                print(f"[Sampler] 第 {idx + 1} 个样本骨架为空，重采样（第 {retry}/{MAX_BODY_RETRIES} 次）")
                content = self._build_request_content(prompt, config)
                resp_list, think_list = self._tool_caller.complete(content, 1)
                resp, resp_think = resp_list[0], think_list[0]
                print_block(f"[Sampler] 重采样原始响应: {resp}")
                body = extract_body(resp)
                think = resp_think
            if not body:
                dropped += 1
                print(f"[Sampler] 第 {idx + 1} 个样本重采样后骨架仍为空，丢弃（不进入评估）")
                continue
            trimmed_samples.append(body)
            trimmed_thinking.append(think)
        if dropped:
            print(f"[Sampler] 本轮共丢弃 {dropped} 个无效骨架样本")
        return trimmed_samples, trimmed_thinking

    # ------------------------------------------------------------------
    # 提示词装配（委托 PromptInjector）
    # ------------------------------------------------------------------
    def _build_request_content(self, content: str, config: config_lib.Config | None = None) -> str:
        """构造最终发送给 LLM 的内容：任务头 + 残差 + 经验 + 原始 content。

        具体选择与拼装规则见 :class:`drsr_420.agents.prompt_injection.PromptInjector`。
        """
        return self._prompt_injector.build_request_content(
            content, getattr(config, "experience_injection", None))


# 兼容别名：旧模块名 drsr_420.sampler.Sampler 指向本类
Sampler = SamplerAgent
