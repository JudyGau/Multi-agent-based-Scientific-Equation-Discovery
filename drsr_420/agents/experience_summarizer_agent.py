"""经验总结 Agent：让 LLM 分析方程及其得分/错误，输出思考过程分析与改进建议。

角色：ExperienceSummarizerAgent 是"经验总结者"，把 Good/Bad/None 样本的得分与错误
转成结构化分析文本，供采样阶段注入下一轮提示词。

协作：
- 上游：CoordinatorAgent（传入一批样本及其质量标签）；
- 下游：LLMClient（分析调用），产物写入 experiences.json（由 CoordinatorAgent 持久化）。
"""
from __future__ import annotations

from drsr_420.core.console import StreamDeltaPrinter
from drsr_420.core import prompt_config as pc

from drsr_420.agents.base import THREAD_PER_SAMPLER, AgentSpec, BaseAgent
from drsr_420.agents.messages import ExperienceEntry, check_alignment


class ExperienceSummarizerAgent(BaseAgent):
    """对一批方程样本及其质量标签做 LLM 经验总结。"""

    SPEC = AgentSpec(
        key="experience_summarizer",
        role="经验总结者",
        mission="对 Good/Bad/None 样本逐个做 LLM 分析，产出改进建议",
        entrypoints=("analyze",),
        upstream=("coordinator",),
        downstream=(),
        consumes=("samples: list[str]", "quality_for_sample: list[str]",
                  "error_for_sample: list[str | None]", "prompt"),
        produces=("entries: list[ExperienceEntry]",),
        artifacts=("experiences.json",),   # 由 CoordinatorAgent 落盘
        thread_model=THREAD_PER_SAMPLER,
        llm_task="experience",
    )

    def __init__(self, llm_client, prompt_ctx=None):
        self._llm_client = llm_client
        self._prompt_ctx = prompt_ctx

    def analyze(self, samples, quality_for_sample, error_for_sample, prompt
                ) -> list[ExperienceEntry]:
        """对每个样本构造分析提示并调用 LLM，返回经验条目列表。

        Args:
            samples: 生成的方程样本列表。
            quality_for_sample: 每个样本的质量标签（'Good'/'Bad'/'None'）。
            error_for_sample: 每个样本的评估错误信息（None 类别才有意义）。
            prompt: 原始提示（含 .code 属性或为字符串），用于提供上下文。

        Returns:
            与 ``samples`` 等长的 :class:`ExperienceEntry` 列表，其中
            ``sample`` / ``quality`` / ``error`` / ``analysis`` 已填好；
            归属字段（岛屿、序号、分数）由 CoordinatorAgent 补齐后落盘。
            三个平行列表长度不一致时抛 ValueError——历史实现靠 ``zip``
            静默错配样本与经验。
        """
        check_alignment(samples, quality_for_sample, error_for_sample)

        entries: list[ExperienceEntry] = []
        for i, sample_each in enumerate(samples):
            quality = quality_for_sample[i]
            error = error_for_sample[i]
            if self._prompt_ctx is not None:
                new_question = self._prompt_ctx.render_analysis_question(
                    quality,
                    error if quality == 'None' else None,
                )
            else:
                new_question = self._build_default_question(quality, error)

            analysis_prompt = pc.analysis_conversation_template.format(
                prompt=prompt.code if hasattr(prompt, "code") else prompt,
                sample=sample_each,
                question=new_question,
            )
            try:
                # 流式输出：通过 on_delta 回调实时打印思考内容与正文（[思考]/[正文] 视觉分隔）
                printer = StreamDeltaPrinter()
                resp = self._llm_client.chat([
                    {"role": "system", "content": pc.system_prompt},
                    {"role": "user", "content": analysis_prompt},
                ], on_delta=printer.on_delta)
                printer.flush()
                # 兜底：推理模型可能把完整分析输出在 reasoning_content 而 content 为空
                analysis_result = resp.get('content', '') or resp.get('reasoning_content', '')
            except Exception as e:
                print(f"分析请求发生错误: {str(e)}")
                analysis_result = f"分析请求发生错误: {str(e)}"
            entries.append(ExperienceEntry(
                sample=sample_each, quality=quality, analysis=analysis_result, error=error))
        return entries

    def _build_default_question(self, quality, error) -> str:
        """无 prompt_ctx 时的默认分析问题模板。"""
        if quality == 'Good':
            return pc.analysis_question_good.format(
                dependent=pc.dependent_name_in_prompt,
                problem=pc.problem_name_in_prompt,
            )
        elif quality == 'Bad':
            return pc.analysis_question_bad.format(
                dependent=pc.dependent_name_in_prompt,
                problem=pc.problem_name_in_prompt,
            )
        elif quality == 'None':
            return pc.analysis_question_none.format(
                dependent=pc.dependent_name_in_prompt,
                problem=pc.problem_name_in_prompt,
                error=error,
                budget_sentence=(
                    "Treat this failure as a negative example rather than a requirement to satisfy. "
                    "If the error is about parameter length or indexing, do not solve it by asking for more parameters. "
                    "Instead, reduce parameter usage so the equation fits the evaluator's available parameter budget.\n"
                ),
            )
        return ''


# 兼容别名：旧模块名 drsr_420.experience_summarizer.ExperienceSummarizer 指向本类
ExperienceSummarizer = ExperienceSummarizerAgent
