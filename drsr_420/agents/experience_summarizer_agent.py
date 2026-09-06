"""经验总结 Agent：让 LLM 分析方程及其得分/错误，输出思考过程分析与改进建议。

角色：ExperienceSummarizerAgent 是"经验总结者"，把 Good/Bad/None 样本的得分与错误
转成结构化分析文本，供采样阶段注入下一轮提示词。

协作：
- 上游：CoordinatorAgent（传入一批样本及其质量标签）；
- 下游：LLMClient（分析调用），产物写入 experiences.json（由 CoordinatorAgent 持久化）。
"""
from __future__ import annotations

from drsr_420.console import LineStreamPrinter
from drsr_420 import prompt_config as pc


class ExperienceSummarizerAgent:
    """对一批方程样本及其质量标签做 LLM 经验总结。"""

    def __init__(self, llm_client, prompt_ctx=None):
        self._llm_client = llm_client
        self._prompt_ctx = prompt_ctx

    def analyze(self, samples, quality_for_sample, error_for_sample, prompt) -> list[str]:
        """对每个样本构造分析提示并调用 LLM，返回分析结果列表。

        Args:
            samples: 生成的方程样本列表。
            quality_for_sample: 每个样本的质量标签（'Good'/'Bad'/'None'）。
            error_for_sample: 每个样本的评估错误信息（None 类别才有意义）。
            prompt: 原始提示（含 .code 属性或为字符串），用于提供上下文。
        """
        analysis_results = []
        i = 0
        for sample_each in samples:
            if self._prompt_ctx is not None:
                new_question = self._prompt_ctx.render_analysis_question(
                    quality_for_sample[i],
                    error_for_sample[i] if quality_for_sample[i] == 'None' else None,
                )
            else:
                new_question = self._build_default_question(
                    quality_for_sample[i], error_for_sample[i])

            analysis_prompt = pc.analysis_conversation_template.format(
                prompt=prompt.code if hasattr(prompt, "code") else prompt,
                sample=sample_each,
                question=new_question,
            )
            try:
                # 流式输出：通过 on_delta 回调实时打印思考内容与正文（[思考]/[正文] 视觉分隔）
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

                resp = self._llm_client.chat([
                    {"role": "system", "content": pc.system_prompt},
                    {"role": "user", "content": analysis_prompt},
                ], on_delta=_on_delta)
                stream.flush()
                # 兜底：推理模型可能把完整分析输出在 reasoning_content 而 content 为空
                analysis_result = resp.get('content', '') or resp.get('reasoning_content', '')
                analysis_results.append(analysis_result)
            except Exception as e:
                print(f"分析请求发生错误: {str(e)}")
                analysis_results.append(f"分析请求发生错误: {str(e)}")
            i += 1
        return analysis_results

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
