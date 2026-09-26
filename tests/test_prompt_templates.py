"""提示词模板回归：所有经 ``str.format()`` 渲染的模板都必须真的能渲染。

背景（真实缺陷）：``prompt_config.residual_analysis_prompt`` 的"输出格式"示例里
含未转义的 JSON 花括号，``str.format()`` 会把 ``{\\n "analysis"`` 当成字段名，
抛 ``KeyError: '\\n    "analysis"'``。这条分支是"无 PromptContext 时的兜底模板"
——也就是说，任何不走 pipeline 的调用方（自定义脚本、llm_api 旧路径）一旦触发
残差分析就会直接崩掉，而且异常发生在 try 块之外，不会被兜住。

护栏方式：用 ``string.Formatter().parse()`` 取出模板里**实际声明**的占位符，
要求它们全部落在调用点允许提供的字段集合内。任何多余的未转义花括号都会解析成
未知字段名 → 测试失败。
"""
from __future__ import annotations

import string
import unittest

from drsr_420.core import prompt_config as pc


#: 模板名 → 该模板调用点允许提供的字段（模板可只用其中一部分）
FORMATTED_TEMPLATES = {
    "head_template": ("dependent", "problem", "independent"),
    "analysis_question_good": ("dependent", "problem"),
    "analysis_question_bad": ("dependent", "problem"),
    "analysis_question_none": ("dependent", "problem", "error", "budget_sentence"),
    "analysis_conversation_template": ("prompt", "sample", "question"),
    "residual_analysis_prompt": ("last_analysis", "residual", "sample"),
    "residual_block_title": ("problem",),
    "idea_item_prefix": ("index", "label"),
}


def _declared_fields(template: str) -> list[str]:
    """模板中真正声明的 ``{field}`` 名（未转义花括号也会被解析出来）。"""
    return [name for _, name, _, _ in string.Formatter().parse(template)
            if name is not None]


class PromptTemplateFormatTest(unittest.TestCase):
    def test_declared_placeholders_are_all_allowed(self):
        for name, allowed in FORMATTED_TEMPLATES.items():
            template = getattr(pc, name, None)
            self.assertIsNotNone(template, f"prompt_config.{name} 不存在")
            fields = _declared_fields(template)
            with self.subTest(template=name):
                unexpected = [f for f in fields if f not in allowed]
                self.assertEqual(
                    unexpected, [],
                    f"prompt_config.{name} 含未转义花括号或未知占位符: {unexpected}",
                )

    def test_all_formatted_templates_render(self):
        for name in FORMATTED_TEMPLATES:
            template = getattr(pc, name)
            fields = _declared_fields(template)
            with self.subTest(template=name):
                rendered = template.format(**{f: f"<<{f}>>" for f in fields})
                for f in fields:
                    self.assertIn(f"<<{f}>>", rendered, f"{name} 未替换 {f}")

    def test_residual_analysis_prompt_keeps_json_shape(self):
        """转义花括号后，渲染结果里出现的仍是单层花括号（JSON 示例仍可读）。"""
        rendered = pc.residual_analysis_prompt.format(
            last_analysis="<<L>>", residual="<<R>>", sample="<<S>>")
        self.assertIn('"output_format": {', rendered)
        self.assertIn('"independent_to_dependent_relationships": {', rendered)
        self.assertNotIn("{{", rendered)
        self.assertNotIn("}}", rendered)
        for value in ("<<L>>", "<<R>>", "<<S>>"):
            self.assertIn(value, rendered)

    def test_guard_detects_unescaped_brace(self):
        """护栏本身有效：未转义花括号一定会被检出（两种形态都覆盖）。"""
        allowed = FORMATTED_TEMPLATES["residual_analysis_prompt"]

        # 形态一：配对的未转义花括号（即修复前的真实形态）→ 被解析成未知字段名
        unescaped = pc.residual_analysis_prompt.replace("{{", "{").replace("}}", "}")
        unexpected = [f for f in _declared_fields(unescaped) if f not in allowed]
        self.assertTrue(unexpected, "护栏失效：未转义花括号未被解析成未知字段名")
        with self.assertRaises(KeyError):
            unescaped.format(**{f: "x" for f in allowed})

        # 形态二：落单的 '{' → parse 阶段直接报错
        with self.assertRaises(ValueError):
            _declared_fields(pc.residual_analysis_prompt + '\n  "extra": {\n')


class AnalysisPromptConstraintsTest(unittest.TestCase):
    """分析提示词的硬约束（初次分析与残差分析共用 ``PromptContext._task_section``）。

    回归（物理量张冠李戴）：实测初始残差分析把 lambda12/lambda23 说成 "shear rate
    ratios"、把压缩（compress）模式的 sigma 说成 "shear stress"，还引了 shear-thinning
    ——把 shear 文献的语境套到了 compress 任务上，与题面对轴长比/压缩应力的定义冲突。
    该断言会随上一次分析结果注入每条采样提示，必须在源头拦住。
    """

    def _ctx(self):
        return pc.PromptContext(
            n_features=2, feature_names=["lambda12", "lambda23"], dependent_name="sigma",
            background=("compress-mode MRF: lambda12 = L1/L2, lambda23 = L2/L3, "
                        "sigma = compressive stress."))

    def test_initial_analysis_forbids_reinterpreting_variables(self):
        text = self._ctx().render_initial_analysis_prompt()
        self.assertIn("Use ONLY the variable meanings given in the task description", text)
        self.assertIn("NOT a shear-rate ratio", text)
        self.assertIn("NOT a shear stress", text)

    def test_residual_analysis_forbids_reinterpreting_variables(self):
        text = self._ctx().render_residual_analysis_prompt("prev", "residual", "sample")
        self.assertIn("Use ONLY the variable meanings given in the task description", text)

    def test_fallback_residual_template_keeps_the_same_constraint(self):
        # 不走 pipeline 的调用方用这条 str.format 模板，约束必须一致
        self.assertIn("Never reinterpret a variable", pc.residual_analysis_prompt)

    def test_analysis_must_not_leak_self_talk(self):
        """回归：实测 analysis 字段里出现过模型独白（"Maybe search literature? ... Keep to
        analysis output."），完全没按 output_format 输出。这类元话语会原样注入采样提示。"""
        for text in (self._ctx().render_initial_analysis_prompt(),
                     self._ctx().render_residual_analysis_prompt("prev", "residual", "sample")):
            self.assertIn("Output ONLY the structured result below", text)
            self.assertIn("no plan or self-talk", text)
            self.assertIn("no tool-intent comments", text)
        self.assertIn("ONLY the structured result below", pc.residual_analysis_prompt)


class ResidualIncrementPromptTest(unittest.TestCase):
    """残差通道必须做"增量"分析：不得复述上一轮结论，且必须给出由残差列派生的字段。

    背景（实测本实验的 residual_analyze.json）：残差提示词与初次分析共用同一份 schema，
    而提示词里唯一符合该 schema 的范例就是上一轮分析文本，模型于是逐字复述——21 轮里
    8 轮与上一轮**完全相同**（最长公共前缀 = 全文长度，相似度 1.000）。复述还会挤掉
    真正有用的内容：注入采样提示时按段落截断到固定字符数。
    """

    def _ctx(self):
        return pc.PromptContext(
            n_features=2, feature_names=["lambda12", "lambda23"], dependent_name="sigma",
            background=("compress-mode MRF: lambda12 = L1/L2, lambda23 = L2/L3, "
                        "sigma = compressive stress."))

    def test_residual_prompt_forbids_restating_previous_conclusions(self):
        text = self._ctx().render_residual_analysis_prompt("prev", "residual", "sample")
        self.assertIn("Do NOT restate", text)
        # 上一轮结论必须标成未经校验的假设（旧措辞 "previous conclusions:" 像既定事实）
        self.assertIn("UNVERIFIED HYPOTHESIS", text)

    def test_residual_prompt_requires_residual_derived_fields(self):
        text = self._ctx().render_residual_analysis_prompt("prev", "residual", "sample")
        for key in ("residual_sign_pattern", "worst_fit_rows", "suggested_structural_change"):
            self.assertIn(key, text)

    def test_rendered_residual_format_block_is_valid_json(self):
        """格式块必须仍是合法 JSON（新增字段的逗号/中括号要拼对）。

        示例块是"外层花括号省略"的片段（模型历史回复也只写 ``"output_format": {...}``），
        故补一对花括号后解析。
        """
        import json
        text = self._ctx().render_residual_analysis_prompt("prev", "residual", "sample")
        block = text[text.index('  "output_format": {'):]
        parsed = json.loads("{" + block + "}")
        analysis = parsed["output_format"]["analysis"]
        self.assertIn("independent_to_dependent_relationships", analysis)
        self.assertIn("inter_relationships_between_independents", analysis)
        self.assertIn("lambda12 ", analysis["residual_sign_pattern"])

    def test_initial_analysis_keeps_its_own_schema(self):
        """初次分析没有残差列，不能带残差专有要求/字段（否则模型要凭空编残差）。"""
        text = self._ctx().render_initial_analysis_prompt()
        self.assertNotIn("residual_sign_pattern", text)
        self.assertNotIn("Do NOT restate", text)

    def test_task_numbering_follows_extra_requirements(self):
        self.assertIn("4.##Output Format##", self._ctx().render_initial_analysis_prompt())
        # 残差通道的编号 = 3 条基础要求 + N 条残差专有要求 + 1；加了"方向词/单调性口径"
        # 那条之后 N=3，故格式块编号顺延到 7。
        self.assertIn("7.##Output Format##",
                      self._ctx().render_residual_analysis_prompt("prev", "res", "sample"))

    def test_direction_wording_rule_is_in_both_prompt_paths(self):
        """方向词自检 + 不得自行重判/重计数单调性（20260926-110809 的实测教训）。

        同一轮里分析把一段下降序列（339.5776 -> 296.6507）写成 "sigma rises ... up
        through 296.6507"，又按自己的口径数出与事实表不一致的反转次数——而提示词规定
        事实表是唯一依据。静态兜底模板与动态 PromptContext 两条路径都必须带这条要求。
        """
        dynamic = self._ctx().render_residual_analysis_prompt("prev", "res", "sample")
        for text in (pc.residual_analysis_prompt, dynamic):
            with self.subTest(path=text[:40]):
                self.assertIn("CHECKED FACTS, not your judgement", text)
                self.assertIn("must not be called a rise", text)
                self.assertIn("Never re-derive, re-count or re-word", text)

    def test_fallback_residual_template_carries_the_same_requirements(self):
        self.assertIn("Do NOT restate", pc.residual_analysis_prompt)
        self.assertIn("residual_sign_pattern", pc.residual_analysis_prompt)


class FlattenAnalysisTest(unittest.TestCase):
    """落盘前剥掉分析模型按 ##Output Format## 回显的 ``output_format`` 外壳。

    实测 20260926-110809 / 094330 等多轮：提示词第 5/6 条**明确要求**这个外壳，
    模型是在照做，而 ``analysis`` 字段是当纯文本存并被注入采样提示的——带外壳只会
    浪费 token、把 schema 噪声喂给采样器。
    """

    #: 真实形态：模型常带 ```json 围栏、且省略最外层花括号（骨架里本来就没有）
    ENVELOPE = (
        "```json\n"
        '"output_format": {\n'
        '  "analysis": {\n'
        '    "independent_to_dependent_relationships": {\n'
        '      "lambda12": [\n'
        '        "sigma is NOT monotone in lambda12: it falls 339.5776 -> 296.6507."\n'
        "      ]\n"
        "    },\n"
        '    "inter_relationships_between_independents": {\n'
        '      "lambda12 vs lambda23": [\n'
        '        "|r| = 0.9996 in log space."\n'
        "      ]\n"
        "    }\n"
        "  }\n"
        "}\n"
        "```\n"
    )

    def test_envelope_is_flattened(self):
        out = pc.flatten_analysis(self.ENVELOPE)
        self.assertNotIn("output_format", out)
        self.assertNotIn("```", out)
        self.assertIn("independent_to_dependent_relationships", out)
        self.assertIn("sigma is NOT monotone in lambda12: it falls 339.5776 -> 296.6507.", out)
        self.assertIn("|r| = 0.9996 in log space.", out)

    def test_outer_braces_variant_also_flattens(self):
        body = self.ENVELOPE.replace("```json\n", "").replace("```\n", "").rstrip()
        with_braces = "{" + body + "}"          # 模型有时会补上最外层花括号
        out = pc.flatten_analysis(with_braces)
        self.assertNotIn("output_format", out)
        self.assertIn("|r| = 0.9996 in log space.", out)

    def test_plain_text_is_returned_unchanged(self):
        text = "sigma is not monotone in lambda12; nothing to flatten here."
        self.assertEqual(pc.flatten_analysis(text), text)

    def test_unparsable_envelope_keeps_the_original(self):
        """解析不出来宁可留外壳，也不能因为"美化"而丢内容。"""
        broken = '```json\n"output_format": {\n  "analysis": { oops not json\n```'
        self.assertEqual(pc.flatten_analysis(broken), broken)

    def test_empty_input_is_safe(self):
        self.assertEqual(pc.flatten_analysis(""), "")
        self.assertEqual(pc.flatten_analysis(None), None)


class DataQuotingInstructionTest(unittest.TestCase):
    """采样指令必须禁止"重造数据行"。

    实测 MRFCompress-Cuboid_20260925-134149：采样器写下 "Data points (inferred)：
    (2,4.46655)=339.58; (2.5,3.5733)=317.23; …"，这些 λ23 全部等于 8.9331/λ12
    （把某一行的 λ23 当成了"恒定乘积"），真值是 8.9331 / 7.4185 / 6.3166 …，真实乘积
    17.87~19.59 并非常数——它随后基于这些伪造点推导"岭上 U 形"。指令里已有
    "data availability"，但没有"不得反推/重算数据行"。
    """

    def _instruction(self):
        ctx = pc.PromptContext(n_features=2, feature_names=["lambda12", "lambda23"],
                               dependent_name="sigma", background="bg")
        return ctx.render_instruction()

    def test_instruction_forbids_reconstructing_data_rows(self):
        for text in (pc.instruction_prompt, self._instruction()):
            self.assertIn("Never present a reconstructed or 'inferred' data row", text)
            self.assertIn("do not replace one independent variable by a product", text)
            self.assertIn("label them as your own derivation", text)

    def test_existing_rules_are_preserved(self):
        text = pc.instruction_prompt
        self.assertIn("Use only params[0], params[1], ... within the available parameter budget", text)
        self.assertIn("Output nothing but the code block", text)


class SamplingSystemPromptTest(unittest.TestCase):
    """采样系统提示里的文献约束：必须是"选用文献时的约束"，不是"必须检索"。"""

    def test_literature_use_is_explicitly_optional(self):
        # 回归：早期写法以 "Literature rules (mandatory): Call search_paper first" 开头，
        # 模型读成硬性检索要求，不需要文献也要先搜一轮（实测 Sampler-0 原话引用了它）。
        text = pc.sampling_system_prompt
        self.assertIn("using literature is optional", text)
        self.assertIn("never search just because of these rules", text)
        self.assertNotIn("Literature rules (mandatory)", text)

    def test_doi_must_come_verbatim_from_search_results(self):
        text = pc.sampling_system_prompt
        self.assertIn("verbatim", text)
        self.assertIn("Never invent, guess, complete, or recall a DOI", text)
        # 回归：规则原先只承认 search_paper 作为 DOI 来源，而 search_kb 同样返回真实的
        # (title, DOI)（实测唯一一次 read_paper 用的就是 search_kb 的返回）。来源必须
        # 同时覆盖两个检索工具，否则按字面把合法调用判成违规。
        self.assertIn("search_paper OR search_kb result", text)
        self.assertIn("never pair a title with a DOI unless that exact pair appears in a tool "
                      "result", text)

    def test_irrelevant_tool_output_must_not_be_cited(self):
        self.assertIn("unrelated to the question, ignore it", pc.sampling_system_prompt)

    def test_literature_ladder_covers_all_three_tools_in_order(self):
        """三级阶梯必须写清顺序与各自适用条件（覆盖 search_kb/search_paper/read_paper）。

        实测缺陷：模型只盯着 search_paper 反复换措辞重搜（54 次 search_paper、
        1 次 search_kb、0 次 read_paper），因为元数据只有标题、判断不了相关性，
        于是永远升不到 read_paper。
        """
        text = pc.sampling_system_prompt
        self.assertIn("search_kb FIRST", text)
        self.assertIn("title alone can NOT establish relevance", text)
        self.assertIn("at most ONE clearly relevant hit", text)
        self.assertIn("Do NOT re-run the SAME search with reworded wording", text)
        # 禁重搜只禁同义改写：Crossref 对措辞敏感，全禁会连"换实质不同的关键词本可命中"
        # 一起堵死（与"文献可选"叠加后模型可能直接不搜），故显式允许一次实质换词。
        self.assertIn("may retry at most ONCE with genuinely different keywords", text)
        # 旧写法 "either read one specific DOI or stop" 语义自相矛盾（没搜到时无 DOI 可读），
        # 已改为"换词重搜失败即停用文献"。
        self.assertNotIn("either read one specific DOI or stop", text)

    def test_tool_call_budget_was_raised(self):
        # 一条完整阶梯（search_kb -> search_paper -> read_paper）本身就要 3 轮
        self.assertIn("at most 4-6 tool calls", pc.sampling_system_prompt)


if __name__ == "__main__":
    unittest.main()
