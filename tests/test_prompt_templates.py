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
        self.assertIn("Do NOT re-run a search with reworded wording", text)

    def test_tool_call_budget_was_raised(self):
        # 一条完整阶梯（search_kb -> search_paper -> read_paper）本身就要 3 轮
        self.assertIn("at most 4-6 tool calls", pc.sampling_system_prompt)


if __name__ == "__main__":
    unittest.main()
