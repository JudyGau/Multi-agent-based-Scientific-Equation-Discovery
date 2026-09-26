"""
集中管理 DRSR 的 LLM 提示词模板。

这里不再保留旧的 oscillator 专用默认文案，统一对齐到当前共享 spec：
- 外层 prompt 使用 x0/x1/.../y 变量命名；
- 若有 metadata，则在 prompt 中带上物理语义；
- 避免继续向模型暴露 with driving force / col0 / col1 之类的历史模板残留。
"""

import json
import re

# 任务头中使用的占位参数（用于 _do_request 中的 head 文本格式化）
problem_name_in_prompt = 'target relation'
dependent_name_in_prompt = 'y'
independent_name_in_prompt = 'x0 and x1'


# 采样阶段：说明性指令（拼接在代码 prompt 前）
# 示例由 PromptContext.render_instruction 按当前变量名动态渲染，
# 避免模板写死 x0/x1 而实际变量为 x1/x2 时模型照抄出未定义变量（NameError）。
instruction_prompt = (
    "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. "
    "Complete the 'equation' function below, considering the physical meaning and relationships of inputs.\n"
    "STRICT output requirements:\n"
    "1. Wrap your final answer in a ```python code block containing a complete function, e.g.:\n"
    "   ```python\n"
    "   def equation_v1(x, y, params):\n"
    "       return params[0]*x + params[1]*y + params[2]\n"
    "   ```\n"
    "2. Use ONLY the independent variable names listed under 'Variables' — never introduce, rename, or abbreviate them.\n"
    "3. Use only params[0], params[1], ... within the available parameter budget; never index params beyond the last one.\n"
    # 实测（20260925-134149）：采样器把给定数据表"重造"了一遍——它写下
    # "Data points (inferred): (2,4.46655)=339.58; (2.5,3.5733)=317.23; …"，
    # 这些 λ23 全部等于 8.9331/λ12（把某一行的 λ23 当成了"恒定乘积"），而真值是
    # 8.9331 / 7.4185 / 6.3166 …（真实乘积 17.87~19.59，并非常数）。它随后基于
    # 这些伪造的点推导"岭上 U 形"。数据行只能照抄，派生量必须标明是自己的推导。
    "4. Quote data values exactly as they appear in this conversation. Never present a "
    "reconstructed or 'inferred' data row as data — e.g. do not replace one independent "
    "variable by a product or ratio of the others, and do not claim the dataset has a "
    "constant product/ratio unless a given table or measured fact says so. Derived "
    "quantities you compute yourself are allowed, but label them as your own derivation.\n"
    "5. Output nothing but the code block.\n\n"
)

# 采样后经验对话整体模板（包含上下文与追问占位）
analysis_conversation_template = (
    "Here's our previous conversation:\n\n"
    "user: {prompt}\n\n"
    "assistant: {sample}\n\n"
    "user: {question}\n"
)


# 采样后“基于得分的经验总结”三种追问模板
analysis_question_good = (
    "The optimized function skeleton you just answered scored higher. Please summarize useful experience.\n"
    "STRICTLY follow these rules:\n"
    "1. Use the exact phrasing \"when seeking for the mathematical function skeleton that represents {dependent}, I can ...\"\n"
    "2. Summarize ONLY the key success factors\n"
    "3. You need to make your answer as concise as possible\n"
)

analysis_question_bad = (
    "The optimized function skeleton you just answered scored lower. What lessons can you draw from it?\n"
    "STRICTLY follow these rules: \n"
    "1. Use the exact phrasing \"when seeking for the mathematical function skeleton that represents {dependent}, I can ...\"\n"
    "2. Identify ONE crucial improvement point\n"
    "3. You need to make your answer as concise as possible\n"
)

analysis_question_none = (
    "The optimized function skeleton you just answered failed with error: {error}. What lessons can you draw from it?\n"
    "{budget_sentence}"
    "STRICTLY follow these rules:\n"
    "1. Use the exact phrasing \"when seeking for the mathematical function skeleton that represents {dependent}, I need ...\"\n"
    "2. Address the SPECIFIC error: {error}\n"
    "3. Treat this failed sample as a negative example to avoid, not as a target requirement to satisfy\n"
    "4. Identify ONE concrete change that would prevent the next sample from repeating this failure\n"
    "5. You need to make your answer as concise as possible\n"
)

# 经验注入区块标题与条目前缀
ideas_block_title = "\n\n### The following are ideas summarized based on past experiences in solving such problems. ###\n\n"
idea_item_prefix = "idea{index} ({label}):\n"

# 残差分析注入区块标题
#
# 口吻必须明确"未经校验"：这些文本是上一轮 LLM 的自由分析，实测出现过把全局极值
# 说错（"peaks at lambda12=2"，真实最大值在 lambda12=1）、把常数乘积脊线说成
# "≈19–19.6"（7 个点里 4 个 <19）这类错误，而下游把它当既有结论继续引用。
residual_block_title = (
    "\n\n### The following is a model-written analysis from an earlier round. Treat it as an "
    "UNVERIFIED HYPOTHESIS, not as an established fact: check any number or claim in it "
    "against the data before relying on it, and prefer the measured data facts whenever the "
    "two disagree. ###\n\n"
)

# 系统角色提示：角色设定与任务数据分离（采样/分析/残差/解释通用）
system_prompt = (
    "You are a physics-informed scientific equation discovery assistant. "
    "Analyze data and discover mathematical function structures "
    "using physical knowledge from literature when available.\n"
)

# 带工具使用引导的系统提示（用于支持工具调用的采样/解释循环）
sampling_system_prompt = system_prompt + (
    "When you need literature background to judge physical relationships or mechanisms, "
    "you may call the provided tools (search_paper, read_paper, search_kb) "
    "to retrieve relevant references before answering.\n"
    "However, prefer writing the equation skeleton directly: only search literature when the "
    "physical relationship is genuinely uncertain, and make at most 4-6 tool calls in total. "
    "The final answer must be a complete Python function wrapped in a ```python code block, "
    "e.g. '```python\\ndef equation_v1(x, y, params):\\n    return params[0]*x + params[1]\\n```', "
    "using only the independent variable names given in the prompt.\n"
    # 实证约束：实测中模型会自行编造 DOI 与标题配对，read_paper 因此取回完全无关的论文，
    # 无关摘要被当成文献证据注入推理上下文（白烧 token 且误导结论）。
    # 这些是"使用文献时的约束"，不是"必须检索文献"的义务：早期写法以
    # "Literature rules (mandatory): Call search_paper first" 开头，模型把它读成硬性检索
    # 要求，于是明明不需要文献也要先搜一轮（实测 Sampler-0 原话："The mandatory rule says
    # 'Call search_paper first' ... Maybe one search is fine"），白烧 token 且与本段前面的
    # "prefer writing the equation skeleton directly" 自相矛盾。
    # 三级阶梯（覆盖全部三个工具）：实测模型只盯着 search_paper 反复换措辞重搜
    # （单轮实验 54 次 search_paper、1 次 search_kb、0 次 read_paper），因为元数据只有标题、
    # 判断不了相关性，于是永远升不到 read_paper。这里把"先知识库、再检索、最后才读"的顺序
    # 和各自的适用条件写死，并明确"标题判断不了相关性"。
    # 禁重搜也不能一刀切（早期写法是"一次搜不到就停用文献"）：Crossref 关键词检索对措辞
    # 很敏感，全禁会把"换一组实质不同的关键词本可命中"的情况一并堵死，叠加"文献可选"后
    # 模型容易直接不搜——只是把浪费从 token 换成放弃文献证据。故只禁同义改写，允许一次
    # 实质换词（不同物理量/机制）。
    "Literature rules (constraints that apply ONLY IF you choose to use literature -- "
    "using literature is optional and often unnecessary; never search just because of these "
    "rules). If you do use literature, follow this ladder in order and do NOT skip steps:\n"
    "- Step 1 -- physical background / functional form: call search_kb FIRST. It returns real "
    "text excerpts from the local literature knowledge base (already populated for this domain) "
    "and is the cheapest way to get physical grounding. Each hit also carries its real "
    "(title, DOI), so a search_kb hit can be fed straight into Step 3.\n"
    "- Step 2 -- you need an actual DOI: call search_paper. A hit's title alone can NOT establish "
    "relevance; judge it by the returned abstract, and only treat an entry as worth reading when "
    "that abstract clearly addresses THIS problem.\n"
    "- Step 3 -- at most ONE clearly relevant hit: call read_paper with that entry's exact "
    "(title, DOI). A DOI may ONLY be copied verbatim from a search_paper OR search_kb result in "
    "this conversation -- both return real (title, DOI) pairs, and a search_kb hit is the better "
    "choice when it already matches the problem. Never invent, guess, complete, or recall a DOI "
    "from memory, and never pair a title with a DOI unless that exact pair appears in a tool "
    "result. Do not call read_paper for a paper you did not see in search results.\n"
    "- Do NOT re-run the SAME search with reworded wording -- that only burns the budget. You "
    "may retry at most ONCE with genuinely different keywords (a different physical quantity or "
    "mechanism, not a synonym of the same one). If that retry still shows nothing clearly "
    "relevant, stop using literature: say the results were not relevant and move on. Finding "
    "nothing relevant is an acceptable outcome.\n"
    "- Tool output is evidence only if it actually addresses this problem. If a tool result "
    "(or a read_paper 'title mismatch' notice) is unrelated to the question, ignore it and say "
    "so -- do NOT cite it as support for any physical claim.\n"
)

# RAG 文献知识库检索结果注入区块标题
# pipeline 检索到文献上下文后替换 {literature_context} 占位符，检索不到时替换为空串。
literature_block_title = (
    "\n\n### The following are relevant excerpts retrieved from the literature knowledge base. "
    "Use them as physical background when analyzing the data. ###\n\n"
)


# 采样阶段：任务头（追加在发送前）
head_template = (
    "Find the mathematical function skeleton that represents {dependent}, "
    "given data on {independent}. \n"
)


# 残差分析提示模板（包含固定格式与输出要求）
#
# 注意：本模板经 ``str.format()`` 渲染，因此"输出格式"里的 JSON 花括号必须写成
# ``{{`` / ``}}`` 转义。此前未转义，导致 ``.format()`` 抛
# ``KeyError: '\n    "analysis"'``——即"无 prompt_ctx 兜底模板"这条分支从未可用。
residual_analysis_prompt = (
    "You are a data analysis expert.\n"
    "previous conclusions (an UNVERIFIED HYPOTHESIS written by the same model in the previous "
    "round -- it may be wrong, check it against the numbers below):{last_analysis}\n"
    "dataset:{residual}\n"
    "The equation whose residuals are listed above:{sample}\n\n"
    "The independent variables are x0 and x1.\n"
    "The dependent variable is y.\n"
    "The fourth column contains residuals (calculated as observed value - predicted value from the equation).\n"
    "Each row represents a set of independent variables and the corresponding dependent variable and residual.\n\n"
    "Task Requirements:\n\n"
    "1. Please analyze and summarize the influence of the changes in the values of different independent variables on the dependent variable,\n"
    "as well as the possible intrinsic relationships among different independent variables.\n\n"
    "Your response must contain ONLY the structured result below: no reasoning process, no plan or "
    "self-talk, no tool-intent comments, and no text outside the structure.\n\n"
    "2. Use ONLY the variable meanings given in the task description. Never reinterpret a variable as a "
    "different physical quantity (e.g. a geometric ratio is not a rate ratio, and a compressive stress is "
    "not a shear stress), and do not import mechanisms from another mode, geometry, or material system.\n\n"
    # 残差通道专有要求：不得复述上一轮结论，且必须给出由残差列派生的字段
    # （实测 21 轮残差分析里 8 轮与上一轮逐字相同，最长公共前缀 = 全文长度）。
    "3. This round analyzes ONLY the residuals of the equation given above. Do NOT restate, paraphrase "
    "or summarize the previous conclusions: a statement that merely repeats them is a failed answer; "
    "re-derive from these numbers instead, and mark contradicted claims as contradicted.\n\n"
    "4. Derive `residual_sign_pattern` (per independent variable: intervals whose residuals keep the "
    "same sign = systematic misfit; intervals with alternating signs = noise/overfitting), "
    "`worst_fit_rows` (rows with the largest |residual|, with sign and magnitude) and "
    "`suggested_structural_change` (ONE concrete skeleton change justified by the pattern). Quote only "
    "numbers that appear in the dataset above.\n\n"
    # 方向词与单调性口径（20260926-110809 的实测教训）：分析文本把一段下降序列
    # （339.5776 -> 296.6507）写成 "sigma rises ... up through 296.6507"；同一轮它又
    # 按自己的口径数反转次数，与事实表给的不一致——而提示词规定事实表是唯一依据。
    "5. Direction wording and the monotonicity verdict are CHECKED FACTS, not your judgement. "
    "(a) Before writing any sentence containing rises/increases/falls/decreases, re-read the "
    "numbers that follow it in that same sentence and check that the wording matches their "
    "direction (a sequence descending from 339.5776 to 296.6507 must not be called a rise); if "
    "you cannot check it, drop the direction word. (b) Never re-derive, re-count or re-word the "
    "monotonicity verdict or the number of reversals: the fact block above is the single "
    "authority — quote its count and its reversal points, and if your own reading differs, say so "
    "explicitly instead of silently using your own count.\n\n"
    "6.##Output Format##:\n"
    "STRICTLY deliver results in the following structured format:\n\n"
    "  \"output_format\": {{\n"
    "    \"analysis\": {{\n"
    "      \"independent_to_dependent_relationships\": {{\n"
    "        \"x0 \": [\n"
    "          \"Hint: analyze the functional relationship between x0 and y in different intervals\"\n"
    "        ],\n"
    "        \"x1 \": [\n"
    "          \"Hint: analyze the functional relationship between x1 and y in different intervals\"\n"
    "        ]\n"
    "      }},\n"
    "      \"inter_relationships_between_independents\": {{\n"
    "        \"x0 vs x1\": [\n"
    "          \"Hint: analyze the possible functional relationship between x0 and x1 in different intervals. If not, leave blank.\"\n"
    "        ]\n"
    "      }},\n"
    "      \"residual_sign_pattern\": {{\n"
    "        \"x0 \": [\n"
    "          \"Hint: intervals of x0 whose residuals keep the same sign, and those whose signs alternate\"\n"
    "        ]\n"
    "      }},\n"
    "      \"worst_fit_rows\": [\n"
    "        \"Hint: rows with the largest |residual|, with sign and magnitude\"\n"
    "      ],\n"
    "      \"suggested_structural_change\": [\n"
    "        \"Hint: ONE concrete skeleton change justified by the residual pattern\"\n"
    "      ]\n"
    "    }}\n"
    "  }}\n"
)


# ==========================
# 动态渲染：无装饰器版本的上下文类
# ==========================

DEFAULT_BACKGROUND = "The physical properties of this equation are unknown and need to be analyzed based on experience."

def _ensure_feature_names(n, names):
    """确保有 n 个自变量名；缺省则按 x1..xN 生成。"""
    if names is None:
        return [f"x{i+1}" for i in range(n)]
    if len(names) != n:
        raise ValueError(f"feature_names 长度应为 {n}，实际为 {len(names)}")
    return names

def _ind_phrase(names):
    """生成 “x1, x2, and x3” 风格短语。"""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f", and {names[-1]}"

def _pairwise(names):
    """两两组合。"""
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j]))
    return pairs


class PromptContext:
    """提示词渲染上下文（无装饰器版本）。

    用法：
        ctx = PromptContext(n_features=X.shape[1], feature_names=None, dependent_name=None,
                            problem_name=None, background=None)
        head = ctx.render_head()
        instruction = ctx.render_instruction()
        q = ctx.render_analysis_question('Good')
        residual_prompt = ctx.render_residual_analysis_prompt(last_analysis, residual, sample)
    """

    def __init__(
        self,
        n_features,
        feature_names=None,
        dependent_name=None,
        problem_name=None,
        background=None,
        feature_descriptions=None,
        target_description=None,
        max_params=None,
    ):
        self.n_features = n_features
        self.feature_names = feature_names
        self.dependent_name = dependent_name
        self.problem_name = problem_name
        self.background = background
        self.feature_descriptions = feature_descriptions
        self.target_description = target_description
        self.max_params = max_params

    # 规范化后的属性
    @property
    def features(self):
        return _ensure_feature_names(self.n_features, self.feature_names)

    @property
    def dependent(self):
        return self.dependent_name or "y"

    @property
    def problem(self):
        # 匿名实验中不向模型暴露 problem_name。
        return problem_name_in_prompt

    @property
    def background_text(self):
        return (self.background or DEFAULT_BACKGROUND).strip()

    @property
    def normalized_feature_descriptions(self):
        features = self.features
        descriptions = self.feature_descriptions or []
        result = []
        for idx, name in enumerate(features):
            desc = descriptions[idx] if idx < len(descriptions) else None
            result.append((name, str(desc).strip() if desc and str(desc).strip() else None))
        return result

    @property
    def dependent_text(self):
        desc = self.target_description
        if desc and str(desc).strip():
            return f"{self.dependent} ({str(desc).strip()})"
        return self.dependent

    @property
    def max_param_count(self):
        try:
            value = int(self.max_params)
        except Exception:
            return None
        return value if value > 0 else None

    def _feature_phrase(self):
        items = []
        for name, desc in self.normalized_feature_descriptions:
            if desc:
                items.append(f"{name} ({desc})")
            else:
                items.append(name)
        return _ind_phrase(items)

    def _variables_block(self):
        lines = ["- Independents:"]
        for name, desc in self.normalized_feature_descriptions:
            if desc:
                lines.append(f"  - {name}: {desc}")
            else:
                lines.append(f"  - {name}")
        lines.append("- Dependent:")
        if self.target_description and str(self.target_description).strip():
            lines.append(f"  - {self.dependent}: {str(self.target_description).strip()}")
        else:
            lines.append(f"  - {self.dependent}")
        return "\n".join(lines)

    # 渲染方法
    def render_instruction(self):
        feats = self.features
        # 示例随当前变量名动态生成：return params[0]*x1 + params[1]*x2 + params[2]
        example_terms = [f"params[{i}]*{name}" for i, name in enumerate(feats)]
        example_terms.append(f"params[{len(feats)}]")
        example = "return " + " + ".join(example_terms)
        feats_phrase = _ind_phrase(feats)
        return (
            instruction_prompt
            + f"Example: {example}\n\n"
            + "Variables:\n"
            + f"{self._variables_block()}\n"
            + f"Background: {self.background_text}\n"
            + self._data_availability_line()
            + (
                f"Your function must reference the independent variables EXACTLY as {feats_phrase} "
                f"(no renaming, no shorthand such as 'l12' or 'x'), and its signature must be "
                f"def equation_vN({', '.join(feats)}, params).\n"
            )
        )

    def _data_availability_line(self):
        """提示模型训练数据仅包含下列自变量，避免幻觉使用背景描述里未提供数据的变量。"""
        feats = self.features
        return (
            f"Data availability: the training data only provides the following independent "
            f"variable(s): {_ind_phrase(feats)}. Do not use or mention any other variable "
            "that is not listed above.\n"
        )

    def render_head(self):
        return head_template.format(
            dependent=self.dependent_text,
            problem=self.problem,
            independent=self._feature_phrase(),
        )

    def render_analysis_question(self, quality, error=None):
        if quality == "Good":
            return analysis_question_good.format(dependent=self.dependent, problem=self.problem)
        if quality == "Bad":
            return analysis_question_bad.format(dependent=self.dependent, problem=self.problem)
        if quality == "None":
            max_params = self.max_param_count
            if max_params is None:
                budget_sentence = (
                    "Treat this failure as a negative example rather than a requirement to satisfy. "
                    "If the error is about parameter length or indexing, do not solve it by asking for more parameters. "
                    "Instead, reduce parameter usage so the equation fits the evaluator's available parameter budget.\n"
                )
            else:
                max_index = max_params - 1
                budget_sentence = (
                    f"The current evaluator passes exactly {max_params} trainable parameters, "
                    f"indexed from params[0] to params[{max_index}]. "
                    "Treat this failure as a negative example rather than a requirement to satisfy. "
                    "If the error is about parameter length or indexing, do not solve it by asking for more parameters. "
                    f"Instead, rewrite the equation so it stays within params[0]..params[{max_index}] "
                    f"and avoid any explicit minimum-length checks above {max_params}.\n"
                )
            return analysis_question_none.format(
                dependent=self.dependent,
                problem=self.problem,
                error=str(error or ""),
                budget_sentence=budget_sentence,
            )
        raise ValueError(f"unknown quality: {quality}")

    def render_residual_block_title(self):
        return residual_block_title.format(problem=self.problem)

    def _output_format_block(self, residual: bool = False) -> str:
        """构建输出格式块（变量名、因变量、两两组合均动态生成）。

        ``residual=True`` 时追加三个**只能由残差列派生**的字段：残差通道此前与初次
        分析共用同一份"自变量→因变量关系"schema，而提示词里唯一符合该 schema 的范例
        就是上一轮分析文本，模型因此逐字复述（实测 21 轮里 8 轮与上一轮完全相同，
        最长公共前缀 = 全文长度）。schema 本身要求"从残差算出来"的内容，复述旧文本
        就不再是格式正确的答案。
        """
        inds = self.features
        dep = self.dependent

        # 条目之间用逗号连接、末尾不加逗号：旧写法给每项都缀了 ','
        # （最后一项因此多一个尾逗号），示例本身不是合法 JSON，而提示词却要求模型
        # "STRICTLY deliver results in the following structured format"。
        ind_to_dep = ",\n".join([
            f'        "{name} ": [\n'
            f'          "Hint: analyze the functional relationship between {name} and {dep} in different intervals"\n'
            f"        ]"
            for name in inds
        ])

        pairs = _pairwise(inds)
        inter_lines = ",\n".join([
            f'        "{a} vs {b}": [\n'
            f'          "Hint: analyze possible functional relationship between {a} and {b} in different intervals. If not, leave blank."\n'
            f"        ]"
            for a, b in pairs
        ])
        if not inter_lines:
            inter_lines = '        "": []'

        residual_blocks: list[str] = []
        if residual:
            sign_lines = ",\n".join([
                f'        "{name} ": [\n'
                f'          "Hint: for {name}, list the intervals of {name} whose residuals keep the SAME sign '
                f'(systematic misfit there) and the intervals whose residual signs alternate (noise/overfitting)"\n'
                f"        ]"
                for name in inds
            ])
            residual_blocks = [
                '      "residual_sign_pattern": {\n'
                f"{sign_lines}\n"
                '      }',
                '      "worst_fit_rows": [\n'
                '        "Hint: the specific dataset rows with the largest |residual|, each with its '
                'sign and magnitude (numbers copied from the dataset above)"\n'
                '      ]',
                '      "suggested_structural_change": [\n'
                '        "Hint: ONE concrete change to the equation skeleton that the residual pattern '
                'justifies (which term to add, drop or re-shape), with the rows that support it"\n'
                '      ]',
            ]

        fields = [
            '      "independent_to_dependent_relationships": {\n'
            f"{ind_to_dep}\n"
            '      }',
            '      "inter_relationships_between_independents": {\n'
            f"{inter_lines}\n"
            '      }',
            *residual_blocks,
        ]
        return (
            '  "output_format": {\n'
            '    "analysis": {\n'
            f"{',\n'.join(fields)}\n"
            '    }\n'
            '  }\n'
        )

    def _residual_requirements(self) -> tuple[str, ...]:
        """残差通道**独占**的要求（初次分析没有残差列，不能带这些要求）。

        两条实测缺陷的护栏：
        * 复述——上一轮结论是提示词里唯一符合 schema 的范例，模型把它抄一遍就当答案
          （实测本实验 21 轮残差分析里 8 轮与上一轮逐字相同，最长公共前缀 = 全文长度）；
        * 空转——分析只说"某一轮的趋势"，对当前方程哪里错、该怎么改只字未提，注入
          采样提示后不提供任何新信息。故要求必须给出由残差列派生的字段。
        """
        return (
            "This round analyzes ONLY the residuals of the equation given above. Do NOT restate, "
            "paraphrase or summarize the previous conclusions: a statement that merely repeats them "
            "(or repeats your own earlier wording) is a failed answer. If the numbers support an "
            "earlier claim, re-derive it from these numbers; if the residuals contradict it, say so "
            "explicitly and mark that claim as contradicted.",
            "Derive the residual-specific fields from the residual column itself: "
            "`residual_sign_pattern` (for each independent variable, which intervals keep the same "
            "residual sign -- the equation's shape is systematically wrong there -- and which "
            "intervals alternate in sign -- noise or overfitting), `worst_fit_rows` (the specific "
            "rows with the largest |residual|, with sign and magnitude), and "
            "`suggested_structural_change` (ONE concrete change to the skeleton justified by the "
            "pattern). Quote only numbers that appear in the dataset or the code-measured facts.",
            # 方向词与单调性口径（20260926-110809 的实测教训）：分析把一段下降序列
            # （339.5776 -> 296.6507）写成 "sigma rises ... up through 296.6507"，同一轮又
            # 按自己的口径数反转次数、与事实表不一致——而提示词规定事实表是唯一依据。
            "Direction wording and the monotonicity verdict are CHECKED FACTS, not your judgement. "
            "(a) Before writing any sentence containing rises/increases/falls/decreases, re-read the "
            "numbers that follow it in that same sentence and check that the wording matches their "
            "direction (a sequence descending from 339.5776 to 296.6507 must not be called a rise); "
            "if you cannot check it, drop the direction word. (b) Never re-derive, re-count or "
            "re-word the monotonicity verdict or the number of reversals: the fact block above is "
            "the single authority — quote its count and its reversal points, and if your own "
            "reading differs, say so explicitly instead of silently using your own count.",
        )

    def _task_section(self, role_text: str, extra_requirements: tuple[str, ...] = ()) -> str:
        """构建“任务要求 + 输出格式引导”段落（初次分析与残差分析共用）。

        ``extra_requirements`` 是残差通道特有的编号要求（见 :meth:`_residual_requirements`），
        插在 "##Output Format##" 之前；编号顺延，初次分析不传它 → 编号与旧版一致。
        """
        extra = "".join(f"{4 + i}. {req}\n\n" for i, req in enumerate(extra_requirements))
        format_index = 4 + len(extra_requirements)
        return (
            f"{role_text}\n\n"
            "Task Requirements:\n\n"
            "1. Analyze and summarize how changes of each independent variable influence the dependent variable, "
            "and the possible intrinsic relationships among independent variables.\n\n"
            # 实测缺陷：模型会自行改写数据行（把 (2, 8.933) 复述成 (2, 2)）、
            # 把先验当结论、把全局极值说错。数据与代码实测的事实表才是唯一合法出处。
            "2. Quote numbers only from the provided dataset and, when present, from the code-measured "
            "data facts. Do not restate values from memory, re-round them differently, or invent rows "
            "that are not in the dataset. Do not state a global maximum/minimum or a correlation unless "
            "it appears in those facts. If a physical prior or expectation conflicts with the measured "
            "facts, report the conflict explicitly instead of repeating the prior as established fact.\n\n"
            # 实测缺陷：分析阶段会把独白写进 analysis 字段（如 "Maybe search literature? Could
            # search for MR effect compress mode papers quickly, but not required. Keep to
            # analysis output."），完全没按 output_format 输出；这类元话语会原样注入采样
            # 提示，白烧 token 且挤掉真正的分析内容。
            "Output ONLY the structured result below -- no reasoning process, no plan or "
            "self-talk (e.g. 'maybe I should search the literature'), no tool-intent comments, "
            "and no text before or after the structure.\n\n"
            # 实测缺陷：初始残差分析把 lambda12/lambda23 说成 "shear rate ratios"、把压缩模式的
            # sigma 说成 "shear stress"，还引了 shear-thinning —— 把 shear 文献的语境套到了
            # compress 任务上（与题面对轴长比/压缩应力的定义直接冲突）。变量含义必须以题面
            # 与背景为准，不得替换为背景里没出现过的物理量。
            '3. Use ONLY the variable meanings given in the task description and background above. '
            "Never reinterpret a variable as a different physical quantity: for example, a geometric "
            "axis-length ratio is NOT a shear-rate ratio, and a compressive (squeeze-mode) stress is "
            "NOT a shear stress. Do not import mechanisms, regimes, or terminology from another mode, "
            "geometry, or material system that the background does not mention.\n\n"
            f"{extra}"
            f'{format_index}.##Output Format##:\n'
            'STRICTLY deliver results in the following structured format:\n\n'
        )

    def render_initial_analysis_prompt(self) -> str:
        """渲染“初次数据分析”提示模板。

        返回模板包含 {csv_data} 占位符，由 DataAnalyzer.analyze 替换为实际数据内容。
        变量名、因变量、输出格式均按 PromptContext 动态渲染，避免领域/变量名硬编码。
        """
        role_lines = [
            "The independent variables are:",
            *[
                f"- {name}: {desc}" if desc else f"- {name}"
                for name, desc in self.normalized_feature_descriptions
            ],
            "",
            f"The dependent variable is {self.dependent_text}.",
            "Each row represents a set of independent variables and the corresponding dependent variable value.",
            self._data_availability_line().strip(),
        ]
        role_text = "\n".join(role_lines)

        return (
            "csv\n{csv_data}\n"
            "You are a data analysis expert. I have provided a dataset for scientific analysis.\n"
            "{literature_context}"
            f"{self._task_section(role_text)}\n"
            f"{self._output_format_block()}"
        )

    def render_residual_analysis_prompt(self, last_analysis, residual, sample):
        """渲染“残差分析”提示模板。

        与初次分析的区别（都是实测缺陷的护栏）：

        * 上一轮结论**显式标注为未经校验的假设**——它由同一个模型在上一轮自由生成，
          实测含事实错误（把全局极值说错、把常数乘积脊线写成 ``≈19–19.6``），旧措辞
          ``previous conclusions:`` 读起来像既定事实，会一路传到采样提示；
        * 追加残差专有要求（:meth:`_residual_requirements`）与三个只能由残差列派生的
          输出字段（:meth:`_output_format_block`）——旧版与初次分析共用同一份 schema，
          提示词里唯一符合该 schema 的范例就是上一轮分析文本，模型于是逐字复述
          （实测 21 轮里 8 轮与上一轮完全相同）。
        """
        role_lines = [
            "The independent variables are:",
            *[
                f"- {name}: {desc}" if desc else f"- {name}"
                for name, desc in self.normalized_feature_descriptions
            ],
            "",
            f"The dependent variable is {self.dependent_text}.",
            "The last column contains residuals (observed - predicted).",
            self._data_availability_line().strip(),
        ]
        role_text = "\n".join(role_lines)

        return (
            "You are a data analysis expert.\n"
            f"Background: {self.background_text}\n"
            f"previous conclusions (an UNVERIFIED HYPOTHESIS written by the same model in the "
            f"previous round -- it may be wrong, check it against the numbers below):"
            f"{last_analysis}\n"
            f"dataset:{residual}\n"
            f"The equation whose residuals are listed above:{sample}\n\n"
            f"{self._task_section(role_text, self._residual_requirements())}\n"
            f"{self._output_format_block(residual=True)}"
        )


# ── 分析结果落盘前的清洗 ────────────────────────────────────────
#: ```json 围栏行（模型常把结构化答案包在代码块里）。
_FENCE_RE = re.compile(r"^\s*```[A-Za-z]*\s*$")


def flatten_analysis(text: str) -> str:
    """剥掉分析模型按 ##Output Format## 回显的 JSON 外壳，返回扁平化纯文本。

    提示词第 5/6 条**明确要求**输出 ``"output_format": {"analysis": {...}}`` 外壳，
    模型是在照做；但 ``residual_analyze.json`` 的 ``analysis`` 字段是**当纯文本**存、
    并会被注入采样提示（``prompt_injection``）与"上一轮结论"（``residual_analyzer``）
    ——带着外壳只会浪费 token，并把 schema 噪声喂给采样器（实测 20260926-110809 等多
    次运行都原样存了外壳）。

    解析不出来时**原样返回**：宁可留外壳，也不能因为"美化"而丢内容。
    """
    if not text or "output_format" not in text:
        return text
    payload = _parse_analysis_envelope(text)
    if payload is None:
        return text
    rendered = _render_analysis_payload(payload)
    return rendered or text


def _parse_analysis_envelope(text: str):
    """尽力把外壳解析成 ``analysis`` 字典；失败返回 None。

    模型常省略最外层花括号（骨架里就没有），故两种写法都试；中途任何异常都只当
    "没解析出来"，交给调用方原样保留。
    """
    body = "\n".join(line for line in text.splitlines() if not _FENCE_RE.match(line)).strip()
    for candidate in (body, "{" + body + "}"):
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        inner = (data.get("output_format") or {}).get("analysis")
        if isinstance(inner, dict):
            return inner
        if isinstance(data.get("analysis"), dict):   # 模型省掉了 output_format 这一层
            return data["analysis"]
    return None


def _render_analysis_payload(payload: dict) -> str:
    """把 ``analysis`` 结构渲染成"字段名 + 逐条"的纯文本（信息不变，只去外壳）。"""
    lines: list[str] = []
    for key, value in payload.items():
        lines.append(f"### {key} ###")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                lines.append(f"- {sub_key}:")
                for item in _as_items(sub_value):
                    lines.append(f"    - {item}")
        else:
            for item in _as_items(value):
                lines.append(f"- {item}")
    return "\n".join(lines).strip()


def _as_items(value) -> list[str]:
    """把字段值统一成非空字符串列表（标量当单条）。"""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []
