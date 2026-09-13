"""历史经验与残差分析的提示词注入：把"上一轮学到的教训"拼进这一轮采样提示词。

角色归属
--------
SamplerAgent 的**提示词装配部件**：不调用 LLM、不发起采样，只负责决定
"注入哪些历史内容、注入多少、以什么顺序拼"。

为什么单独一个模块
------------------
注入策略（类目配额、新鲜度窗口、Good/Bad 排序、失败经验附带的参数预算提示、
按概率注入残差）是采样质量的关键旋钮，与"调 LLM 拿骨架"的编排职责正交；
拆开后两者的演进互不干扰，也便于单独测试（见 ``tests/test_agent_behavior.py``）。

注入顺序（最终提示词 = 任务头 + 残差块 + 经验块 + 原始 content）：:

    PromptInjector.build_request_content()
        ├── inject_experiences()   从 experiences.json 选条目拼经验块
        ├── inject_residual()      按概率注入最近一条残差分析
        └── render_head()          任务头（动态 PromptContext 或默认模板）

超参数由 ``Config.experience_injection`` 提供，``None`` 或字段缺失时回落到
``core.config.ExperienceInjectionConfig`` 的默认值（默认值只在该处定义一份）。
"""
from __future__ import annotations

import json
import os
import random
import traceback

from drsr_420.core import config as config_lib
from drsr_420.core import prompt_config as pc
from drsr_420.core.console import print_block


def resolve_policy(exp_cfg) -> config_lib.ExperienceInjectionConfig:
    """把注入超参数归一化为 ``ExperienceInjectionConfig``。

    历史实现对 ``exp_cfg`` 的每个字段做 ``if exp_cfg else <默认值>``——即配置对象
    只要缺一个字段就整体退化为"不注入"（异常被同一层 try 吞掉，连残差注入也一起失效）。
    这里显式归一化：缺字段只回落该字段，其余照常生效。
    """
    defaults = config_lib.ExperienceInjectionConfig()
    if exp_cfg is None:
        return defaults
    if isinstance(exp_cfg, config_lib.ExperienceInjectionConfig):
        return exp_cfg
    return config_lib.ExperienceInjectionConfig(
        optional_category_probability=getattr(
            exp_cfg, "optional_category_probability", defaults.optional_category_probability),
        max_per_category=dict(
            getattr(exp_cfg, "max_per_category", None) or defaults.max_per_category),
        freshness_threshold=getattr(
            exp_cfg, "freshness_threshold", defaults.freshness_threshold),
        freshness_window_ratio=getattr(
            exp_cfg, "freshness_window_ratio", defaults.freshness_window_ratio),
        inject_residual_probability=getattr(
            exp_cfg, "inject_residual_probability", defaults.inject_residual_probability),
        max_analysis_chars=getattr(
            exp_cfg, "max_analysis_chars", defaults.max_analysis_chars),
    )


class PromptInjector:
    """把 ``experiences.json`` / ``residual_analyze.json`` 注入采样提示词。

    Args:
        prompt_ctx: 动态提示词上下文（``core.prompt_config.PromptContext``）；
            为 ``None`` 时使用默认模板。
        base_dir: 产物目录（``Config.results_root``），经验/残差文件从这里读。
    """

    #: 注入的经验块中，每类分析的字符上限兜底值（正常由 policy 提供）。
    _DEFAULT_MAX_ANALYSIS_CHARS = 500

    def __init__(self, prompt_ctx=None, base_dir: str = ".") -> None:
        self.prompt_ctx = prompt_ctx
        self.base_dir = base_dir
        # 经验/残差 JSON 内存缓存：{'path': ..., 'mtime': ..., 'data': ...}。
        # 跨轮采样复用，避免每次构造提示词都重读磁盘（大批量时是 IO 热点）；
        # CoordinatorAgent 以原子写（os.replace）更新这些文件，mtime 变化即失效重读。
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def build_request_content(self, content: str, exp_cfg=None) -> str:
        """构造最终发送给 LLM 的内容：任务头 + 残差 + 经验 + 原始 content。

        经验注入规则（超参数见 ``core.config.ExperienceInjectionConfig``）：

        * ``None``（失败教训）始终注入，最多 ``max_per_category['None']`` 条（默认 3）；
        * Good / Bad 各自以 ``optional_category_probability`` 概率参与；
        * 样本进度超过 ``freshness_threshold`` 后，只注入 sample_order 落在
          ``[current * freshness_window_ratio, current]`` 的经验（新鲜度窗口）；
        * Good 按 score 降序（最成功优先），Bad 按 score 升序（最差教训优先）。
        """
        policy = resolve_policy(exp_cfg)
        content = content.strip('\n').strip()
        try:
            content = self.inject_experiences(content, policy)
            content = self.inject_residual(content, policy.inject_residual_probability)
        except Exception as e:
            print(f"加载经验数据时出错: {str(e)}")
            print("Error details:")
            traceback.print_exc()

        content = self.render_head() + '\n' + content
        print_block("========================最终输入给大模型的content========================\n")
        print_block(content)
        return content

    def render_head(self) -> str:
        """任务头：有 PromptContext 时用动态渲染，否则用默认模板。"""
        if self.prompt_ctx is not None:
            return self.prompt_ctx.render_head()
        return pc.head_template.format(
            dependent=pc.dependent_name_in_prompt,
            problem=pc.problem_name_in_prompt,
            independent=pc.independent_name_in_prompt,
        )

    def inject_experiences(
            self, content: str, policy: config_lib.ExperienceInjectionConfig) -> str:
        """从 experiences.json 选经验条目拼块，前置注入 content；无经验文件则原样返回。"""
        experiences = self.load_json_cached(self._path("experiences.json"))
        if experiences is None:
            return content

        current_sample_order = self._current_sample_order(experiences)
        selected = self.select_experiences(experiences, current_sample_order, policy)
        if not selected:
            return content

        experience_prompt = self.build_experience_prompt(
            selected, policy.max_analysis_chars)
        if experience_prompt:
            print_block(f"[经验] 注入 {len(selected)} 条经验（进度 sample_order={current_sample_order}）")
            content = experience_prompt + "\n\n" + content
        return content

    def inject_residual(self, content: str, inject_residual_probability: float) -> str:
        """以 inject_residual_probability 概率注入最近一条残差分析，前置 content。

        合并原先 if/else 两个几乎重复的分支：统一处理 analysis 为 list 的历史格式
        （取首条），并截断到 2000 字符。
        """
        if random.random() >= inject_residual_probability:
            return content
        # 仅当经验文件存在（已进入采样循环）时才注入残差，避免首轮空注入
        if not os.path.exists(self._path("experiences.json")):
            return content

        residual_data = self.load_json_cached(self._path("residual_analyze.json"))
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
            self.prompt_ctx.render_residual_block_title()
            if self.prompt_ctx is not None
            else pc.residual_block_title.format(problem=pc.problem_name_in_prompt)
        )
        print_block(f"[残差] 注入最近残差分析（sample_order={last.get('sample_order', 'unknown')}）")
        return block_title + last_analysis + "\n\n" + content

    # ------------------------------------------------------------------
    # 选择与拼装
    # ------------------------------------------------------------------
    def select_experiences(self, experiences: dict, current_sample_order: int,
                           policy: config_lib.ExperienceInjectionConfig) -> list:
        """按类别筛选 + 排序 + 截断，返回扁平化经验条目列表。

        每条含 type/analysis/sample_order；None 类额外附 error（过滤掉指定噪声错误）。
        """
        selected = []
        for category in ("None", "Good", "Bad"):
            category_exps = experiences.get(category) or []
            if not category_exps:
                continue
            # None 类始终注入；Good / Bad 先按概率决定是否注入
            if category != "None" and random.random() >= policy.optional_category_probability:
                continue
            # 新鲜度窗口：样本进度超过阈值后，只保留近期经验
            if current_sample_order > policy.freshness_threshold:
                min_order = current_sample_order * policy.freshness_window_ratio
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
            limit = policy.max_per_category.get(category, 2)
            category_exps = category_exps[:limit]

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

    def build_experience_prompt(self, selected: list, max_analysis_chars: int) -> str:
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
                self.prompt_ctx.max_param_count
                if self.prompt_ctx is not None else None
            )
            if max_params is not None:
                prompt += (
                    f"Note: the evaluator passes exactly {max_params} trainable parameters "
                    f"(params[0]..params[{max_params - 1}]). "
                    "Keep every equation within this budget; do not request more parameters.\n"
                )
        return prompt

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _path(self, name: str) -> str:
        return os.path.join(self.base_dir or ".", name)

    @staticmethod
    def _current_sample_order(experiences: dict) -> int:
        """当前样本进度 = 各类别中最大的 sample_order。

        历史实现用条数之和，多轮累计后远大于真实样本序号，会把新鲜度窗口
        误判成"已进入后期"，从而把所有经验都过滤掉。
        """
        current = 0
        for category in ("None", "Good", "Bad"):
            for exp in experiences.get(category, []):
                order = exp.get("sample_order", 0)
                if isinstance(order, (int, float)):
                    current = max(current, int(order))
        return current

    def load_json_cached(self, path: str):
        """读取 JSON 并按 mtime 缓存；文件不存在或解析失败返回 None。

        PromptInjector 实例随 CoordinatorAgent 生命周期复用，因此缓存覆盖整个
        采样线程的连续轮次；仅在文件被原子替换（mtime 变化）时重新读盘。
        """
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            # 文件不存在或不可访问：清掉可能的旧缓存并返回 None
            if self._cache.get('path') == path:
                self._cache.clear()
            return None
        cache = self._cache
        if cache.get('path') == path and cache.get('mtime') == mtime:
            return cache.get('data')
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        self._cache = {'path': path, 'mtime': mtime, 'data': data}
        return data
