"""残差分析 Agent：让 LLM 根据输入残差分析方程，输出对数据的思考过程与改进建议。

角色：ResidualAnalyzerAgent 是"残差分析者"，把评估残差的统计信息与上一次分析上下文
拼进提示词，让 LLM 输出结构化修正方向，供采样阶段注入下一轮提示词。

协作：
- 上游：CoordinatorAgent（传入最优样本及其残差矩阵）；
- 下游：LLMClient（分析调用），产物写入 residual_analyze.json（由 CoordinatorAgent 持久化）。
"""
from __future__ import annotations

import json
import os
from drsr_420.core.console import StreamDeltaPrinter, print_block
import traceback

import numpy as np

from drsr_420.core import prompt_config as pc

from drsr_420.agents.base import THREAD_PER_SAMPLER, AgentSpec, BaseAgent
from drsr_420.agents.messages import ResidualInsight
from drsr_420.evaluation.architecture_facts import (
    features_from_equation,
    load_terrain_inputs,
    render_terrain,
    sampling_terrain,
    with_score_breakdown,
)
from drsr_420.evaluation.data_facts import load_facts, render_facts


class ResidualAnalyzerAgent(BaseAgent):
    """根据输入残差让 LLM 分析方程。"""

    SPEC = AgentSpec(
        key="residual_analyzer",
        role="残差分析者",
        mission="统计残差并结合上一次分析，让 LLM 输出结构化修正方向",
        entrypoints=("analyze",),
        upstream=("coordinator",),
        downstream=(),
        consumes=("sample: str", "residual: np.ndarray  # 最后一列为残差值"),
        produces=("insight: ResidualInsight",),
        artifacts=("residual_analyze.json",),   # 由 CoordinatorAgent 落盘
        thread_model=THREAD_PER_SAMPLER,
        llm_task="residual",
    )

    def __init__(self, llm_client, prompt_ctx=None, results_root='.'):
        self._llm_client = llm_client
        self._prompt_ctx = prompt_ctx
        self._results_root = results_root or '.'

    def _load_facts_block(self) -> str:
        """读取代码实测的数据事实表并渲染（缺失/损坏时返回空串，不注入）。

        残差分析只拿到 (输入, 残差) 矩阵、拿不到原始因变量，无法自己重算统计量；
        事实表由初次分析（DataAnalyzerAgent）写入同一实验目录，这里读盘复用，
        保证"峰在哪""谁与响应相关"这类断言必须与实测数字一致。
        """
        try:
            facts = load_facts(self._results_root)
        except Exception as e:
            print(f"读取数据事实表失败（跳过注入）: {e}")
            return ""
        return render_facts(facts) if facts else ""

    def _load_terrain_block(self, sample) -> str:
        """读取"该方程所属架构在历史采样中的地形"并渲染（缺失/不足时不注入）。

        与采样通道共用同一份机器事实（见 :mod:`drsr_420.evaluation.architecture_facts`），
        但 ``target`` 是**当前被分析的方程**：残差通道要说清"这个架构是不是已经触底、
        它最小的未试删项邻域是什么"，否则改进建议只会停留在局部改动上（实测该实验
        52/87 个样本是同一二阶响应面的重新参数化，删一个平方项的形式从未被提出）。

        与采样通道同样带上两样东西（缺了它们这一注会被读成"别回那个家族"）：
        未试邻域的**实测 NMSE**（同评估器口径拟合一遍）与分数的**分解**（拟合 MSE
        与体检罚分分开——否则"MSE 0.197 + 罚分 36.06"会被当成胜利）。
        """
        try:
            path = os.path.join(self._results_root, "experiences.json")
            with open(path, "r", encoding="utf-8") as f:
                experiences = json.load(f)
        except (OSError, json.JSONDecodeError):
            return ""
        if not isinstance(experiences, dict):
            return ""

        features = None
        if self._prompt_ctx is not None:
            features = [str(name) for name in self._prompt_ctx.features]
        if not features or len(features) != 2:
            features = features_from_equation(sample)
        if not features or len(features) != 2:
            return ""

        entries = []
        for category in ("None", "Good", "Bad"):
            entries.extend(experiences.get(category) or [])
        inputs = load_terrain_inputs(self._results_root)
        terrain = with_score_breakdown(
            sampling_terrain(entries, features, target=str(sample or ""),
                             facts=inputs["facts"]),
            inputs["records"])
        return render_terrain(terrain, features, pc.architecture_block_title)

    def analyze(self, sample, residual) -> ResidualInsight:
        """构造残差分析提示并调用 LLM，返回残差洞察（样本 + 分析文本）。

        归属字段（``island_id`` / ``sample_order`` / ``best_score``）由
        CoordinatorAgent 在落盘前补齐——本 Agent 不掌握这些信息。
        """
        print("========================进入了残差分析函数========================")
        # 计算残差的统计信息（供日志与后续扩展；当前提示词模板只用残差矩阵本身，
        # 因而这三个量不进提示词——不要误以为它们已被使用）
        res_values = residual[:, -1]  # 最后一列是残差值
        mean_res = np.mean(res_values)
        max_res = np.max(np.abs(res_values))
        std_res = np.std(res_values)

        # 读取上一次残差分析，作为上下文
        last_analysis = ""
        try:
            residual_file = os.path.join(self._results_root, "residual_analyze.json")
            if os.path.exists(residual_file):
                with open(residual_file, "r", encoding="utf-8") as f:
                    experiences = json.load(f)
                # 提取最后一条分析
                if experiences:
                    last_analysis = experiences[-1].get("analysis", "")
        except Exception as e:
            print(f"加载残差数据时出错: {str(e)}")
            traceback.print_exc()

        # 构建分析提示
        if self._prompt_ctx is not None:
            res_analyze = self._prompt_ctx.render_residual_analysis_prompt(
                last_analysis, residual, sample)
        else:
            res_analyze = pc.residual_analysis_prompt.format(
                last_analysis=last_analysis,
                residual=residual,
                sample=sample,
            )
        # 附上代码实测的数据事实表（与初次分析共用同一份 <results_root>/data_facts.json）
        facts_block = self._load_facts_block()
        if facts_block:
            res_analyze += facts_block
        # 再附上"该架构在历史采样中的地形"：架构是否触底 + 未试的删项邻域（机器事实）
        terrain_block = self._load_terrain_block(sample)
        if terrain_block:
            res_analyze += terrain_block

        print_block("========这是输入的残差提示词==========\n")
        print_block(res_analyze)
        # 调用远程API分析结果（仅使用注入的 llm_client）
        try:
            # 流式输出：通过 on_delta 回调实时打印思考内容与正文（[思考]/[正文] 视觉分隔）
            printer = StreamDeltaPrinter()
            resp = self._llm_client.chat([
                {"role": "system", "content": pc.system_prompt},
                {"role": "user", "content": res_analyze},
            ], on_delta=printer.on_delta)
            printer.flush()
            print()  # 流式结束后换行
            # 兜底：推理模型可能把完整分析输出在 reasoning_content 而 content 为空
            analysis_result = resp.get('content', '') or resp.get('reasoning_content', '')
            return ResidualInsight(sample=sample, analysis=analysis_result)
        except Exception as e:
            print(f"残差分析请求发生错误: {str(e)}")
            return ResidualInsight(sample=sample, analysis=f"分析请求发生错误: {str(e)}")
