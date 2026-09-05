"""残差分析：让 LLM 根据输入残差分析方程，输出对数据的思考过程与改进建议。

从 sampler.py 的 analyze_equations_with_residual 拆分，可注入 mock client 单独单测。
"""
from __future__ import annotations

import json
import os
import traceback

import numpy as np

from drsr_420 import prompt_config as pc


class ResidualAnalyzer:
    """根据输入残差让 LLM 分析方程。"""

    def __init__(self, llm_client, prompt_ctx=None, results_root='.'):
        self._llm_client = llm_client
        self._prompt_ctx = prompt_ctx
        self._results_root = results_root or '.'

    def analyze(self, sample, residual) -> str:
        """构造残差分析提示并调用 LLM，返回分析结果字符串。"""
        print("========================进入了残差分析函数========================")
        # 计算残差的统计信息
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

        print("========这是输入的残差提示词==========\n")
        print(res_analyze)
        # 调用远程API分析结果（仅使用注入的 llm_client）
        try:
            resp = self._llm_client.chat([{"role": "user", "content": res_analyze}])
            analysis_result = resp.get('content', '')
            print(f"残差分析结果：{analysis_result}")
            return analysis_result
        except Exception as e:
            print(f"残差分析请求发生错误: {str(e)}")
            return f"分析请求发生错误: {str(e)}"
