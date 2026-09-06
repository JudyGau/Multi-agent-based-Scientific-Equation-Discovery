"""DRSR 多 Agent 系统 —— agents 子包。

本包把 DRSR（Deep Reasoning Symbolic Regression）主循环中的协作角色
拆分为独立的 agent 类，每个文件对应一个 agent 角色 + 其私有辅助函数。

agent 协作图
============

DataAnalyzerAgent ──初次数据分析──▶ residual_analyze.json (sample_order=0)
         │
CoordinatorAgent (Sampler-i 线程，共享 ExperienceBuffer / _SAMPLER_LOCK / 全局采样计数)
  ├─▶ SamplerAgent ──▶ ToolCallerAgent ──▶ LLMClient ──▶ MCP 工具 (tool_runner)
  ├─▶ EvaluatorAgent ──▶ LocalSandbox(常驻 worker) ──▶ evaluate_on_problems(多起点 least_squares)
  ├─▶ ExperienceSummarizerAgent ──▶ experiences.json
  └─▶ ResidualAnalyzerAgent ──▶ residual_analyze.json
收尾：find_best_eq()（工具函数，非 agent）

角色一览
========
- CoordinatorAgent        协调者：驱动"采样→评估→反思→持久化"主循环，多线程并行（Sampler-i）。
- SamplerAgent            采样者：调用 LLM 生成方程骨架，含提示词构造（经验/残差注入）与空骨架重采样。
- ToolCallerAgent         工具调用者：与 LLM 多轮对话并执行其发起的 MCP 工具调用。
- EvaluatorAgent          评估者：编译骨架、在常驻沙箱 worker 中执行并评分（多起点 least_squares）。
- ExperienceSummarizerAgent 经验总结者：对 Good/Bad/None 样本做 LLM 分析，产出改进建议。
- ResidualAnalyzerAgent   残差分析者：统计残差并让 LLM 输出结构化修正方向。
- DataAnalyzerAgent       数据分析者：初次数据分析（写 sample_order=0 的初始残差记录）。

注意：本包 __init__ 刻意不做任何 import（防止循环导入）。
agents 内部一律 `from drsr_420.agents.xxx import ...`；
旧模块名（drsr_420.sampler / drsr_420.evaluator 等）由兼容层 re-export 到本包。
"""
