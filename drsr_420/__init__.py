"""DRSR —— 基于多智能体的科学方程发现（Multi-agent Scientific Equation Discovery）。

分层结构（依赖方向自上而下，硬约束由 tests/test_architecture.py 守护）：

    cli        命令行入口
    harness    论文实验设施（跨 run 指标与汇总）
    runtime    编排与执行（pipeline）
    agents     ★ 多 Agent 角色层（7 个 Agent + 契约 + 消息）
    reporting  收尾分析与报告装配（剪枝 / 泛化验证 / 物理解释）
    knowledge  外部知识（RAG 知识库 / MCP 工具）
    evidence   ★ 可验证证据层（数据事实 / 架构地形 / 未试邻域实测）
    execution  评估执行机制（多起点拟合打分 / 常驻沙箱 / 加速）
    llm        LLM 接入（客户端 / 工厂 / 提供商适配 / 统计）
    equations  公式领域模型（解析 / 求值 / 样本词汇 / 数值病理内核）
    core       领域无关基础设施（记忆 / 配置 / 日志 / AST / 提示词模板）

旧的一层平铺路径（drsr_420.buffer 等）已随分层重构清退，规范导入路径见
docs/ARCHITECTURE.md（§2 分层表 / §8 兼容层）。

注意：本模块**不 import 任何子模块**，以避免循环导入与无谓的启动开销。
"""

__version__ = "0.4.0"   # 0.4 = 清退兼容层（见 docs/REFACTOR_PLAN.md §11）

__all__ = ["__version__"]
