# DrSR：基于多智能体的科学方程发现

Multi-agent based Scientific Equation Discovery（类似符号回归）。

## 简介

本项目实现了一个"基于多智能体的方程搜索 + 数据拟合"的工作流：

- **公式发现智能体**：调用文献搜索/阅读工具获取领域知识，结合经验总结与残差分析的结果推导公式
- **经验总结智能体**：经验缓冲区（Experience Buffer）保留更优样本，持续迭代搜索更好结构
- **残差分析智能体**：分析当前公式的预测值与数据集实际值之间的残差
- **公式解释智能体**：对参数拟合后的公式进行力学/物理角度的解释

参考论文：Wang et al., *DrSR: LLM based Scientific Equation Discovery with Dual Reasoning from Data and Experience*, arXiv:2506.04282。

## 安装依赖

```bash
pip install -r requirements.txt
```

> 仅依赖 NumPy/SciPy/Pandas 等核心包，已移除 torch/transformers 等重依赖。
> RAG 默认走嵌入 API（见下），本地嵌入所需 `sentence-transformers` 已标注为可选。

## 快速开始

CSV 需带表头：前 n-1 列为特征，最后一列为因变量。

```bash
python main.py \
  --problem_name oscillator1 \
  --data_csv ./data/oscillator1/train.csv \
  --background 'Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on position, and velocity.'
```

运行后在 `experiments/{problem}_{时间戳}/` 下生成所有产物。

可选调参示例（最大采样数 ≈ niterations × num_samplers × samples_per_iteration）：

```bash
python main.py --problem_name oscillator1 --data_csv ./data/oscillator1/train.csv \
  --niterations 50 --samples_per_iteration 8
```

批量示例见根目录 `example.sh`。

## LLM 配置（按 提供商_模型.config 命名）

根目录提供按 `提供商_模型.config` 命名的 JSON 配置文件（如 `glm_glm-5.3-flash.config`、`deepseek_deepseek-v4-flash.config`），用于配置大模型访问与采样参数：

```json
{
  "host": "api.deepseek.com",
  "api_key": "xxx",
  "model": "deepseek/deepseek-v4-flash",
  "max_tokens": 65536,
  "temperature": 0.7,
  "top_p": 0.95,
  "tasks": {
    "sampling": {"reasoning_effort": "low"},
    "analysis": {"reasoning_effort": "high"},
    "summary": {"reasoning_effort": "high"},
    "experience": {"reasoning_effort": "high"},
    "residual": {"reasoning_effort": "high"},
    "explain": {"reasoning_effort": "high"}
  }
}
```

- `api_key` 请替换为真实密钥，否则会报"未提供令牌"。
- `model` 使用 `provider/model` 形式。支持提供商：`deepseek`、`siliconflow`、`deepinfra`、`ollama`、`blt`（柏拉图）、`cstcloud`（科技云）、`glm`（智谱）。
- 配置文件按提供商与模型命名（`提供商_模型.config`），与具体任务解耦：任务级私有参数（如思考强度）统一放在 `tasks` 字段中按任务声明。
- 切换模型直接修改对应配置文件名即可（如 `deepseek_deepseek-v4-flash.config`）；`api_key` 留空时回退读取对应环境变量（如 `DEEPSEEK_API_KEY`、`ZHIPU_API_KEY`、`SILICONFLOW_API_KEY`）。
- 运行时每个任务实例化一个 LLM Client 并全程复用，并行任务互不影响。

## RAG 文献知识库

项目内置 Chroma 持久化向量库（`knowledge_base/chroma_db`），用于检索文献背景注入提示词。

- 配置：`rag.config`（嵌入后端 `local`/`api`、API 主机/密钥/模型、分块大小、检索 `k` 等）。
- 默认 `backend=api` 走 OpenAI 兼容嵌入接口（如智谱 `embedding-3`、SiliconFlow `BAAI/bge-m3`）；`api_key` 留空时按主机回退环境变量。

入库文献：

```bash
python -m drsr_420.rag_build --ingest [--dir pdf_downloads] [--limit N] [--rebuild]
```

检索：

```bash
python -m drsr_420.rag_build --query "磁流变 屈服应力 压缩" [--k 5]
```

> 切换嵌入模型后维度会变化，需带 `--rebuild` 重建集合。嵌入 API 按 token 计费。

## 文献工具与 MCP

- `drsr_420/tools/search_paper.py`：Crossref 文献检索
- `drsr_420/tools/read_paper.py`：文献下载（Sci-Hub 多镜像 + Unpaywall OA）+ PDF 解析总结
- `drsr_420/tools/mcp_server.py`：将上述工具封装为 MCP 服务器（stdio / HTTP）
- `drsr_420/tool_runner.py`：agent 通过 MCP 调用工具的统一入口（替代直接调用）

程序运行时会经 MCP 拉起工具服务器并复用；`read_paper` 的总结模型由对应模型配置文件（如 `glm_glm-5.3-flash.config`）配置。

## 结果产物与目录结构

以 `experiments/oscillator1_20250101-120000/` 为例：

- `run.out` / `run.err`：标准输出/错误输出
- `spec_dynamic.txt`：本次运行的动态 spec（便于复现）
- `experiences.json`：采样过程中的经验/总结
- `residual_analyze.json`：残差分析结果
- `explain.txt`：最终公式的力学解释（若启用）
- `samples/`：每次评分的样本 JSON（`samples_N.json`），含 `score`、`function`、`params`

## 多 Agent 系统架构

本项目按角色将工作流拆分为独立 Agent，由 `CoordinatorAgent` 并行协调，体现"感知 → 决策 → 行动 → 反思"的多智能体闭环：

```
                 ┌────────────────────────────────────────────┐
                 │            CoordinatorAgent                 │
                 │  (每个 Sampler-i 线程一个实例，共享记忆)        │
                 │  驱动主循环：采样→评估→反思→持久化              │
                 └──────┬──────────┬──────────┬─────────┬──────┘
                        │          │          │         │
        ┌───────────────▼──┐  ┌────▼─────┐ ┌──▼─────┐ ┌▼──────────────┐
        │   SamplerAgent   │  │Evaluator │ │Experience│ │ Residual      │
        │  生成方程骨架      │  │  Agent   │ │Summarizer│ │ Analyzer      │
        │  空骨架重采样      │  │运行+拟合  │ │Agent     │ │ Agent         │
        └───────┬──────────┘  │+打分      │ │经验总结   │ │残差结构修正建议 │
                │             └──────────┘ └─────────┘ └───────────────┘
        ┌───────▼──────────┐
        │  ToolCallerAgent │──▶ MCP 工具（文献搜索/阅读）
        └───────┬──────────┘
        ┌───────▼──────────┐
        │     LLMClient    │── 采样/总结/残差分析各自独立实例（独立温度）
        └──────────────────┘
```

- **DataAnalyzerAgent**：实验启动时做初次数据分析（含 RAG 文献注入），产出 `residual_analyze.json` 基线
- **CoordinatorAgent**：多线程（`Sampler-i`）并行协调者，共享 `ExperienceBuffer` 与全局采样计数，循环执行采样→评估→经验总结→残差分析→持久化
- **SamplerAgent**：基于提示词 + 经验注入生成方程骨架，空骨架自动重采样
- **EvaluatorAgent**：在 `LocalSandbox`（常驻 worker 池）中运行候选方程，多起点 `least_squares` 拟合打分
- **ExperienceSummarizerAgent / ResidualAnalyzerAgent**：对评估结果反思，产出经验条目与残差修正建议并写回经验缓冲
- **find_best_eq**：收尾工具函数（非 Agent），对最优方程做参数拟合与物理解释

### 模块映射

| 新模块（agents 子包）                 | 旧模块（兼容层，仅 re-export）     |
|--------------------------------------|-----------------------------------|
| `agents/data_analyzer_agent.py`      | `data_analyse_real.py`            |
| `agents/tool_caller_agent.py`        | `tool_caller.py`                  |
| `agents/sampler_agent.py`            | `sampler.py`（Sampler/LLM）       |
| `agents/evaluator_agent.py`          | `evaluator.py`                    |
| `agents/experience_summarizer_agent.py` | `experience_summarizer.py`      |
| `agents/residual_analyzer_agent.py`  | `residual_analyzer.py`            |
| `agents/coordinator_agent.py`        | `sampler.py`（SamplingOrchestrator）|

旧模块名全部保留，外部代码与测试无需改动。

## 仓库结构

```
main.py                       # 入口（CSV 动态模式）
llm.py                        # LLM 客户端与 ClientFactory（多提供商 + 参数统一注入）
glm_glm-5.3-flash.config / deepseek_deepseek-v4-flash.config / rag.config   # 配置文件（不入库）
example.sh                    # 批量运行示例
drsr_420/
  pipeline.py                 # 主流程编排：初始化记忆/Profiler/Evaluator，启动多 Agent 并行采样
  agents/                     # 多 Agent 系统子包（角色化实现）
    coordinator_agent.py      # 协调 Agent：采样→评估→反思→持久化主循环（多线程）
    sampler_agent.py          # 采样 Agent：LLM 骨架生成 + 工具调用 + 空骨架重采样
    evaluator_agent.py        # 评估 Agent：运行候选方程、BFGS 拟合与打分
    tool_caller_agent.py      # 工具调用 Agent：多轮 MCP 工具调用循环
    experience_summarizer_agent.py  # 经验总结 Agent：样本质量分析
    residual_analyzer_agent.py      # 残差分析 Agent：残差统计与结构修正建议
    data_analyzer_agent.py    # 数据分析 Agent：初次数据分析 + RAG 文献注入
  sampler.py / evaluator.py / tool_caller.py / experience_summarizer.py / residual_analyzer.py / data_analyse_real.py   # 兼容层（re-export）
  evaluate_on_problems.py     # BFGS 与评分逻辑（返回拟合参数）
  evaluator_accelerate.py     # numba 加速装饰（可选）
  buffer.py                   # 经验缓冲（多岛与聚类抽样）
  code_manipulation.py        # AST 解析与函数/程序拼装
  prompt_config.py            # 提示词模板与 PromptContext
  profile.py                  # 采样结果记录（samples/*.json）
  find_best_eq.py             # 最优公式解释
  sensitivity_prune.py        # 敏感度剪枝，降低公式复杂度
  tool_runner.py              # MCP 工具调用入口
  rag_kb.py / rag_build.py    # RAG 向量库与入库/检索 CLI
  tools/
    mcp_server.py             # 工具 MCP 服务器
    tools_description.py      # 工具 schema
    search_paper.py           # 文献搜索工具
    read_paper.py             # 文献下载/阅读工具
specs/                        # 历史静态 spec（动态模式无需）
experiments/{problem}_{timestamp}/   # 本次运行产物
```
