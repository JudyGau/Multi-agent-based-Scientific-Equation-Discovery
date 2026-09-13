# DrSR：基于多智能体的科学方程发现

Multi-agent based Scientific Equation Discovery（符号回归）。

用多个 LLM Agent 协作发现科学方程：协调者驱动「采样 → 评估 → 反思 → 持久化」闭环，
把 LLM 生成的方程骨架放进常驻沙箱执行、多起点拟合参数，按得分分层反馈回提示词，
并用多岛经验缓冲维持种群多样性。

参考论文：Wang et al., *DrSR: LLM based Scientific Equation Discovery with Dual
Reasoning from Data and Experience*, arXiv:2506.04282。

> **架构文档**：分层结构、依赖规则、7 个 Agent 的角色卡与协作时序见
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。
> 想直接看系统长什么样，运行 `python -m drsr_420.agents` 会打印由代码生成的
> Agent 组织图。

## 安装依赖

```bash
pip install -r requirements.txt
```

> 仅依赖 NumPy/SciPy/Pandas 等核心包，已移除 torch/transformers 等重依赖。
> RAG 默认走嵌入 API（见下），本地嵌入所需 `sentence-transformers` 已标注为可选。

也可以作为包安装（`pyproject.toml` 提供元数据，依赖清单复用 `requirements.txt`）：

```bash
pip install -e .          # 之后可用 `drsr420` 命令；包自包含，不再依赖"仓库根在 sys.path 上"
```

## 快速开始

CSV 需带表头：前 n-1 列为特征，最后一列为因变量。

```bash
python -m drsr_420.cli.main \
  --problem_name oscillator1 \
  --data_csv ./data/oscillator1/train.csv \
  --background 'Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on position, and velocity.'
```

运行后在 `experiments/{problem}_{时间戳}/` 下生成所有产物。

> 须在**仓库根目录**执行：`--data_csv ./data/…` 与 `--llm_config llm.config` 都按当前
> 工作目录解析（IDE 运行配置的 `WORKING_DIRECTORY` 也正是 `$PROJECT_DIR$`）。

可选调参示例（最大采样数 ≈ niterations × num_samplers × samples_per_iteration）：

```bash
python -m drsr_420.cli.main --problem_name oscillator1 --data_csv ./data/oscillator1/train.csv \
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
- 仓库提供 `llm.config.example` 作为模板：真实配置文件受 `.gitignore` 的 `llm.config` / `*.config` 规则保护不会入库，故模板以 `.example` 结尾以便随仓库分发。新克隆的仓库执行 `cp llm.config.example llm.config` 并填入密钥即可运行；`.idea/runConfigurations/` 下的 IDE 运行配置默认使用 `--llm_config llm.config`。
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
python -m drsr_420.knowledge.rag_build --ingest [--dir pdf_downloads] [--limit N] [--rebuild]
```

检索：

```bash
python -m drsr_420.knowledge.rag_build --query "磁流变 屈服应力 压缩" [--k 5]
```

> 切换嵌入模型后维度会变化，需带 `--rebuild` 重建集合。嵌入 API 按 token 计费。

## 文献工具与 MCP

- `drsr_420/knowledge/tools/search_paper.py`：Crossref 文献检索
- `drsr_420/knowledge/tools/read_paper.py`：文献下载（Sci-Hub 多镜像 + Unpaywall OA）+ PDF 解析总结
- `drsr_420/knowledge/tools/mcp_server.py`：将上述工具封装为 MCP 服务器（stdio / HTTP）
- `drsr_420/knowledge/tool_runner.py`：agent 通过 MCP 调用工具的统一入口

程序运行时会经 MCP 拉起工具服务器并复用；`read_paper` 的总结模型由对应模型配置文件（如 `glm_glm-5.3-flash.config`）配置。

## 结果产物

以 `experiments/oscillator1_20250101-120000/` 为例：

- `run.out` / `run.err`：标准输出/错误输出
- `spec_dynamic.txt`：本次运行的动态 spec（便于复现）
- `config_snapshot.json`：超参与 LLM 配置快照（api_key 打码）
- `experiences.json`：采样过程中的经验/总结
- `residual_analyze.json`：残差分析结果
- `checkpoint.json`：断点续跑用的经验缓冲 + 全局采样数
- `round_progress.csv`：每轮进度（墙钟、岛屿、最佳分、采样数）
- `explain.txt`：最终公式的力学解释（若启用）
- `samples/`：每次评分的样本 JSON（`samples_N.json`），含 `score`、`function`、`params`

字段级说明见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §6。

## 多 Agent 系统

7 个协作角色（协调者 / 采样者 / 工具调用者 / 评估者 / 经验总结者 / 残差分析者 /
数据分析者）由 `CoordinatorAgent` 统一编排：

```
CoordinatorAgent（每个 Sampler-i 线程一个实例，共享经验缓冲）
  ├─▶ SamplerAgent ──▶ ToolCallerAgent ──▶ LLMClient ──▶ MCP 工具（文献检索）
  ├─▶ EvaluatorAgent ──▶ LocalSandbox（常驻 worker）──▶ 多起点 least_squares 拟合
  ├─▶ ExperienceSummarizerAgent ──▶ experiences.json
  └─▶ ResidualAnalyzerAgent ──▶ residual_analyze.json
收尾：find_best_eq()（工具函数，非 Agent）
```

- 每个 Agent 的**角色卡**（职责、入口、上下游、输入输出契约、落盘产物）由代码声明
  （`agents/base.py` 的 `AgentSpec`），可枚举、可校验、可渲染成组织图；
- Agent 之间传递的是 `agents/messages.py` 里的显式类型，而不是裸元组或 `**kwargs` 袋；
- 详细职责、关键接口与调用示例见 [`drsr_420/agents/README.md`](drsr_420/agents/README.md)；
  分层与依赖规则见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

```bash
python -m drsr_420.agents            # 打印 Agent 组织图（由代码生成）
python -m drsr_420.agents --check    # 契约自检：上下游引用 / 可达性 / 线程模型
```

## 仓库结构

```
glm_glm-5.3-flash.config / deepseek_deepseek-v4-flash.config / rag.config   # 配置文件（不入库）
example.sh                    # 批量运行示例
drsr_420/                     # 单一顶层包（8 层，依赖方向自底向上）
  core/                       # 领域无关基础设施
    buffer.py                 #   经验缓冲（多岛 + 聚类抽样）
    code_manipulation.py      #   AST 解析与函数/程序拼装
    config.py                 #   实验配置 dataclass
    console.py                #   线程前缀输出 / 流式增量打印
    profile.py                #   样本与进度记录（samples/*.json、progress.json）
    prompt_config.py          #   提示词模板与 PromptContext
    llm_stats.py              #   实验级全局 token / 耗时统计
  llm/                        # LLM 接入层
    client.py                 #   LLMClient：请求/重试/流式/参数适配/记账
    providers.py              #   各提供商子类
    factory.py                #   ClientFactory / 配置加载与归一化
    tools_schema.py           #   工具调用 schema
  evaluation/                 # 评估执行机制
    problems.py               #   多起点 least_squares 拟合与打分
    sandbox.py                #   Sandbox / LocalSandbox（常驻 worker、超时重建）
    accelerate.py             #   可选 numba 加速
  knowledge/                  # 外部知识
    rag_kb.py / rag_build.py  #   Chroma 知识库与入库/检索 CLI
    tool_runner.py            #   MCP 调用入口
    tools/                    #   search_paper / read_paper / MCP 服务器
  agents/                     # ★ 多 Agent 角色层
    base.py                   #   AgentSpec（角色卡）+ BaseAgent
    messages.py               #   Agent 间消息与落盘条目类型
    coordinator_agent.py      #   协调者：采样→评估→反思→持久化主循环（多线程）
    sampler_agent.py          #   采样者：采样编排 + 空骨架重采样
    prompt_injection.py       #     └ 提示词装配（经验/残差注入策略，非 Agent）
    skeleton.py               #     └ 骨架提取（从混合文本切出可执行体）
    tool_caller_agent.py      #   工具调用者：多轮 MCP 工具调用循环
    evaluator_agent.py        #   评估者：编译 + 委托沙箱 + 打分 + 注册经验
    experience_summarizer_agent.py  # 经验总结者
    residual_analyzer_agent.py      # 残差分析者
    data_analyzer_agent.py    #   数据分析者：初次数据分析 + RAG 注入
  analysis/                   # 收尾分析
    find_best_eq.py           #   收尾编排：最佳样本 → 解释 → 剪枝与可视化
    expr_parse.py             #   骨架字符串 → SymPy 表达式（where/Eq/中间变量）
    explain.py                #   物理解释（ReAct + RAG）→ explain.txt
    sensitivity_prune.py      #   敏感度剪枝（遍历与决策）
    expr_evaluation.py        #     └ 采样网格 / 求值 / 敏感度度量
    prune_stats.py            #     └ 剪枝记录与统计
    expr_viz.py               #   预览图与表达式树（可选依赖，失败仅告警）
    prune_demo.py             #   剪枝行为演示（python -m …prune_demo）
  runtime/
    pipeline.py               #   实验主流程编排
  cli/
    main.py                   #   命令行入口：python -m drsr_420.cli.main（拆分为可测函数）
specs/                        # 历史静态 spec（动态模式已不使用，保留备查）
experiments/{problem}_{timestamp}/   # 本次运行产物
```

> `drsr_420/` 顶层只有 `__init__.py`：实现全部分层，历史的一层平铺路径已清退
> （见 `docs/ARCHITECTURE.md` §8）。
>
> **仓库根目录没有任何 `.py` 模块**：命令行入口是 `python -m drsr_420.cli.main`
> （安装后等价于 `drsr420` 命令），`.idea/runConfigurations/` 的 4 个 MRF 运行配置、
> `example.sh` 与 `MRFCompress-3.sh` 都以模块方式启动。因此包自包含、克隆即可运行，
> 也不会被 PyPI 上的同名 `llm` 包劫持。

## 测试

```bash
# Windows 下务必用这个入口（LocalSandbox 用 spawn，需要 __main__ 保护）
python tests/run_tests.py
```

测试包含功能回归、**Agent 行为测试**与**架构护栏**（分层目录、依赖方向、旧路径不得复活、
`__file__` 路径锚点、Agent 契约完整性），见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §9。
