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

## LLM 配置：角色 → 档案

配置分三层，**扩展名就是保密边界**：

```
config/
├── agents.config.json                      # 入库（无密钥）：角色 → 档案 + 参数覆盖
├── <提供商>_<模型>.config                  # 不入库（含密钥）：连接信息 + 生成参数
└── <提供商>_<模型>.config.example          # 入库：模板（api_key 留空）
```

**「哪个 Agent 用哪套配置」只在一个地方声明**——`config/agents.config.json`：

```json
{
  "default": "glm_glm-5.3-flash",
  "roles": {
    "sampling":   {"params": {"reasoning_effort": "low"}},
    "analysis":   {"params": {"reasoning_effort": "high"}},
    "experience": {"params": {"reasoning_effort": "high", "temperature": 0.0}},
    "residual":   {"params": {"reasoning_effort": "high", "temperature": 0.4}},
    "explain":    {"config": "deepseek_deepseek-v4-flash"},  // 给单个角色换模型
    "summary":    {"config": "ustc_deepseek-v4-flash"}       // 自定义提供商（校内网关）
  }
}
```

6 个角色：`sampling`（采样者+工具调用者）、`analysis`（数据分析者）、`experience`（经验总结者）、
`residual`（残差分析者）、`explain`（收尾物理解释）、`summary`（文献摘要，跑在 MCP 子进程里）。
省略 `config` 即沿用 `default`。上面前两个角色单独绑定，是因为"给某个角色换模型"在旧结构里
**原理上无法表达**（详见 `docs/CONFIG_PLAN.md`）。

```bash
python -m drsr_420.llm.roles            # 打印「角色 → 档案（生效来源）」
python -m drsr_420.llm.roles --check    # 离线自检：档案存在、model 合法、密钥可达
python -m drsr_420.llm.roles --ping     # 联网自检：每份档案发一次真实请求（消耗少量 token）
```

`--check` 不联网（给 CI / 预检用），`--ping` 才会真的打端点；两者都会指出问题落在哪份档案、
哪个角色。客户端是**按需构造**的：一份档案暂时不可用（密钥没填、端点写错）只会让**用到它的
那个角色**报错，不会拦住整个实验；启动时会跑一遍离线自检并逐条 `[WARN]`。

**档案文件**（`config/<提供商>_<模型>.config`）配置连接与生成参数：

```json
{
  "base_url": "https://open.bigmodel.cn/api/paas/v4",
  "api_key": "xxx",
  "model": "glm/glm-5.3-flash",
  "max_tokens": 65536,
  "temperature": 0.7,
  "top_p": 0.95
}
```

端点必须写成**完整 URL**（`https://<主机>/<路径>`，通常以 `/v1` 结尾）；**不写裸主机域名**，
也不会再替你补 `https://`——`api.deepseek.com` 这种写法会直接报错并提示该写成什么样。
键名统一为 `base_url`（旧拼写 `host` 已下线，写了同样报错），而不是被静默忽略：
忽略会让请求打到内置默认端点。

首次使用：`cp config/glm_glm-5.3-flash.config.example config/glm_glm-5.3-flash.config` 并填入密钥。
注册表里单独绑定过档案的角色还需各自的档案，例如
`cp config/ustc_deepseek-v4-flash.config.example config/ustc_deepseek-v4-flash.config`；
`python -m drsr_420.llm.roles --check` 会把缺哪些档案、该复制哪个模板逐条列出来。

- 文件名的唯一权威是文件内的 `model` 字段（`provider/model` 形式，会做格式校验）；**代码不解析文件名**，写错只会"读不到文件"而不会静默连错模型。
- 支持提供商：`deepseek`、`siliconflow`、`deepinfra`、`ollama`、`blt`（柏拉图）、`cstcloud`（科技云）、`glm`（智谱）；别名（`zhipu`/`bigmodel`/`cst`/`bltcy`…）会自动归一。
- `api_key` 留空时回退对应环境变量（`ZHIPU_API_KEY`、`DEEPSEEK_API_KEY`、`SILICONFLOW_API_KEY` 等）。
- **自定义提供商**：provider 段可以是一个代码从未见过的名字（如 `ustc`），只要档案里给出
  `base_url` 就能用，**不需要改代码**——端点属于"连接谁"，本就归档案管。可照抄
  `config/ustc_deepseek-v4-flash.config.example`。三个可选字段：`api_key_env`（密钥环境变量名，
  缺省按 provider 段派生：`ustc` → `USTC_API_KEY`）、`api_key_required`（本地免鉴权服务写 `false`）、
  `dialect`（请求体方言 `openai`/`glm`/`deepseek`/`ollama`，缺省 `openai`：不认识的一律不发，
  避免 400）。端点独有的私有字段（如 vLLM 的 `chat_template_kwargs`）写进 `extra_body`，
  会原样并入请求体。
- **一次性覆盖**：`--llm_config <档案>` 改的是"默认档案"（未绑定档案的角色共用它）；`--role-config explain=<档案>` 精确覆盖某个角色，`--role-config '*=<档案>'` 强制所有角色。每次实验的解析结果都记进 `config_snapshot.json` 的 `llm.roles`。
- `.idea/runConfigurations/` 的 4 个 MRF 运行配置显式指定 `--llm_config config/glm_glm-5.3-flash.config`。

> 配置层设计（为什么这样分、与 CrewAI/AutoGen 的对照）见
> [`docs/CONFIG_PLAN.md`](docs/CONFIG_PLAN.md)。

## RAG 文献知识库

项目内置 Chroma 持久化向量库（`knowledge_base/chroma_db`），用于检索文献背景注入提示词。

- 配置：`config/rag.config`（嵌入后端 `local`/`api`、API 端点/密钥/模型、分块大小、检索 `k` 等）。
  首次使用：`cp config/rag.config.example config/rag.config`。
- 默认 `backend=api` 走 OpenAI 兼容嵌入接口（如智谱 `embedding-3`、SiliconFlow `BAAI/bge-m3`）；`api_key` 留空时按端点回退环境变量。
- 端点键名与写法两端统一：LLM 档案是 `base_url`，RAG 档案是 `api_base_url`，都必须是
  **完整 URL**（`https://…`，不接受裸主机域名）；旧拼写 `host` / `api_host` 已下线——
  写了会报错并提示改名。`backend=api` 时 RAG 端点缺失同样报错。

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
config/                       # 配置目录（.json 入库 / .config 不入库，扩展名即保密边界）
  agents.config.json          #   ★ 角色 → LLM 档案 + 参数覆盖（唯一声明处，无密钥）
  <提供商>_<模型>.config      #   连接信息 + 生成参数（含 api_key，不入库）
  <提供商>_<模型>.config.example   # 模板（入库）
  rag.config(.example)        #   文献知识库配置
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
    client.py                 #   LLMClient：请求/重试/流式/记账
    adapt.py                  #   请求体方言适配（glm/deepseek/ollama/openai，含自定义提供商）
    providers.py              #   提供商子类 + OpenAICompatClient（任意兼容端点）
    factory.py                #   ClientFactory / 档案定位与归一化（Q1+Q2）
    roles.py                  #   ★ 角色 → 档案 解析（Q3，唯一声明处）
    role_clients.py           #   按角色提供已参数化的客户端（独立克隆 + 按档案缓存）
    role_diagnostics.py       #   角色配置表格渲染与 --check 自检
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
    llm_setup.py              #   档案加载 / 角色客户端池（配置没准备好时给出可操作报错）
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
