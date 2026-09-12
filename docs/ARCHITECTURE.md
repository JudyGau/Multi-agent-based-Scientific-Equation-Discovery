# DrSR 架构说明

> 本文是**唯一权威**的架构文档。文中两处内容由代码生成，请勿手工编辑：
> 「Agent 组织图」来自 `python -m drsr_420.agents`，「分层指标 / 依赖表」来自实测扫描。
> 结构与依赖规则由 [`tests/test_architecture.py`](../tests/test_architecture.py) 守护——
> 文档与代码不一致时，测试会失败。

DrSR（Deep Reasoning Symbolic Regression）用**多个 LLM Agent 协作**发现科学方程：
协调者驱动「采样 → 评估 → 反思 → 持久化」闭环，把 LLM 生成的方程骨架放进常驻沙箱
中执行、多起点拟合参数、按得分分层反馈回提示词，并用多岛经验缓冲维持多样性。

参考论文：Wang et al., *DrSR: LLM based Scientific Equation Discovery with Dual
Reasoning from Data and Experience*, arXiv:2506.04282。

---

## 1. 一分钟看懂运行链路

```
python main.py --problem_name X --data_csv data/X/train.csv
  └─ drsr_420.cli.main.main()
       ├─ runtime.pipeline.main()
       │    ├─ core.buffer.ExperienceBuffer        共享记忆（含断点续跑恢复）
       │    ├─ DataAnalyzerAgent.analyze()          初次数据分析 + RAG 文献注入
       │    └─ 以 Sampler-i 线程并行启动 CoordinatorAgent × N
       └─ analysis.find_best_eq()                   收尾：参数拟合 + 物理解释
```

一轮采样（每个 Sampler 线程独立跑）：

```
CoordinatorAgent.run()  while 未达采样上限/时长上限:
  ├─ buffer.get_prompt()                    取岛 + 聚类软采样出可参考的方程框架
  ├─ SamplerAgent.draw_samples()            拼提示词（指令+任务头+经验/残差注入）→ 骨架
  │    └─ ToolCallerAgent.complete()        与 LLM 多轮对话，按需调用 MCP 文献工具
  ├─ EvaluatorAgent.analyze()               编译 → 沙箱执行 → 多起点 least_squares
  ├─ 分类 Good / Bad / None                 以"评估前该岛最佳分"为基准
  ├─ ExperienceSummarizerAgent.analyze()    逐样本 LLM 分析 → experiences.json
  ├─ ResidualAnalyzerAgent.analyze()        残差统计 + 结构化修正方向 → residual_analyze.json
  └─ checkpoint.json / round_progress.csv   断点续跑与可观测性
```

---

## 2. 分层结构

依赖方向**自底向上**，禁止越界与倒置；规则见 §3 并由测试强制。

| 层 | 位置 | 职责 |
|---|---|---|
| `core` | `drsr_420/core/` | 领域无关基础设施：经验记忆（多岛 + 聚类抽样）、AST 与程序拼装、配置、线程前缀输出、样本/进度记录、提示词模板、全局 token 统计 |
| `llm` | `drsr_420/llm/` | LLM 接入：客户端（重试/流式/参数适配/记账）、提供商子类、客户端工厂与配置、工具调用 schema |
| `evaluation` | `drsr_420/evaluation/` | 评估执行**机制**：多起点拟合打分、常驻子进程沙箱（超时重建）、可选 numba 加速 |
| `knowledge` | `drsr_420/knowledge/` | 外部知识：Chroma RAG 知识库与入库 CLI、MCP 工具（文献检索/阅读）与其 stdio 服务器 |
| `agents` | `drsr_420/agents/` | ★ **多 Agent 角色层**：7 个 Agent + 契约（`base.py`）+ 消息（`messages.py`） |
| `analysis` | `drsr_420/analysis/` | 收尾分析：最优方程的参数拟合与物理解释、敏感度剪枝 |
| `runtime` | `drsr_420/runtime/` | 编排：实验主流程（初始化 → 并行采样 → 收尾） |
| `cli` | `drsr_420/cli/` | 命令行入口：参数解析、输出归档、数据集加载、spec 渲染、产物快照 |

顶层 `drsr_420/*.py`（除 `__init__.py`）**只剩兼容层**：转发到分层子包，供历史脚本与
旧导入路径使用（见 §8）。

### 实测规模

| 层 | 文件 | 行数 | 实际依赖 |
|---|---|---|---|
| `core/` | 8 | 1857 | —（最底层） |
| `llm/` | 6 | 965 | `core` |
| `evaluation/` | 4 | 615 | `core` |
| `knowledge/` | 8 | 1344 | `llm` |
| `agents/` | 11 | 2585 | `core`, `evaluation`, `knowledge`, `llm` |
| `analysis/` | 3 | 1073 | `core`, `knowledge`, `llm` |
| `runtime/` | 2 | 292 | `agents`, `analysis`, `core`, `knowledge` |
| `cli/` | 2 | 456 | `agents`, `core`, `evaluation`, `llm`, `runtime` |

包内实现共约 10,300 行；顶层兼容层 20 个文件。

---

## 3. 依赖规则（硬约束）

```
cli ──▶ runtime ──▶ agents ──▶ evaluation ──▶ core
         │            │  │                        ▲
         │            │  └──▶ knowledge ──▶ llm ───┘
         └──▶ analysis ──▶ core / knowledge / llm
```

允许的依赖集合（`tests/test_architecture.py::_ALLOWED_LAYER_DEPS`）：

| 层 | 允许 import |
|---|---|
| `core` | 无（最底层；唯一的例外是 `config.py` 里 `TYPE_CHECKING` 内的注解引用） |
| `llm` | `core` |
| `evaluation` | `core` |
| `knowledge` | `llm` |
| `agents` | `core` `llm` `evaluation` `knowledge` |
| `analysis` | `core` `llm` `knowledge` |
| `runtime` | `core` `agents` `analysis` `knowledge` |
| `cli` | `core` `llm` `evaluation` `agents` `runtime` |

具体规则：

1. 禁止**倒置**：低层不得 import 高层（`core` 不依赖任何层，是最硬的一条）。
2. 禁止**反向依赖入口**：任何库代码都不得 import `cli`。
3. `if TYPE_CHECKING:` 块内的 import **不计入**依赖方向检查——它运行时不执行，
   只是类型注解引用（例如 `core/config.py` 注解 `agents`/`evaluation` 的类型）。
   函数体内的 import **计入**（延迟导入同样是依赖）。
4. 新增一层需要在 `_LAYERS` / `_ALLOWED_LAYER_DEPS` 里显式登记，并说明理由。

> 重构过程中正是靠"先实测、再固化规则"发现了两处真实倒置（`core → llm`、
> `agents → runtime`），并据此把 token 统计下沉到 `core/llm_stats.py`、把评估执行
> 子系统提升为独立的 `evaluation` 层。详见
> [`REFACTOR_PLAN.md`](./REFACTOR_PLAN.md) 的阶段 3 记录。

---

## 4. 多 Agent 系统

### 4.1 组织图（由 `python -m drsr_420.agents` 生成）

```
DRSR 多 Agent 系统（7 个角色）
════════════════════════════════════════════════════════════════════════
编排层（非 Agent）pipeline.main()：初始化共享记忆 → 启动 Sampler-i 线程
────────────────────────────────────────────────────────────────────────
[coordinator] 协调者  (per-sampler-thread  LLM=sampling)
每轮从共享记忆取 prompt，驱动「采样→评估→反思→持久化」主循环
├─→ [sampler] 采样者  (per-sampler-thread  LLM=sampling)
    拼提示词（指令+任务头+经验/残差注入）生成方程骨架，空骨架自动重采样
│   └─→ [tool_caller] 工具调用者  (per-sampler-thread  LLM=sampling)
        与 LLM 多轮对话并执行其发起的 MCP 工具调用，直到给出最终答复
├─→ [evaluator] 评估者  (per-sampler-thread)
    编译骨架、在常驻沙箱中执行并多起点拟合打分，产出 (score, error, residual)
├─→ [experience_summarizer] 经验总结者  (per-sampler-thread  LLM=experience)
    对 Good/Bad/None 样本逐个做 LLM 分析，产出改进建议
└─→ [residual_analyzer] 残差分析者  (per-sampler-thread  LLM=residual)
    统计残差并结合上一次分析，让 LLM 输出结构化修正方向
────────────────────────────────────────────────────────────────────────
编排层直接驱动的单次角色（非 per-sampler 线程）
[data_analyzer] 数据分析者  (single-shot  LLM=analysis)
实验开始时对数据集做初次分析，产出 sample_order=0 的残差分析基线
────────────────────────────────────────────────────────────────────────
收尾（非 Agent）：find_best_eq() —— 参数拟合 + 物理解释
```

### 4.2 每个 Agent 的角色卡

角色卡是**代码**（`agents/base.py` 的 `AgentSpec`），不是文档：每个 Agent 用类属性
`SPEC` 声明自己的 key / 角色 / 使命 / 入口方法 / 上下游 / 输入输出契约 / 落盘产物 /
线程模型 / 使用的 LLM 客户端副本。`BaseAgent.__init_subclass__` 在**类定义时**校验它，
因此"漏声明"或"改名忘同步"会在 import 阶段直接报错。

| key | 类 | 入口 | 输入 → 输出 | 落盘产物 |
|---|---|---|---|---|
| `coordinator` | `CoordinatorAgent` | `sample(profiler)` | `buffer.Prompt` → `SampleBatch` | `checkpoint.json`、`round_progress.csv`、`experiences.json`、`residual_analyze.json` |
| `sampler` | `SamplerAgent` | `draw_samples(prompt, config)` | 提示词 + Config → 骨架 list + 思考 list | — |
| `tool_caller` | `ToolCallerAgent` | `complete(content, repeat)` | content + 次数 → 响应 list + 思考 list | — |
| `evaluator` | `EvaluatorAgent` | `analyze(EvaluationRequest)` | 评估请求 → `EvaluationOutcome` | `samples/samples_N.json`（经 Profiler） |
| `experience_summarizer` | `ExperienceSummarizerAgent` | `analyze(samples, qualities, errors, prompt)` | 一批样本 + 质量标签 → `list[ExperienceEntry]` | `experiences.json`（由协调者写） |
| `residual_analyzer` | `ResidualAnalyzerAgent` | `analyze(sample, residual)` | 最优样本 + 残差矩阵 → `ResidualInsight` | `residual_analyze.json`（由协调者写） |
| `data_analyzer` | `DataAnalyzerAgent` | `analyze(data_source, prompt)` | 数据集 → 分析文本 | `residual_analyze.json`（`sample_order=0`） |

自检与枚举：

```bash
python -m drsr_420.agents            # 打印组织图（由 SPEC 渲染）
python -m drsr_420.agents --check    # 契约自检：上下游引用、可达性、线程模型一致性
python -c "from drsr_420.agents import agent_specs; print(agent_specs())"
```

### 4.3 Agent 之间的消息

`agents/messages.py` 把"跨 Agent 传什么"从裸元组 / `**kwargs` 袋 / 平行列表对齐，
变成显式类型：

| 类型 | 流向 | 说明 |
|---|---|---|
| `EvaluationRequest` | coordinator → evaluator | 样本、岛屿/版本号、记录用归属信息（profiler / 全局采样数 / 采样耗时） |
| `EvaluationOutcome` | evaluator → coordinator | `score` / `error` / `residual`；`to_legacy()` 与迭代支持旧式解包 |
| `ExperienceEntry` | summarizer → coordinator → 磁盘 | `to_json()` 唯一定义 `experiences.json` 的字段 |
| `ResidualInsight` | residual_analyzer → coordinator → 磁盘 | `to_json()` 唯一定义 `residual_analyze.json` 的字段 |
| `ToolCall` | tool_caller 内部记录 | 工具名 / 参数 / 结果 / 错误 |
| `SampleBatch` | coordinator 一轮内的容器 | prompt、骨架、分数、错误、质量、经验条目、本轮最优 |

约定：**分析类 Agent 只填"内容"字段**（样本、分析文本），**归属字段**（岛屿、样本
顺序号、分数）由 Coordinator 在落盘前补齐——这样每个字段的来源都能一眼指认。

---

## 5. 共享状态与并发模型

| 状态 | 保护方式 | 说明 |
|---|---|---|
| `ExperienceBuffer`（多岛经验 + 聚类） | 内部 `threading.RLock` | 所有 Sampler 线程共享 |
| 全局采样计数 `CoordinatorAgent._global_samples_nums` | 模块级可重入锁 `_SAMPLER_LOCK` | 类属性；断点恢复时由 `set_global_sample_nums` 校准 |
| `experiences.json` / `residual_analyze.json` / `checkpoint.json` | 同上（读-改-写整段持锁） | 写入走 `atomic_write_json`（临时文件 + `os.replace`） |
| `EvaluatorAgent` 实例 | **每个 Sampler 线程独享一份** | 避免 `LocalSandbox._last_params`（热启动参数）等实例状态竞态 |
| `LocalSandbox` worker 池 | 任务锁 + 队列 | 串行调度；超时/崩溃时**换一条新队列**并重建 worker（陈旧任务结构上不可达） |
| LLM 客户端 | 按任务克隆独立副本 | 采样 / 经验 / 残差各自独立温度与思考强度，互不覆盖；全局 token 统计带锁累加 |
| `Profiler` 记录 | 内部锁 | 多线程并行时样本数上限判断在锁内 |

线程模型：`pipeline.main` 以 `Sampler-i` 线程启动 N 个 `CoordinatorAgent`（daemon），
每个线程内部再驱动自己的下游 Agent；LLM 调用是阻塞 IO，因此线程并行即可获得吞吐。
`Ctrl-C` 会给每个采样线程最多 60s 收尾当前轮次的 checkpoint 再退出。

---

## 6. 产物清单

一次实验的全部产物在 `experiments/{problem}_{时间戳}/`：

| 文件 | 写入者 | 说明 |
|---|---|---|
| `run.out` / `run.err` | `cli.main.setup_output_tee` | 标准输出/错误的双写归档 |
| `spec_dynamic.txt` | `cli.main.save_dynamic_spec` | 本次动态渲染的 spec（复现用） |
| `config_snapshot.json` | `cli.main.save_config_snapshot` | 超参 + LLM 配置（api_key 打码） |
| `experiences.json` | `CoordinatorAgent._persist_experiences` | 按 `Good/Bad/None` 分类的经验条目 |
| `residual_analyze.json` | `CoordinatorAgent._persist_residual`、`DataAnalyzerAgent` | 残差分析记录（`sample_order=0` 为初次分析） |
| `checkpoint.json` | `CoordinatorAgent._save_checkpoint` | 经验缓冲 + 全局采样数（断点续跑） |
| `round_progress.csv` | `CoordinatorAgent._append_progress` | 每轮进度（墙钟、岛屿、最佳分、采样数） |
| `marker…` / `progress.json` / `best_history/` | `Profiler` | 采样进度与历史最优 |
| `samples/samples_N.json` | `Profiler._write_json` | 单样本记录：`iteration`/`sample_order`/`nmse`/`mse`/`score`/`function`/`params` |
| `explain.txt` | `analysis.find_best_eq` | 最优方程的物理解释（若启用） |

---

## 7. 扩展指南

**新增一个 Agent**

1. 在 `drsr_420/agents/` 下新建 `<role>_agent.py`，继承 `BaseAgent` 并声明 `SPEC`；
2. 需要跨 Agent 传数据时，在 `agents/messages.py` 里加类型，不要在签名里塞 `**kwargs`；
3. 若它属于"每个 Sampler 线程一个实例"，把 `SPEC.thread_model` 设为
   `THREAD_PER_SAMPLER`，并在 `coordinator.downstream` 里登记（自检会强制检查可达性）；
4. 把它加入 `agents/__init__.py` 的 `_AGENT_CLASSES` 与 `AGENT_ORDER`；
5. 跑 `python -m drsr_420.agents --check` 与 `pytest`/`tests/run_tests.py`。

**新增一个模块**

先判断它属于哪一层：与领域无关的基础设施 → `core`；LLM 通信 → `llm`；执行机制 →
`evaluation`；外部数据源 → `knowledge`；协调角色 → `agents`；收尾分析 → `analysis`；
编排 → `runtime`。**不要放回 `drsr_420/` 顶层**（`LayerLayoutTest` 会失败）。

**新增一层**

在 `tests/test_architecture.py` 的 `_LAYERS` 与 `_ALLOWED_LAYER_DEPS` 中登记，并在本节
说明理由——分层规则是显式清单，不允许隐式放宽。

---

## 8. 兼容层

历史导入路径全部保留为**转发层**（20 个顶层 shim + 6 个 Agent shim），旧脚本与旧测试
无需改动：

| 旧路径 | 新路径 |
|---|---|
| `drsr_420.buffer` / `code_manipulation` / `config` / `console` / `profile` / `prompt_config` | `drsr_420.core.*` |
| `drsr_420.evaluate_on_problems` / `evaluator_accelerate` | `drsr_420.evaluation.problems` / `.accelerate` |
| `drsr_420.rag_kb` / `rag_build` / `tool_runner` / `tools.*` | `drsr_420.knowledge.*` / `drsr_420.knowledge.tools.*` |
| `drsr_420.tools.tools_description` | `drsr_420.llm.tools_schema` |
| `drsr_420.find_best_eq` / `sensitivity_prune` | `drsr_420.analysis.*` |
| `drsr_420.pipeline` | `drsr_420.runtime.pipeline` |
| `drsr_420.sampler` / `evaluator` / `tool_caller` / `experience_summarizer` / `residual_analyzer` / `data_analyse_real` | `drsr_420.agents.*` |
| `llm`（根模块） | `drsr_420.llm` |
| `main`（根脚本） | `drsr_420.cli.main` |

兼容层的行为约定（`tests/test_architecture.py` 逐条断言）：

* **读**：模块级 `__getattr__` 转发所有名字，含私有名（`llm._post_with_retry` 等）；
* **写 / 删**：转发到**定义该名字的子模块**而不是门面——否则
  `mock.patch('llm._post_with_retry')` 的打桩会落在门面上，而 `LLMClient.chat` 在
  `client` 模块的全局命名空间里查找，桩等于没打（这是重构中实测踩到的坑）；
* `-m` 可执行的模块（`mcp_server` / `rag_build` / `main`）额外转发 `main()`，
  保证 `python -m 旧路径` 仍能启动；
* 每个 shim 的模块名与规范模块指向**同一对象**（`is` 比较），不允许复制成副本。

删除时机：当外部脚本（`.idea/runConfigurations/*.xml`、`example.sh`、
`MRFCompress-3.sh`）与历史测试全部改到新路径后，可逐层删除；删除前请确认没有
"用户尚未察觉的依赖"。

---

## 9. 验证与自检

```powershell
# 全量测试（Windows 下必须这样启动：LocalSandbox 用 spawn，需 __main__ 保护）
$env:ZHIPU_API_KEY='dummy-test-key'
& .venv2\Scripts\python.exe tests\run_tests.py

# 多 Agent 契约自检 + 组织图
python -m drsr_420.agents --check
python -m drsr_420.agents

# 命令行入口（旧路径 shim 与规范路径都应可用）
python main.py --help
python -m drsr_420.cli.main --help          # 等价
python -m drsr_420.knowledge.rag_build --help   # 新路径
python -m drsr_420.rag_build --help             # 旧路径（兼容层）
```

架构护栏（`tests/test_architecture.py`）覆盖：

| 检查 | 说明 |
|---|---|
| 兼容层对象同一性 | 旧路径与规范路径必须 `is` 同一对象（防 shim 分叉成副本） |
| 分层目录 | 8 层子包都存在且带 `__init__.py`；顶层只允许兼容层 |
| 依赖方向 | 禁止越界/倒置/库代码依赖 `cli`（AST 扫描全部运行时 import） |
| `__file__` 路径锚点 | 仓库根锚点与独立运行的 `sys.path` 兜底必须指对（用子进程实测） |
| Agent 契约 | 7 个 Agent 都继承 `BaseAgent`、`SPEC` 自洽、规范入口不用英式拼写 |
| 评估子系统边界 | 角色文件不得再含进程池痕迹；机制符号仍可从历史路径导入 |
