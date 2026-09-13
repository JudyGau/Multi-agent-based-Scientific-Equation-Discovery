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
python -m drsr_420.cli.main --problem_name X --data_csv data/X/train.csv
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
| `llm` | `drsr_420/llm/` | LLM 接入：客户端（重试/流式/记账）、**请求体方言适配**（`adapt.py`）、提供商子类（含通用 `OpenAICompatClient`）、客户端工厂与档案定位、**角色 → 档案 解析**（`roles.py` / `role_clients.py` / `role_diagnostics.py`）、工具调用 schema |
| `evaluation` | `drsr_420/evaluation/` | 评估执行**机制**：多起点拟合打分、常驻子进程沙箱（超时重建）、可选 numba 加速 |
| `knowledge` | `drsr_420/knowledge/` | 外部知识：Chroma RAG 知识库与入库 CLI、MCP 工具（文献检索/阅读）与其 stdio 服务器 |
| `agents` | `drsr_420/agents/` | ★ **多 Agent 角色层**：7 个 Agent + 契约（`base.py`）+ 消息（`messages.py`）+ 层内部件（`skeleton.py` / `prompt_injection.py`） |
| `analysis` | `drsr_420/analysis/` | 收尾分析：最优方程的解析/解释/剪枝/可视化（`find_best_eq` 只做编排） |
| `runtime` | `drsr_420/runtime/` | 编排：实验主流程（初始化 → 并行采样 → 收尾） |
| `cli` | `drsr_420/cli/` | 命令行入口：参数解析、输出归档、数据集加载、spec 渲染、产物快照 |

`drsr_420/` 顶层只有 `__init__.py`：**实现全部在分层子包里**，历史的一层平铺路径已清退
（见 §8）。仓库根目录同样不再有任何 `.py` 模块——命令行入口是
`python -m drsr_420.cli.main`（或安装后的 `drsr420` 命令），因此 `drsr_420/` 之外
没有任何代码需要被导入，包是自包含的。

### 实测规模

| 层 | 文件 | 行数 | 实际依赖 |
|---|---|---|---|
| `core/` | 8 | 1857 | —（最底层） |
| `llm/` | 10 | 2027 | `core` |
| `evaluation/` | 4 | 615 | `core` |
| `knowledge/` | 8 | 1359 | `llm` |
| `agents/` | 13 | 2656 | `core`, `evaluation`, `knowledge`, `llm` |
| `analysis/` | 9 | 1304 | `core`, `knowledge`, `llm` |
| `runtime/` | 2 | 299 | `agents`, `analysis`, `core`, `knowledge` |
| `cli/` | 3 | 542 | `agents`, `core`, `evaluation`, `llm`, `runtime` |

包内实现共 10,733 行；顶层只剩 `__init__.py`（0 行实现）。全局最长文件 `llm/client.py` 491 行。

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

## 8. 兼容层（已清退）

分层迁移期间曾在 `drsr_420/` 顶层保留 24 个转发模块（20 个顶层 shim + `tools/` 下 4 个），
让旧导入路径继续可用。**这些转发层已按本节的时机全部删除**，现在只有规范路径：

| 曾经的旧路径 | 现在的规范路径 |
|---|---|
| `drsr_420.buffer` / `code_manipulation` / `config` / `console` / `profile` / `prompt_config` | `drsr_420.core.*` |
| `drsr_420.evaluate_on_problems` / `evaluator_accelerate` | `drsr_420.evaluation.problems` / `.accelerate` |
| `drsr_420.rag_kb` / `rag_build` / `tool_runner` / `tools.*` | `drsr_420.knowledge.*` / `drsr_420.knowledge.tools.*` |
| `drsr_420.tools.tools_description` | `drsr_420.llm.tools_schema` |
| `drsr_420.find_best_eq` / `sensitivity_prune` | `drsr_420.analysis.*` |
| `drsr_420.pipeline` | `drsr_420.runtime.pipeline` |
| `drsr_420.sampler` / `evaluator` / `tool_caller` / `experience_summarizer` / `residual_analyzer` / `data_analyse_real` | `drsr_420.agents.*` |

**删除时机（当初写下的判据，现已满足）**：外部脚本与历史测试全部改到新路径。
具体是：`tests/` 下的旧路径 import 全部迁移完毕，`.idea/runConfigurations/*.xml`、
`example.sh`、`MRFCompress-3.sh` 也都改为 `python -m drsr_420.cli.main`，
`_POST_WRITE`（写/删转发）只在兼容层内部被用到——于是转发层成为纯负债。

### 8.1 仓库根的两个顶层模块（阶段 7，已删除）

包内转发层清退后，根目录还留着 `main.py`（命令行入口）与 `llm.py`（旧模块名的
再导出）。它们当时的保留理由是"项目的对外接口"，但这份"接口"完全可以用 `-m` 表达，
代价却是仓库根必须一直待在 `sys.path` 上：包不自包含，`import llm` 还可能被 PyPI 上的
同名包劫持。因此两者一并删除，全部引用点同步改为模块方式：

| 曾经的用法 | 现在的调用方式 | 已同步的引用点 |
|---|---|---|
| `python main.py ...` | `python -m drsr_420.cli.main ...` | 4 个 IDE 运行配置、`example.sh`、`MRFCompress-3.sh`、`tests/test_console_encoding.py` |
| `import llm` / `python llm.py` | `from drsr_420 import llm` / `python -m drsr_420.llm` | 包内代码与 `tests/` 全部走规范路径 |

IDE 运行配置的改法：`.idea/runConfigurations/*.xml` 里把 `SCRIPT_NAME` 从
`$PROJECT_DIR$/main.py` 改成模块名 `drsr_420.cli.main`，并置 `MODULE_MODE=true`；
`WORKING_DIRECTORY` 仍是 `$PROJECT_DIR$`，所以 `--data_csv ./data/…` 与
`--llm_config llm.config` 这些相对路径不受影响。命令行参数与产物文件名全程未变。

**迁移期间踩过、值得记住的坑**（都不再需要，但改回"门面转发"就会重新踩）：

* **只做读转发的 shim 会让 `mock.patch` 静默失效**：桩落在门面上，而
  `LLMClient.chat` 在 `drsr_420/llm/client.py` 的全局命名空间里查找 `_post_with_retry`
  ——所以现在测试直接打 `drsr_420.llm.client._post_with_retry`；
* **`-m` 可执行的模块必须转发 `main()`**，否则 `python -m 旧路径` 静默起不来
  （MCP 客户端要等 120s 超时才会发现）；
* **新旧路径必须是同一对象**（`is` 比较），复制成副本会让"改一处生效另一处不生效"。

`tests/test_architecture.py` 反向守护这两件事：`LegacyPathRemovalTest` 管**包内**旧路径
（导入必须失败、`drsr_420/` 顶层只允许 `__init__.py`、源码不得再 import 已删除路径），
`RootEntrypointRemovalTest` 管**仓库根**（根目录不得再有 `.py`、`main`/`llm` 不得从仓库根
解析、源码不得 import 这两个顶层名）。

---

## 9. 验证与自检

```powershell
# 全量测试（Windows 下必须这样启动：LocalSandbox 用 spawn，需 __main__ 保护）
$env:ZHIPU_API_KEY='dummy-test-key'
& .venv2\Scripts\python.exe tests\run_tests.py

# 多 Agent 契约自检 + 组织图
python -m drsr_420.agents --check
python -m drsr_420.agents

# 角色 → LLM 档案（唯一声明处：config/agents.config.json）
python -m drsr_420.llm.roles
python -m drsr_420.llm.roles --check      # 档案存在 / model 合法 / 密钥可达
python -m drsr_420.llm.roles --profiles --templates   # 本机已有档案 / 随仓库模板

# 命令行入口（两者等价；安装后还可用 console script `drsr420`）
python -m drsr_420.cli.main --help
drsr420 --help

# 知识库 CLI 与语义剪枝演示
python -m drsr_420.knowledge.rag_build --help
python -m drsr_420.analysis.prune_demo
```

架构护栏（`tests/test_architecture.py`）覆盖：

| 检查 | 说明 |
|---|---|
| 旧路径已清退 | 已删除的包内路径导入必须失败、顶层只留 `__init__.py`、源码不得再 import 旧路径 |
| 仓库根已清空 | 根目录不得再有 `.py` 模块；`main` / `llm` 不得从仓库根解析；源码不得 import 这两个顶层名 |
| 分层目录 | 8 层子包都存在且带 `__init__.py`，且每层都有实现（空层是"分层被掏空"的信号） |
| 依赖方向 | 禁止越界/倒置/库代码依赖 `cli`（AST 扫描全部运行时 import） |
| `__file__` 路径锚点 | 仓库根锚点与独立运行的 `sys.path` 兜底必须指对（用子进程实测） |
| Agent 契约 | 7 个 Agent 都继承 `BaseAgent`、`SPEC` 自洽、规范入口不用英式拼写 |
| 评估子系统边界 | 角色文件不得再含进程池痕迹；机制符号仍可从角色模块导入（同对象） |
| 控制台编码 | 以 `PYTHONIOENCODING=gbk` 跑典型入口，打印内容必须能被 GBK 编码 |
| 配置无硬编码档案名 | 库代码里不得出现写死的 `*.config` 文件名（须经角色解析）；注册表无密钥；每个已建档案都要有入库模板 |

行为测试（`tests/test_sampler_agent.py`、`tests/test_agent_behavior.py`）覆盖各 Agent
的**行为**而非结构：骨架提取与重采样上界、经验/残差注入策略、工具循环收敛、
反思失败兜底、协调者的质量判定与归属字段计算。

---

## 10. 配置：角色 → 档案

一份"配置"其实在回答三个寿命与保密等级都不同的问题，本系统按此把它们**分成三层**：

| | 问题 | 在哪 | 是否入库 |
|---|---|---|---|
| Q1 | 连接谁？用哪把钥匙？ | `config/<提供商>_<模型>.config` 的 `base_url` / `api_key` | ✗（`*.config` 被 `.gitignore` 忽略） |
| Q2 | 生成参数是什么？ | 同一档案的 `model` / `temperature` / `max_tokens` … | 随 Q1 同文件 |
| Q3 | **哪个角色用哪套？覆盖什么？** | `config/agents.config.json` | ✓（无密钥，需 review） |

**扩展名本身就是保密边界**：`.json` 可入库、`.config` 不入库、`.config.example` 是模板。

键名同样定死：**端点统一叫 `base_url`**（RAG 档案里是 `api_base_url`），且必须是
**完整 URL**（`https://<主机>/<路径>`）——**不接受裸主机域名**，也不会再替你补
`https://`。旧拼写 `host` / `api_host` 已下线，出现即报错并提示改名，不做静默兼容：
内置提供商自带默认端点，忽略旧键会让请求悄悄打到默认地址而不是用户写的那一个。
这条规则只在客户端构造处守一次（`llm/client.require_absolute_url`），因此配置、
内置默认值、环境变量兜底（如 `ZHIPU_API_BASE`）三条通道都覆盖到了。

### 10.1 六个角色

| 角色 | 使用者 |
|---|---|
| `sampling` | SamplerAgent + ToolCallerAgent |
| `analysis` | DataAnalyzerAgent |
| `experience` | ExperienceSummarizerAgent |
| `residual` | ResidualAnalyzerAgent |
| `explain` | `analysis/explain.py`（非 Agent） |
| `summary` | MCP 工具 `read_paper`（非 Agent，跑在子进程里） |

角色表是代码常量（`llm/roles.py::TASKS`），`AgentSpec.llm_task` 必须落在其中（有护栏）。

随仓库分发的绑定：`explain` → 官方 DeepSeek 端点，`summary` → 校内网关（走下面的
自定义提供商路径），其余四个角色共用 `default`。两个角色之所以单独绑定，是因为
"给某个角色换模型"在旧结构里**原理上无法表达**（见 §8）。

### 10.2 解析优先级

```
--role-config <role>=<file>            (最高：命令行精确覆盖，'*' 表示所有角色)
  > 环境变量 DRSR_ROLE_CONFIG_<ROLE>   (容器 / CI / 子进程传递)
  > agents.config.json 的 roles.<role>.config
  > --llm_config                       (CLI 指定的**默认**档案)
  > agents.config.json 的 default
  > 内置 DEFAULT_PROFILE               (最低：保证永不"无配置")
```

第 3 步高于第 4 步是刻意的：`--llm_config` 的语义是"**默认**档案"（未绑定档案的角色
共用它）。若反过来，IDE 运行配置里那句 `--llm_config` 会把注册表的角色绑定永久屏蔽
——**这正是旧结构下"给 explain 换模型"无法生效的根因**。

参数按角色合并：`内置 BUILTIN_ROLE_PARAMS` < `档案 tasks[role]`（旧格式，兼容）<
`注册表 roles[role].params`。

### 10.3 子进程与可追溯

`read_paper` 跑在 MCP server 子进程里，拿不到父进程的对象，因此
`cli.main` 会把每个角色的解析结果**回写环境变量** `DRSR_ROLE_CONFIG_<ROLE>`，
`tool_runner._server_env()` 再把它并入子进程环境（`mcp` SDK 默认只透传系统级白名单）。
每次实验的解析结果都写进 `config_snapshot.json` 的 `llm.roles`，让"这次实验的 explain
到底用了哪个模型"可查。

### 10.4 一条硬约束：代码不得解析文件名

档案文件名的唯一权威是文件内的 `model` 字段（`ClientFactory` 校验 `provider/model`
格式），文件名只是给人看的标签。因此文件名写错**不会**静默连错模型，只会让"读不到
文件"尽早暴露。护栏（`tests/test_config_roles.py`）直接扫描库代码里的字符串字面量：
出现写死的 `*.config` 文件名即失败——这条堵死的正是"explain 硬编码了一个不存在的
档案、异常被吞、`explain.txt` 长期为空"那类故障的成因。

### 10.5 自定义提供商：端点写在档案里，不是代码里

内置提供商（`deepseek` / `siliconflow` / `deepinfra` / `ollama` / `blt` / `cstcloud` / `glm`）
把"默认 base_url + 密钥环境变量名"写在代码表 `ClientFactory._PROVIDER_SPECS` 里。
provider 段**不在**这张表里时不再报错，而是走**自定义提供商**路径——只要档案给了
`base_url` 就照常构造（`OpenAICompatClient`，`llm/providers.py`）。这仍然是 §10 的
分工：端点属于 Q1，本就归档案；塞进代码等于"接一个校内网关"要改代码 + 重新 review
客户端工厂。

```json
{
  "base_url": "https://api.llm.ustc.edu.cn/v1",
  "api_key": "",
  "model": "ustc/deepseek-v4-flash",
  "max_tokens": 65536
}
```

| 字段 | 作用 | 缺省 |
|---|---|---|
| `base_url` | 端点；自定义提供商**必填**。必须是完整 URL（`https://<主机>/<路径>`，不接受裸主机域名）；旧拼写 `host` 已下线，写了报错 | 内置提供商用自己的默认值 |
| `api_key_env` | 密钥环境变量名 | 按 provider 段派生（`ustc` → `USTC_API_KEY`） |
| `api_key_required` | 是否强制要求密钥 | 内置表的约定；自定义提供商默认 `true`（本地免鉴权写 `false`） |
| `dialect` | 请求体方言：`openai` / `glm` / `deepseek` / `ollama` | `openai`（不认识的一律不发） |

**方言**（`llm/adapt.py`）是"发给谁时字段长什么样"的唯一定义处：同一个
`reasoning_effort`，智谱要配 `thinking` 开关、DeepSeek 直通、Ollama 要变成 `think`
布尔、纯 OpenAI 兼容端点则必须**不发**（否则 400）。自定义提供商默认落在最后一种；
若其端点恰好与某个已知家族一致，写一行 `"dialect": "deepseek"` 就能复用该分支，
不必改代码。拼错的方言会直接报错、不静默降级——静默降级等于"配了却没生效"。

端点上仍有独有的私有字段时（如 vLLM 的 `chat_template_kwargs`），用 `extra_body`
原样并入请求体：它由 `_build_payload` **最后**并入，所以方言规则不会把它删掉。
（此前它是个死字段：工厂不注入、适配层又 pop 掉，模板里写了也从没上过线。）

护栏：`tests/test_llm_custom_provider.py` 锁住"未知 provider + base_url 即可用"与密钥
解析规则；`tests/test_config_roles.py` 要求**每个模板填上占位密钥后都能构造出客户端**，
并跳过不含 `model` 的模板（`rag.config` 是知识库配置，不是 LLM 档案）。

设计与迁移过程见 [`CONFIG_PLAN.md`](./CONFIG_PLAN.md)。
