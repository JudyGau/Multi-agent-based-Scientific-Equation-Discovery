# DrSR 重构方案：让「多 Agent 架构」一眼可见

| 项 | 值 |
|---|---|
| 文档 | `docs/REFACTOR_PLAN.md` |
| 状态 | **已执行完成（阶段 0–5 全部落地）**，执行记录与实测差异见 §11 |
| 目标 | 让代码结构直接体现多 Agent 协作，且整体架构清晰易懂 |
| 约束 | **不改变算法行为**、不引入新依赖、旧导入路径全部保留、既有测试保持通过（基线 217 项） |
| 规模 | 重构前：源码 9,375 行（`drsr_420/` + `tests/`）／7 个 Agent／21 个顶层模块 |
| 配套文档 | [`ARCHITECTURE.md`](./ARCHITECTURE.md)（权威架构说明，实施后新增） |

---

## 执行状态

| 阶段 | 内容 | 提交 | 结果 |
|---|---|---|---|
| 0 | 基线与结构护栏 | `0df720b` | 217 → 221 测试通过 |
| 1 | 统一 Agent 契约（角色卡 + 基类 + 组织图） | `37611da` | 221 → 229 |
| 2 | 显式消息契约（替代裸元组/kwargs 袋） | `7d2d6b1` | 229 → 248 |
| 3 | 8 层目录结构 + 兼容层 | `aff876d` | 248 → 259 |
| 4 | 评估执行子系统拆分 | `54559cc` | 259 → 262 |
| 5 | 文档、死代码清理、打包元数据 | `fba8691` | 262 → 262 |

---

## 0. 结论（TL;DR）

项目**已经**是一个多 Agent 系统：`drsr_420/agents/` 下 7 个角色、`CoordinatorAgent` 统一编排、
`agents/README.md` 有完整角色手册。问题不在"没有多 Agent"，而在**"多 Agent 没有被代码结构表达出来"**：

1. **没有统一 Agent 契约** —— 7 个 Agent 是 7 个各自为政的类，入口方法名互不相同
   （`sample` / `draw_samples` / `complete` / `analyse` / `analyze` × 3），
   连 `analyse` 与 `analyze` 拼写都不一致；没有任何基类、没有注册表、无法枚举"系统里有哪些 Agent"。
2. **Agent 之间的接口是隐式的** —— 靠裸元组 `(score, error, residual)` 和 `**kwargs` 袋传数据，
   协作关系只能靠读 445 行 `CoordinatorAgent` 反推。
3. **分层缺失** —— `agents/` 之外，21 个模块平铺在 `drsr_420/` 顶层：
   框架原语（buffer/config/console/profile）、LLM 接入（根目录 `llm.py`）、
   RAG、收尾分析、6 个兼容 shim 混在一起，看不出层次。
4. **两个编排层分散两处** —— `pipeline.py`（顶层）与 `CoordinatorAgent`（agents 子包）分居两地。
5. **评估子系统职责过载** —— `agents/evaluator_agent.py`（516 行）同时装着
   AST visitor、`Sandbox` 抽象、`LocalSandbox` 进程池、worker 函数、程序拼装、拟合逻辑、Agent 本体。

**方案**：四个支柱（统一契约 / 显式消息 / 分层目录 / 文档可视化），分 6 个阶段、每阶段一个可独立回滚的提交。
推荐执行 **全部阶段**；若只想降低风险，可只做 **阶段 0–2 + 5**（不动文件位置，收益已占约 60%）。

---

## 1. 现状诊断（证据均为实测）

### 1.1 分层混杂：21 个 `.py` 文件平铺（15 个实现 + 6 个兼容 shim）

```
drsr_420/
  # —— 框架原语 ——
  config.py(102)  console.py(97)  code_manipulation.py(292)  buffer.py(442)  profile.py(366)
  # —— 提示词资产 ——
  prompt_config.py(445)
  # —— 编排 ——
  pipeline.py(288)
  # —— 评估执行 ——
  evaluate_on_problems.py(164)  evaluator_accelerate.py  parallel_bfgs.py(42)
  # —— 文献 / RAG ——
  rag_kb.py(469)  rag_build.py  tool_runner.py(183)  tools/{mcp_server,search_paper,read_paper,tools_description}.py
  # —— 收尾分析 ——
  find_best_eq.py(557)  sensitivity_prune.py(444)
  # —— 兼容 shim（6 个） ——
  sampler.py  evaluator.py  tool_caller.py  experience_summarizer.py  residual_analyzer.py  data_analyse_real.py
  # —— 多 Agent ——
  agents/{coordinator,sampler,tool_caller,evaluator,experience_summarizer,residual_analyzer,data_analyzer}_agent.py
```

**只有 `agents/` 一个子包**。读代码时必须靠文件名猜层，没有任何结构性信号告诉读者
"哪些是 Agent、哪些是 Agent 用的基础设施、哪些是历史包袱"。

### 1.2 包结构不完整（实测）

| 路径 | `__init__.py` | 说明 |
|---|---|---|
| `drsr_420/` | ❌ 不存在 | 隐式命名空间包：无 `__version__`、无包门面、无导出约定 |
| `drsr_420/agents/` | ✅ 存在 | 但仅 31 行文档字符串，**刻意不 import**（防循环导入） |
| `drsr_420/tools/` | ❌ 不存在 | 隐式命名空间包 |

三者策略不一致；`drsr_420` 作为顶层包连版本号都没有。

### 1.3 多 Agent 缺少统一契约（核心问题）

实测 7 个 Agent 的公开入口：

| Agent | 入口方法 | 输入 | 输出 |
|---|---|---|---|
| `CoordinatorAgent` | `sample(**kwargs)` | kwargs 袋 | 无（副作用落盘） |
| `SamplerAgent` | `draw_samples(prompt, config)` | str + Config | `(list, list) \| None` |
| `ToolCallerAgent` | `complete(content, repeat)` | str + int | `(list, list)` |
| `EvaluatorAgent` | **`analyse`**(sample, island_id, version_generated, **kwargs) | 4+ 参数 + kwargs 袋 | `(score, error, residual)` 裸元组 |
| `ExperienceSummarizerAgent` | `analyze(samples, quality, error, prompt)` | 4 个平行 list | `list[str]` |
| `ResidualAnalyzerAgent` | `analyze(sample, residual)` | str + ndarray | `str` |
| `DataAnalyzerAgent` | `analyze(data_source, custom_prompt, max_rows, verbose)` | 4 参数 | `str` |

具体问题：

- **`analyse` vs `analyze`**：唯一的英式拼写出现在 `EvaluatorAgent`。调用方 `CoordinatorAgent._evaluate_batch`
  与 `pipeline._run_initial_analysis` 都得记住这个例外。
- **无基类、无注册表**：`class CoordinatorAgent:`（`coordinator_agent.py:100`）等 7 个类都是
  隐式继承 `object`。没有任何地方能回答"系统里有几个 Agent"——除了一份手写 Markdown。
- **`LLM` 抽象基类放错层**：`class LLM(ABC)` 定义在 `agents/sampler_agent.py:35`，
  它是"采样后端"抽象，却与根目录 `llm.py` 的 `LLMClient` 形成两个都叫 "LLM" 的概念，
  且 `SamplerAgent` 继承它——采样 Agent 与采样后端抽象耦合在一个文件里。
- **`**kwargs` 袋穿透多层**：`CoordinatorAgent._evaluate_batch(**kwargs)` →
  `EvaluatorAgent.analyse(**kwargs)` → 沙箱任务。契约完全不可见，且有跨任务参数泄漏隐患。

### 1.4 评估子系统职责过载

`agents/evaluator_agent.py`（516 行）一个文件里装着 **6 类职责**：

| 行号 | 内容 | 实际所属层 |
|---|---|---|
| 51 | `_FunctionLineVisitor`（AST） | 代码处理 |
| 125 | `class Sandbox(ABC)` | 执行沙箱 |
| 224 | `class LocalSandbox`（常驻多进程 worker 池） | 执行沙箱 |
| ~300 | `_eval_worker` / `_run_evaluation_task` / `_sample_residuals` | 执行沙箱 |
| ~370 | `_sample_to_program` / `_trim_function_body` / `_calls_ancestor` | 程序拼装 |
| 371 | `class EvaluatorAgent` | **Agent 角色** |

结果：想看懂"评估者这个 Agent 做什么"，必须读完全部 516 行的进程池管理细节。
`Sandbox` 还被 `evaluator.py` 兼容层 re-export，且 `config.ClassConfig` 直接引用
`evaluator.Sandbox` 作为类型——说明它本该是独立模块。

### 1.5 入口与配置层位置不当

- **`main.py`（399 行）没有任何函数拆分**：全部逻辑从第 39 行 `if __name__ == '__main__':`
  一直写到第 399 行，包含 CLI 解析、结果目录、`sys.stdout` 重定向、日志配置、
  spec 模板渲染、LLM 客户端构造、快照落盘。无法单测任何一段。
- **变量遮蔽模块名（真实陷阱）**：`main.py:133` / `141` 执行 `config = config.Config(...)`，
  把第 12 行 `from drsr_420 import config` 导入的**模块名覆盖成实例**。此后任何
  `config.ClassConfig` 都会 `AttributeError`。当前靠"用不到"侥幸不炸。
- **重复导入**：`main.py:6`（`import numpy as np`）与 `main.py:50`（`import numpy as _np`）
  各 import 一次 numpy；第 148 行又 `import json as _json`（第 2 行已 import json）。
- **`llm.py`（773 行）在仓库根目录**，却由包内 4 处 `from llm import LLMClient` 依赖
  （`agents/sampler_agent.py:25`、`agents/coordinator_agent.py:35`、`find_best_eq.py:16`、
  `tools/read_paper.py:20`）。
  由于两者父目录相同，**当前一切正常**；风险是：① 无 `pyproject.toml`，包无法独立安装/搬迁；
  ② `llm` 是 PyPI 上的知名包名，一旦安装环境里存在它，`import llm` 可能解析到**另一个库**，
  产生难以定位的导入错误。
- **5 个 `.config` 文件平铺在根目录**（`llm.config`、`glm_glm-5.3-flash.config`、
  `llm_explain.config`、`llm_summary.config`、`rag.config`）。
  注意：`.idea/runConfigurations/*.xml` 以 `--llm_config llm.config` 的相对路径引用它们，
  **本方案不移动这些文件**（理由见 §5.2）。

### 1.6 死代码与历史包袱

| 对象 | 证据 | 处置 |
|---|---|---|
| `drsr_420/parallel_bfgs.py`（42 行） | 全仓 `grep parallel_bfgs` **零引用**（无任何 import、无测试） | 阶段 5 删除 |
| `api.sh` | 文件首行自述"⚠️ 遗留脚本（已失效，请勿使用）"，用已被移除的 `--spec_path` 参数 | 阶段 5 删除或移入 `docs/legacy/` |
| `specs/`（13 个文件） | 仅被失效的 `api.sh` 引用；`README.md:199` 自述"动态模式无需" | 阶段 5 移入 `specs/legacy/`（保留，可能用于复现论文） |
| 根目录 `llm.config.example` 与 6 个 `.config` | 见 §5.2 | 不动 |

### 1.7 诊断汇总

| # | 问题 | 影响 | 等级 |
|---|---|---|---|
| 1 | 7 个 Agent 无统一契约、无注册表、入口命名不一致 | 多 Agent 架构"读不出来" | **高** |
| 2 | Agent 间靠裸元组 + `**kwargs` 袋传数据 | 协作接口不可见、易错 | **高** |
| 3 | `evaluator_agent.py` 516 行 6 类职责 | 最该看懂的角色最难懂 | **高** |
| 4 | 顶层 15 个实现模块平铺、无分层 | 需靠文件名猜层次 | 中 |
| 5 | `main.py` 无函数 + 遮蔽 `config` 模块名 | 不可测、潜伏 `AttributeError` | 中 |
| 6 | 两个编排层分散（`pipeline.py` / `CoordinatorAgent`） | 主循环入口需两处跳转 | 中 |
| 7 | `llm.py` 在根目录、无打包元数据 | 包不可独立安装；`llm` 名冲突风险 | 中 |
| 8 | `llm.py` 773 行混合 4 类职责（传输/工厂/提供商/统计） | 定位提供商适配需全文搜索 | 中 |
| 9 | `drsr_420/` 与 `tools/` 无 `__init__.py` | 无门面、无版本 | 低 |
| 10 | 死代码 `parallel_bfgs.py` / `api.sh` | 误读、误导 | 低 |

---

## 2. 目标架构

### 2.1 目标目录树

```
drsr_420/
├── __init__.py                 # 新增：版本 + 稳定门面（lazy __getattr__）
├── agents/                     # ★ 第 1 层：Agent 角色层（多 Agent 的唯一入口）
│   ├── __init__.py             # 新增：AGENT_REGISTRY + 组织图 + 契约自检
│   ├── base.py                 # 新增：AgentSpec（角色卡）+ BaseAgent 基类
│   ├── messages.py             # 新增：Agent 间消息/产物 dataclass
│   ├── __main__.py             # 新增：python -m drsr_420.agents 打印组织图 / --check
│   ├── coordinator_agent.py    # 协调者
│   ├── sampler_agent.py        # 采样者
│   ├── tool_caller_agent.py    # 工具调用者
│   ├── evaluator_agent.py      # 评估者（瘦身到 ~120 行：只留角色逻辑）
│   ├── experience_summarizer_agent.py
│   ├── residual_analyzer_agent.py
│   ├── data_analyzer_agent.py
│   └── README.md
├── core/                       # ★ 第 2 层：领域无关的共享基础设施与词汇
│   ├── __init__.py
│   ├── buffer.py               # 经验记忆（多岛 + 聚类抽样）
│   ├── code_manipulation.py    # AST / 程序拼装
│   ├── config.py               # 实验配置 dataclass
│   ├── console.py              # 线程前缀输出
│   ├── profile.py              # 样本与进度记录
│   └── prompt_config.py        # 提示词模板 + PromptContext（全体 Agent 共享的词汇表）
├── llm/                        # ★ 第 3 层：LLM 接入（由根 llm.py 拆分）
│   ├── __init__.py             # 对外只有 LLMClient / ClientFactory / 统计函数
│   ├── client.py               # LLMClient：请求/重试/流式/适配/限额
│   ├── factory.py              # ClientFactory + parse_provider_model/normalize_llm_config
│   ├── providers.py            # DeepSeek/Siliconflow/DeepInfra/CSTCloud/Ollama/Blt/Zhipu
│   └── stats.py                # 全局 token/耗时统计（含锁）
├── runtime/                    # ★ 第 4 层：编排与执行
│   ├── __init__.py
│   ├── pipeline.py             # 实验编排（= 现 pipeline.py）
│   └── evaluation/             # 评估执行子系统（从 evaluator_agent.py 拆出）
│       ├── __init__.py
│       ├── problems.py         # ← evaluate_on_problems.py（拟合与打分）
│       ├── sandbox.py          # ← Sandbox/LocalSandbox/_eval_worker/_run_evaluation_task
│       └── accelerate.py       # ← evaluator_accelerate.py（numba 可选加速）
├── knowledge/                  # ★ 第 5 层：外部知识（文献 / RAG / MCP 工具）
│   ├── __init__.py
│   ├── rag_kb.py  rag_build.py  tool_runner.py
│   └── tools/{__init__.py, mcp_server.py, search_paper.py, read_paper.py, tools_description.py}
├── analysis/                   # ★ 第 6 层：收尾分析
│   ├── __init__.py
│   ├── find_best_eq.py  sensitivity_prune.py
└── cli/                        # ★ 第 7 层：命令行
    ├── __init__.py
    └── main.py                 # ← 根 main.py，拆成 main() + 若干纯函数
```

根目录**保留**（兼容，见 §5）：

```
main.py                     # 3 行 shim → drsr_420.cli.main
llm.py                      # shim → drsr_420.llm
drsr_420/{buffer,config,console,profile,code_manipulation,prompt_config,
          evaluate_on_problems,evaluator_accelerate,evaluator,sampler,
          tool_caller,experience_summarizer,residual_analyzer,data_analyse_real,
          rag_kb,rag_build,tool_runner,find_best_eq,sensitivity_prune}.py   # 全部 shim
```

### 2.2 依赖方向规则（硬约束，由测试守护）

```
cli  ──▶ runtime ──▶ agents ──▶ core
              │         │        ▲
              │         └────────┤
              └──▶ knowledge ────┘
analysis ──▶ core / llm / knowledge
llm ──▶ (core 之外无依赖；唯一例外：tools_description 的 schema 常量)
```

- **禁止反向依赖**：`core/` 不得 import `agents/`、`runtime/`、`cli/`。
- **禁止同级循环**：`agents/` 内部只允许 `coordinator → 其它 6 个` 的单向依赖；
  其它 6 个 Agent 之间只能通过 `messages.py` 的类型通信，不得互相 import。
- **`llm/` 不得 import `agents/`**。
- 规则写成 `tests/test_architecture.py` 的断言（基于 AST 扫描 import），CI 可查。

### 2.3 每个 Agent 的「角色卡」

统一用 `AgentSpec` 声明，`agents/__init__.py` 汇总成注册表。这是一张**可执行**的角色表——
不再只是 Markdown 里的文字，而是能被程序枚举、校验、打印的对象。

| key | 类 | 角色 | 入口 | 上游 | 下游 | 产物 |
|---|---|---|---|---|---|---|
| `coordinator` | `CoordinatorAgent` | 协调者 | `run` / `sample` | pipeline | sampler, evaluator, experience_summarizer, residual_analyzer | `checkpoint.json`, `round_progress.csv` |
| `sampler` | `SamplerAgent` | 采样者 | `draw_samples` | coordinator | tool_caller | — |
| `tool_caller` | `ToolCallerAgent` | 工具调用者 | `complete` | sampler | MCP 工具（knowledge） | — |
| `evaluator` | `EvaluatorAgent` | 评估者 | `analyze`（保留 `analyse` 别名） | coordinator, pipeline | sandbox（runtime） | `samples/samples_N.json`（经 Profiler） |
| `experience_summarizer` | `ExperienceSummarizerAgent` | 经验总结者 | `analyze` | coordinator | — | `experiences.json`（由 coordinator 落盘） |
| `residual_analyzer` | `ResidualAnalyzerAgent` | 残差分析者 | `analyze` | coordinator | — | `residual_analyze.json`（由 coordinator 落盘） |
| `data_analyzer` | `DataAnalyzerAgent` | 数据分析者 | `analyze` | pipeline | — | `residual_analyze.json`（`sample_order=0`） |

### 2.4 一次完整协作时序（目标态，读代码即可对照）

```
cli.main
  └─ runtime.pipeline.main()
       ├─ core.buffer.ExperienceBuffer            ← 共享记忆（含断点恢复）
       ├─ DataAnalyzerAgent.analyze()             → residual_analyze.json#sample_order=0   [单次]
       └─ for i in Sampler-i 线程:
            CoordinatorAgent.run()
              └─ while 未达上限:
                   ├─ buffer.get_prompt()                        # 取岛 + 软采样
                   ├─ SamplerAgent.draw_samples()                 # 生成骨架
                   │    └─ ToolCallerAgent.complete()             # 多轮 MCP 工具调用
                   ├─ EvaluatorAgent.analyze() → EvaluationOutcome # 沙箱执行 + 多起点拟合
                   ├─ 分类 Good / Bad / None
                   ├─ ExperienceSummarizerAgent.analyze() → [ExperienceEntry]  → experiences.json
                   ├─ ResidualAnalyzerAgent.analyze()      → ResidualInsight  → residual_analyze.json
                   └─ checkpoint.json / round_progress.csv
       └─ analysis.find_best_eq()                 # 收尾：参数拟合 + 物理解释 [非 Agent]
```

---

## 3. 四大支柱与具体改法

### 支柱 A：统一 Agent 契约（阶段 1）

新增 `agents/base.py`：

```python
@dataclasses.dataclass(frozen=True)
class AgentSpec:
    """一个 Agent 的「角色卡」：声明式描述它是谁、从谁拿输入、给谁输出。"""
    key: str                       # 稳定标识，如 "evaluator"
    role: str                      # 中文角色名，如 "评估者"
    mission: str                   # 一句话使命
    entrypoints: tuple[str, ...]   # 对外入口方法名（至少 1 个）
    upstream: tuple[str, ...]      # 上游（"pipeline" 或其它 agent 的 key）
    downstream: tuple[str, ...]    # 下游
    consumes: tuple[str, ...]      # 输入契约（messages.py 的类名）
    produces: tuple[str, ...]      # 输出契约
    artifacts: tuple[str, ...]     # 落盘产物
    thread_model: str              # "per-sampler-thread" / "single-shot"
    llm_task: str | None           # 使用的 LLM 客户端副本：sampling/experience/residual/None


class BaseAgent(abc.ABC):
    """所有 Agent 的基类：只管「身份 + 契约校验 + 自描述」，不约束业务方法签名。"""

    SPEC: ClassVar[AgentSpec]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        spec = getattr(cls, "SPEC", None)
        if spec is None:
            raise TypeError(f"{cls.__name__} 必须声明 SPEC（AgentSpec）")
        if not spec.entrypoints:
            raise TypeError(f"{cls.__name__}.SPEC.entrypoints 不能为空")
        missing = [m for m in spec.entrypoints if not callable(getattr(cls, m, None))]
        if missing:
            raise TypeError(f"{cls.__name__} 声明的入口方法不存在: {missing}")

    @property
    def agent_spec(self) -> AgentSpec:
        return type(self).SPEC

    def describe(self) -> str:      # 单行角色卡，供组织图与日志使用
        s = self.SPEC
        return f"{type(self).__name__}（{s.role}）: {s.mission}"
```

要点：

- **不强制统一 `run()` 签名**。7 个 Agent 的输入输出本质不同，强行统一只会造出 `**kwargs` 袋。
  统一的是**元数据 + 校验**；`run()` 在阶段 2 作为带类型请求的薄封装出现。
- `__init_subclass__` 在**类定义时**就校验契约，也就是说"漏声明 SPEC"或"入口改名忘同步"
  会在 import 阶段直接报错，而不是等运行到某分支才炸。
- **命名统一**：`EvaluatorAgent.analyse` → `analyze`，保留 `analyse = analyze` 别名一个版本；
  同步 `coordinator_agent.py`、`pipeline.py`、`tests/test_evaluator.py` 的调用点。

`agents/__init__.py`（PEP 562 惰性导出，保留现有"不 eager import"的安全属性）：

```python
"""DRSR 多 Agent 系统 —— 角色层。运行 `python -m drsr_420.agents` 查看组织图。"""
from __future__ import annotations

_AGENT_EXPORTS = {
    "BaseAgent":        "drsr_420.agents.base",
    "AgentSpec":        "drsr_420.agents.base",
    "CoordinatorAgent": "drsr_420.agents.coordinator_agent",
    "SamplerAgent":     "drsr_420.agents.sampler_agent",
    ...
}

def __getattr__(name):                     # 惰性导入，避免循环导入与启动开销
    module_path = _AGENT_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(module_path), name)

def agent_specs() -> dict[str, AgentSpec]:  # 枚举全部角色（供组织图 / --check / 测试）
    ...

def describe_architecture() -> str: ...     # ASCII 组织图
def check_contracts() -> list[str]: ...     # 返回问题列表，空 = 通过
```

`agents/__main__.py`：

```
$ python -m drsr_420.agents
DRSR 多 Agent 系统（7 个角色）
  [coordinator] CoordinatorAgent  协调者   每 Sampler-i 线程一个实例，驱动主循环
    ├─[sampler] SamplerAgent       采样者   ...
    ...

$ python -m drsr_420.agents --check
OK: 7 个 Agent 契约齐全，依赖方向合法
```

### 支柱 B：显式消息契约（阶段 2）

新增 `agents/messages.py`，把现在靠元组/kwargs 传递的数据变成可读类型：

```python
@dataclasses.dataclass
class EvaluationRequest:            # coordinator → evaluator
    sample: str
    island_id: int
    version_generated: int
    global_sample_nums: int
    sample_time: float
    profiler: 'Profiler | None' = None

@dataclasses.dataclass
class EvaluationOutcome:            # evaluator → coordinator（替代 (score, error, residual) 裸元组）
    score: float | None
    error: str | None
    residual: 'np.ndarray | None'

@dataclasses.dataclass
class ExperienceEntry:              # summarizer → coordinator → experiences.json
    sample: str
    quality: str                    # 'Good' | 'Bad' | 'None'
    score: float | None
    error: str | None
    analysis: str

@dataclasses.dataclass
class ResidualInsight:              # residual_analyzer → coordinator → residual_analyze.json
    sample: str
    best_score: float | None
    sample_order: int
    analysis: str

@dataclasses.dataclass
class ToolCall:                     # tool_caller ↔ knowledge.tool_runner
    name: str
    arguments: dict
    result: str | None = None
    error: str | None = None

@dataclasses.dataclass
class SampleBatch:                  # 从 coordinator_agent.py 迁入（已存在，line 84）
    ...
```

收益：

- `EvaluatorAgent.analyze(EvaluationRequest) -> EvaluationOutcome` 一眼可读，
  **且消除 `**kwargs` 穿透沙箱**这一跨任务参数泄漏路径（现状：`analyse(**kwargs)` 把
  `global_sample_nums`/`sample_time`/`profiler` 一路透传）。
- `experiences.json` / `residual_analyze.json` 的字段来源可追溯到具体 dataclass。
- 兼容：`EvaluatorAgent.analyze` 允许传旧式三元组返回（内部 `to_legacy()`），
  但**内部调用点全部切到 dataclass**，旧式仅作过渡。

### 支柱 C：分层目录 + 兼容 shim（阶段 3、4）

**迁移铁律**：每个被移动的模块，在**旧路径**留一个 3–8 行的 re-export shim。
仓库已有成熟先例（6 个 shim 已稳定工作、`tests/test_evaluator.py` 正依赖 `drsr_420.evaluator`），
所以这条路径是**已验证可行**的，不是新实验。

迁移映射表：

| 旧路径 | 新路径 | shim |
|---|---|---|
| `llm.py` | `drsr_420/llm/`（client/factory/providers/stats） | `llm.py` |
| `main.py` | `drsr_420/cli/main.py` | `main.py` |
| `drsr_420/buffer.py` | `drsr_420/core/buffer.py` | ✅ |
| `drsr_420/code_manipulation.py` | `drsr_420/core/code_manipulation.py` | ✅ |
| `drsr_420/config.py` | `drsr_420/core/config.py` | ✅ |
| `drsr_420/console.py` | `drsr_420/core/console.py` | ✅ |
| `drsr_420/profile.py` | `drsr_420/core/profile.py` | ✅ |
| `drsr_420/prompt_config.py` | `drsr_420/core/prompt_config.py` | ✅ |
| `drsr_420/pipeline.py` | `drsr_420/runtime/pipeline.py` | ✅ |
| `drsr_420/evaluate_on_problems.py` | `drsr_420/runtime/evaluation/problems.py` | ✅ |
| `drsr_420/evaluator_accelerate.py` | `drsr_420/runtime/evaluation/accelerate.py` | ✅ |
| `agents/evaluator_agent.py` 内的沙箱部分 | `drsr_420/runtime/evaluation/sandbox.py` | 由 `agents/evaluator_agent.py` re-export |
| `drsr_420/rag_kb.py` | `drsr_420/knowledge/rag_kb.py` | ✅ |
| `drsr_420/rag_build.py` | `drsr_420/knowledge/rag_build.py` | ✅ |
| `drsr_420/tool_runner.py` | `drsr_420/knowledge/tool_runner.py` | ✅ |
| `drsr_420/tools/*` | `drsr_420/knowledge/tools/*` | ✅（含 `__init__.py`） |
| `drsr_420/find_best_eq.py` | `drsr_420/analysis/find_best_eq.py` | ✅ |
| `drsr_420/sensitivity_prune.py` | `drsr_420/analysis/sensitivity_prune.py` | ✅ |

**关键决策：不做"全局改 import 路径"**。包内代码统一改用**新规范路径**；
旧路径 shim 只服务外部脚本与历史测试。这样：
- "读到的就是架构"（包内无旧路径残留）；
- 老代码/脚本/IDE 配置零改动。

### 支柱 D：文档与可视化（阶段 5）

- `docs/ARCHITECTURE.md`：**唯一权威架构文档**——分层图、依赖方向规则、
  7 张角色卡（由 `agent_specs()` 生成，避免手写失同步）、一次协作时序、
  产物清单、扩展指南（"如何新增一个 Agent"）。
- `README.md` 的"多 Agent 系统架构"段收敛为一张图 + 一个指向 `docs/ARCHITECTURE.md` 的链接，
  避免 README 与 `agents/README.md` 两处重复描述互相漂移（现状：两处都在画协作图）。
- `drsr_420/agents/README.md` 保留为**角色手册**（接口与用法），
  删掉与 README 重复的架构总览段。
- **架构自检进测试**：`tests/test_architecture.py` 守护 §2.2 的依赖规则与契约完整性。

---

## 4. 分阶段执行计划

| 阶段 | 内容 | 主要文件 | 预计改动 | 风险 | 回滚 |
|---|---|---|---|---|---|
| **0** | 基线护栏 | `tests/test_architecture.py`（新增，宽松版） | +150 行 | 极低 | 删文件 |
| **1** | 支柱 A：Agent 契约 + 注册表 + 命名统一 | `agents/base.py`(新)、7 个 agent、`agents/__init__.py`、`agents/__main__.py`(新)、`pipeline.py` | +300 / ~60 改 | **低** | 单提交 revert |
| **2** | 支柱 B：消息契约 | `agents/messages.py`(新)、`coordinator_agent.py`、`evaluator_agent.py`、`sampler_agent.py`、`experience_summarizer_agent.py`、`residual_analyzer_agent.py` | +150 / ~120 改 | **中** | 单提交 revert |
| **3** | 支柱 C-1：分层迁移（core/llm/runtime/knowledge/analysis/cli） | ~40 个文件移动 + 19 个新 shim | +400 shim / ~80 改 import | **中高** | 单提交 revert |
| **4** | 支柱 C-2：评估子系统拆分 | `evaluator_agent.py` 瘦身；新增 `runtime/evaluation/{sandbox,problems,accelerate}.py` | ~400 行搬迁 | 中 | 单提交 revert |
| **5** | 支柱 D：文档 + 死代码清理 + 打包元数据 | `docs/ARCHITECTURE.md`(新)、`README.md`、`agents/README.md`、`pyproject.toml`(可选)、删 `parallel_bfgs.py`/`api.sh` | +500 文档 / −100 代码 | 低 | 单提交 revert |

### 阶段 0：基线与护栏

1. 记录基线：`Ran 217 tests ... OK`（命令见 §6）。
2. 新增 `tests/test_architecture.py`，先只断言"现状成立且必须继续成立"的部分：
   - `drsr_420.agents` 可导入，且 `agents/__init__.py` **不 eager import** 任何 Agent
     （防循环导入的现有保证）；
   - 旧兼容路径可用：`drsr_420.evaluator.Evaluator is agents.evaluator_agent.EvaluatorAgent` 等 6 组；
   - `latest` 兼容层与规范模块指向同一对象（`is` 比较，防复制粘贴式分叉）。
3. **验收**：217 + 新增测试全绿。

### 阶段 1：统一 Agent 契约（最高性价比）

1. 新增 `agents/base.py`（`AgentSpec` + `BaseAgent`，代码见 §3-A）。
2. 7 个 Agent 类改为继承 `BaseAgent` 并声明 `SPEC`（角色卡的 key/role/mission/入口/上下游/
   契约/产物/线程模型/LLM 用途）。
3. `EvaluatorAgent.analyse` → `analyze`（保留 `analyse` 别名并标注 deprecated）；
   同步 `coordinator_agent._evaluate_batch`、`pipeline._run_initial_analysis`、
   `tests/test_evaluator.py`、`tests/test_core_contracts.py`。
4. `agents/__init__.py`：`_AGENT_EXPORTS` + `__getattr__` 惰性导出 +
   `agent_specs()` / `describe_architecture()` / `check_contracts()`。
5. `agents/__main__.py`：打印组织图；`--check` 做契约自检（非零退出码表示失败）。
6. `tests/test_architecture.py` 增加：7 个类都继承 `BaseAgent`、都声明 `SPEC`、
   `entrypoints` 全部可调用、`key` 唯一、上下游的 key 都在注册表内（防拼写错）。
7. **验收**：
   - `python -m drsr_420.agents` 输出 7 角色组织图；
   - `python -m drsr_420.agents --check` 退出码 0；
   - 全测试通过；
   - **人工检查点**：把 `agents/__init__.py` 的组织图与 `agents/README.md` 的图对照，若不一致则说明
     有一处已漂移（这正是本次重构要消除的问题）。

> 阶段 1 完成后，**"项目是多 Agent 系统"已经可以被机器验证**，而不再依赖一份手写文档。

### 阶段 2：显式消息契约

1. 新增 `agents/messages.py`（§3-B 的 6 个 dataclass），`SampleBatch` 从
   `coordinator_agent.py:84` 迁入并在原处 re-export（保持 `from ...coordinator_agent import SampleBatch` 可用）。
2. `EvaluatorAgent`：`analyze(self, request: EvaluationRequest) -> EvaluationOutcome`；
   删除 `**kwargs` 袋；`analyse` 作为过渡别名接受旧参数并内部组装 request。
   沙箱调用改传显式字段（顺带消除跨任务参数泄漏路径）。
3. `ExperienceSummarizerAgent.analyze` / `ResidualAnalyzerAgent.analyze` 改为
   接收/返回 dataclass（`list[ExperienceEntry]` / `ResidualInsight`）。
4. `CoordinatorAgent` 内部调用点全部切到 dataclass；`_persist_experiences` /
   `_persist_residual` 从 dataclass 取字段（消除"字段名靠约定对齐"）。
5. 更新受影响的测试；新增断言：`EvaluationOutcome` 字段与 `samples_N.json` 落盘字段一致。
6. **验收**：全测试通过；`grep -c '\*\*kwargs' drsr_420/agents/*.py` 合计为 0
   （`CoordinatorAgent.sample(**kwargs)` 中转发 `profiler` 的 kwargs 改为显式参数）。

### 阶段 3：分层迁移

按 **模块 → 新位置 + 旧 shim** 的顺序逐个搬，**每个子步跑一次全测试**（不攒到最后）：

1. `llm.py` → `drsr_420/llm/`（client/factory/providers/stats 四个文件按 §1.7#8 的职责切分），
   根 `llm.py` 变 shim。**先做这个**，因为它牵动面最广（4 处包内 import + 3 个测试）。
2. `core/`：buffer、code_manipulation、config、console、profile、prompt_config（6 个 shim）。
   注意 `config.py` 的 `TYPE_CHECKING` 分支要指向新路径的 `LLM`/`Sandbox`。
3. `knowledge/`：rag_kb、rag_build、tool_runner、tools/*（含补 `tools/__init__.py`）。
   `rag_build.py` 的 `python -m drsr_420.rag_build` 用法保持不变（shim 支持 `-m` 运行）。
4. `analysis/`：find_best_eq、sensitivity_prune。
5. `runtime/`：pipeline（shim）。
6. `cli/`：`main.py` 拆成 `main(argv=None)` + `build_spec()` / `_load_csv()` /
   `_setup_output_tee()` / `_build_prompt_context()` / `_snapshot_config()` 等纯函数，
   修掉 `config` 变量遮蔽（`config` → `exp_config`）与重复 import；
   根 `main.py` 变 shim（`from drsr_420.cli.main import main; raise SystemExit(main())`）。
7. 更新 `tests/test_architecture.py`：加入 §2.2 依赖方向 AST 扫描断言。
8. **验收**：全测试通过；`python main.py --help` 输出与基线一致（对比保存的 help 文本）；
   `python -m drsr_420.rag_build --query x`（无需真的命中）不报导入错误；
   **人工检查点**：IDE 运行配置 `$PROJECT_DIR$/main.py` 仍可启动（shim 生效）。

### 阶段 4：评估子系统拆分

1. 新建 `runtime/evaluation/sandbox.py`：搬 `Sandbox`、`LocalSandbox`、`_eval_worker`、
   `_run_evaluation_task`、`_sample_residuals`、`_FunctionLineVisitor`（≈320 行）。
2. 新建 `runtime/evaluation/problems.py`（原 `evaluate_on_problems.py`）、
   `accelerate.py`（原 `evaluator_accelerate.py`）。
3. `agents/evaluator_agent.py` 只留 `EvaluatorAgent`（≈120 行）+ 从 sandbox 模块 re-export
   原符号（`LocalSandbox`、`_sample_to_program`、`_run_evaluation_task`、`_sample_residuals`
   被 3 个测试文件引用，必须保持可导入）。
4. `core/config.py` 的 `ClassConfig.sandbox_class` 类型注解指向 `runtime.evaluation.sandbox.Sandbox`
   （`main.py` 里 `evaluator.LocalSandbox` 的用法由 shim 兜住）。
5. **验收**：全测试通过；`evaluator_agent.py` < 200 行；
   `grep -c 'multiprocessing' drsr_420/agents/evaluator_agent.py` == 0。

### 阶段 5：文档、清理、打包

1. 新增 `docs/ARCHITECTURE.md`（唯一权威）；角色卡段落用 `agent_specs()` 生成后粘贴，保证与代码一致。
2. `README.md` 架构段瘦身 + 指向 `docs/ARCHITECTURE.md`；`agents/README.md` 去重。
3. 删除 `drsr_420/parallel_bfgs.py`（零引用）、`api.sh`（自述失效）；
   `specs/` 移入 `specs/legacy/` 并加一行 README 说明（保留复现价值，不删）。
4. 新增 `pyproject.toml`（setuptools，`drsr_420*` 打包，无新增运行依赖），
   使 `pip install -e .` 可用——**这一步才能真正消除 `llm` 顶层模块名的冲突风险**。
5. `.gitignore` 增补 `docs/` 之外无需忽略项；确认 `*.config` 仍被忽略。
6. **验收**：`pip install -e .` 后从任意目录 `python -c "import drsr_420"` 成功；
   `python -m drsr_420.agents --check` 通过。

---

## 5. 兼容策略

### 5.1 保留的旧导入路径

**19 个新增 shim**（§3-C 映射表）+ **6 个既有 Agent shim 保持不变**（`sampler.py` / `evaluator.py` /
`tool_caller.py` / `experience_summarizer.py` / `residual_analyzer.py` / `data_analyse_real.py`），
全部是 3–8 行 re-export，形如：

```python
"""兼容层：旧路径 drsr_420.buffer → 新路径 drsr_420.core.buffer。仅 re-export。"""
from drsr_420.core.buffer import *          # noqa: F401,F403
from drsr_420.core.buffer import ExperienceBuffer, Prompt, Island  # 显式补全 __all__ 之外的名字
```

- shim 保留**至少一个版本**，在 `docs/ARCHITECTURE.md` 标注 `@deprecated since vX`。
- `tests/test_architecture.py` 断言每组 `旧.名字 is 新.名字`，防止 shim 分叉成副本。
- 删除条件：外部脚本（含 `.idea/runConfigurations`、`example.sh`、`MRFCompress-3.sh`）
  与本仓测试全部改到新路径后，按 §5.2 的"不动清单"评估。

### 5.2 明确**不**改动（回归安全线）

| 对象 | 原因 |
|---|---|
| 根 `main.py` 的存在与命令行接口 | `.idea/runConfigurations/*.xml` 用 `$PROJECT_DIR$/main.py` + 6 个 CLI 参数 |
| 根目录 `.config` 文件名与位置 | 运行配置用相对路径 `--llm_config llm.config`，且 `rag.config` 由 `load_config()` 按固定名查找 |
| `python -m drsr_420.rag_build` 调用形式 | `README.md` 已公开该用法 |
| `experiments/{problem}_{ts}/` 产物文件名与 JSON 字段 | 历史实验数据与既有分析脚本依赖（`experiments/` 下已有大量真实结果） |
| CLI 参数名与默认值（`--niterations` 等 11 个） | 用户既有命令与脚本 |
| 6 个旧 Agent shim 模块 | 已稳定工作，且被测试依赖 |
| `data/`、`experiments/`、`knowledge_base/`、`pdf_downloads/` | 数据资产 |
| 算法与数值行为 | 本方案是纯结构重构；残差精度、评分语义等已固化的契约不得回归 |

### 5.3 shim 运行 `-m` 的注意点

`python -m drsr_420.rag_build` 依赖 `rag_build.py` 的 `if __name__ == "__main__":` 块。
把实现搬到 `knowledge/rag_build.py` 后，旧 shim 必须**转发执行**而不是仅 re-export：

```python
if __name__ == "__main__":
    from drsr_420.knowledge.rag_build import main
    raise SystemExit(main())
```

（现有 `rag_build.py` 是否已是函数+入口形式，阶段 3 实施时先确认再定写法。）

---

## 6. 验收标准（Definition of Done）

**全局（每个阶段都必须满足）**

```powershell
$env:ZHIPU_API_KEY='dummy-test-key'
& 'C:\ResearchCode\drsr-main\.venv2\Scripts\python.exe' tests/run_tests.py
# 期望：Ran 217+ tests ... OK   （注意：PowerShell 管道会让退出码失真，以 OK/FAILED 行为准）
```

- 现有 217 项测试**零失败、零跳过**（除既有 skip）；
- 新增测试只增不减；
- `python -m drsr_420.agents --check` 退出码 0（阶段 1 起）；
- `git status` 干净，每阶段一个独立提交、可单独 revert。

**结构类（阶段 3–4 完成后）**

| 指标 | 现状 | 目标 |
|---|---|---|
| `drsr_420/` 顶层**实现**模块数（排除 shim 与 `__init__.py`） | 15（含 1 个死模块） | 0（实现全部进子包） |
| `drsr_420/` 顶层转发文件数 | 6（既有 Agent shim） | 25（6 既有 + 19 新增；均标 `@deprecated`，可择期删除） |
| 子包数 | 2（agents/tools） | 7（agents/core/llm/runtime/knowledge/analysis/cli） |
| `agents/evaluator_agent.py` 行数 | 516 | < 200 |
| 最长文件行数 | 773（`llm.py`） | < 500（拆分后 `llm/client.py` ≈ 460） |
| `**kwargs` 在 `agents/` 中的出现次数 | 11 | 0 |
| Agent 入口命名不一致数 | 1（`analyse`） | 0 |
| 无引用的死模块 | 1（`parallel_bfgs.py`） | 0 |

**可读性类（最终目标，人工检查点）**

一位没读过本项目的工程师，在**不读任何测试、不运行程序**的前提下，应能在 10 分钟内回答：

1. 系统有几个 Agent？各自角色与职责？ → `python -m drsr_420.agents` / `docs/ARCHITECTURE.md`
2. 谁调用谁？一轮完整循环的顺序？ → `docs/ARCHITECTURE.md` §协作时序 + `AgentSpec.upstream/downstream`
3. Agent 之间传什么数据？ → `agents/messages.py`（每个 dataclass 一个注释）
4. 共享状态是什么？并发怎么保护？ → `CoordinatorAgent` 角色卡 + `core/buffer` 文档
5. 产物落在哪、谁写的？ → `AgentSpec.artifacts` + `README.md` 产物表
6. 新增一个 Agent 要改哪几处？ → `docs/ARCHITECTURE.md` §扩展指南

---

## 7. 不做的事（Non-goals）

- ❌ 不改算法、不改提示词内容、不改数值行为（残差精度、评分语义、采样策略等已固化契约保持原样）。
- ❌ 不引入新运行依赖（不引入 pydantic/attrs/typing_extensions 等；只用 `dataclasses` + `abc`）。
- ❌ 不把 7 个 Agent 强行塞进统一 `run(**kwargs)` 签名（会制造新的隐式契约）。
- ❌ 不引入 async/事件总线/消息队列等"更 Agent"的运行时——本项目是进程内线程协作，
  结构清晰即可，不需要重写调度模型。
- ❌ 不用 `git filter-repo` 改写历史（与本文档无关的独立决策）。
- ❌ 不移动 `.config` 文件、不改 CLI 参数、不改 `experiments/` 产物格式。
- ❌ 不为每个 Agent 造 `tests/test_<agent>.py`（现有测试已覆盖行为；本方案只加结构测试）。

---

## 8. 风险与回滚

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 迁移时漏改某处 import → 运行期 `ImportError` | 中 | 运行失败 | shim 兜底（旧路径始终可用）；每子步跑全测试 |
| 循环导入（新 `agents/__init__.py` / `config.py` 类型引用） | 中 | 导入即失败 | `agents/__init__.py` 用 PEP 562 惰性导出；`config.py` 类型引用留在 `TYPE_CHECKING` |
| 测试通过但运行时行为变化 | 低 | 结果不可比 | 不改算法；`python main.py --help` 与基线文本比对；`--check` 自检 |
| 阶段 3 大改动与你的 IDE 提交冲突 | 中 | 合并困难 | 每阶段独立小提交、与用户协调提交时机；不跨阶段攒改动 |
| `python -m drsr_420.rag_build` 被 shim 破坏 | 低 | 文档命令失效 | shim 转发 `main()`（§5.3）；阶段 3 显式验证该命令 |
| 重构引入"改名但不改语义"的隐性行为差异 | 低 | 难查 | 阶段 2 的 dataclass 与旧元组做等价性单测（`to_legacy()` 往返断言） |

回滚：每个阶段一个提交，`git revert <sha>` 即可；shim 的存在保证回滚后外部脚本不受影响。

---

## 9. 附录：影响面清单（实施时对照）

**`import llm` 的位置（阶段 3 必须同步）**

| 文件 | 行 |
|---|---|
| `drsr_420/agents/sampler_agent.py` | 25 |
| `drsr_420/agents/coordinator_agent.py` | 35 |
| `drsr_420/find_best_eq.py` | 16 |
| `drsr_420/tools/read_paper.py` | 20 |
| `drsr_420/profile.py` | 21（`import llm as llm_stats`，包在 try 内） |
| `main.py` | 15（`import llm as llm_mod`） |
| `tests/test_core_contracts.py` / `test_llm_provider_adapt.py` / `test_llm_stream.py` | 21 / 15 / 14 |

**被测试直接依赖的符号（拆文件时必须保持可导入）**

| 符号 | 现位置 | 引用方 |
|---|---|---|
| `LocalSandbox` | `agents/evaluator_agent.py:224` | `tests/test_evaluator.py`、`test_core_contracts.py:207`、`test_monitoring_hardening.py:17` |
| `_run_evaluation_task`、`_sample_residuals` | `agents/evaluator_agent.py` | `tests/test_evaluator.py:10` |
| `_sample_to_program` | `agents/evaluator_agent.py` | `tests/test_core_contracts.py:95` |
| `drsr_420.evaluator.Evaluator` | shim | `tests/test_evaluator.py:9` |
| `evaluate_on_problems` | `drsr_420/evaluate_on_problems.py` | `tests/test_evaluate_on_problems.py:7`、`test_core_contracts.py:24`、`main.py:17` |
| `chunk_text` / `DEFAULT_CONFIG` | `drsr_420/rag_kb.py` | `tests/test_mcp_and_rag_cli.py:19`、`test_monitoring_hardening.py:16` |
| `find_best_eq` 的若干函数 | `drsr_420/find_best_eq.py` | `tests/test_expr_substitution.py:6` |
| `sensitivity_prune` 的若干函数 | `drsr_420/sensitivity_prune.py` | `tests/test_sensitivity_prune.py:7` |
| `Profiler` | `drsr_420/profile.py` | `tests/test_monitoring_hardening.py:18` |

**基线指标（阶段 0 记录，阶段 5 对比）**

| 指标 | 基线值 |
|---|---|
| 测试数 | 217 |
| `drsr_420/` + `tests/` 源码行数 | 9,375 |
| `drsr_420/` 顶层 `.py` 数 | 21（15 实现 + 6 shim） |
| 子包数 | 2 |
| 最长文件 | `llm.py` 773 行 |
| Agent 数 | 7 |
| 兼容 shim 数 | 6 |

---

## 10. 建议执行顺序与决策点

**推荐**：阶段 0 → 1 → 2 →（评审）→ 3 → 4 → 5。
阶段 0–2 是**纯增量**（只加文件、改少量调用点，不动文件位置），收益已覆盖"多 Agent 一眼可见"的核心诉求；
阶段 3–4 是真正的分层迁移，风险集中在 import 路径，由 shim 与逐子步测试兜住。

**决策点（已确认）**：

1. **范围**：全量（0–5）——已按此执行。
2. **提交节奏**：每阶段一个提交（便于单独 revert）——已按此执行。
3. **打包**：引入 `pyproject.toml`（依赖清单复用 `requirements.txt`，版本取自
   `drsr_420.__version__`）——已完成。
4. **兼容期**：shim 长期保留（成本低，且是新旧路径对象同一性的活文档）；删除时机
   见 [`ARCHITECTURE.md`](./ARCHITECTURE.md) §8。

---

## 11. 执行记录（阶段 0–5）

### 11.1 与计划的偏差（均为实测证据驱动，故优于原计划）

| # | 计划 | 实际 | 原因 |
|---|---|---|---|
| 1 | 7 层（`runtime/evaluation/` 归 runtime） | **8 层**：评估执行子系统提升为独立的 `evaluation` 层 | 实测 `agents → runtime` 是**真实倒置**：`agents/evaluator_agent.py` 要用拟合与沙箱，而它们本在 runtime 之下。照搬计划会固化错误分层 |
| 2 | `llm/stats.py` | **`core/llm_stats.py`** | 实测 `core/profile.py` 要读 LLM 用量统计 → `core → llm` 倒置。统计是"共享基础设施"，下沉到底层即消除倒置 |
| 3 | `specs/` 移入 `specs/legacy/` | **未移动** | 它只被已失效的 `api.sh` 引用；在自身目录内改名不产生价值，改为在 README 标注"历史静态 spec，动态模式已不使用" |
| 4 | `agents/evaluator_agent.py` < 200 行 | **223 行**（原 516 行，−57%） | 剩余部分是一个角色的完整实现 + 兼容 re-export 清单；为凑阈值压缩代码会损害可读性，与目标相反 |
| 5 | 最长文件 < 500 行 | `llm.py` 773 → `llm/client.py` 482 ✓；全局最长仍是 `analysis/find_best_eq.py` 557 | 后者是**未列入计划的既有大文件**，本次不动（属可选后续） |

### 11.2 实测指标对照

| 指标 | 计划目标 | 实测 |
|---|---|---|
| 测试数 | 217 → 不减少 | **262**（新增 45 项：契约/消息/分层/依赖/路径/模板护栏） |
| 子包数 | 2 → 7 | **9**（8 层 + 兼容层用的 `tools/`） |
| `drsr_420/` 顶层**实现**模块 | 0 | **0**（21 个顶层 `.py` = `__init__.py` + 20 个兼容层） |
| `agents/evaluator_agent.py` | < 200 行 | 223 行（`multiprocessing` 出现次数 0） |
| 最长文件 | < 500 | 482（`llm/client.py`）；包内最长 557（既有文件） |
| `**kwargs` 在 agents 契约方法中 | 0 | **0**（仅剩 `__init_subclass__` 与 `clone_llm_client` 的参数转发） |
| Agent 入口命名不一致 | 0 | **0**（`analyze` 统一，`analyse` 为显式 deprecated 别名） |
| 无引用的死模块 | 0 | **0**（`parallel_bfgs.py`、`api.sh` 已删除） |
| 依赖倒置 | 0 | **0**（AST 扫描 + 可达性自检守护） |

### 11.3 重构过程中发现并修复的真实缺陷

这些都是"改结构时顺手被测试/实测抓出来"的既有缺陷，均已修复并加回归护栏：

1. **`mock.patch` 打桩穿透兼容层失效**（阶段 3）：只做读转发的 shim 会让
   `mock.patch('llm._post_with_retry')` 落在门面上，而 `LLMClient.chat` 在 `client`
   模块的全局命名空间里查找——桩等于没打；`mock` 退出时的 `delattr → hasattr → setattr`
   还原路径还会把属性丢在门面上，导致后续调用 `NameError`。已实现"写入转发到定义模块
   + owner 缓存"。
2. **`agents/messages.py` 之前的隐式契约**（阶段 2）：`ExperienceSummarizerAgent` 靠
   `zip` 对齐 5 个平行列表，任一处错位都会把经验静默配到别的样本上；现已改为显式
   dataclass + 长度校验。
3. **`prompt_config.residual_analysis_prompt` 的未转义花括号**：`.format()` 恒抛
   `KeyError`——"无 PromptContext 的兜底模板"这条分支从未可用，且异常在 try 之外。
4. **`LocalSandbox._respawn_workers` 用 `get_nowait()` 清空队列不可靠**：多进程队列由
   feeder 线程异步投递，残留的陈旧任务会被新 worker 消费（结果管道已关闭 → 白白重拟合
   并连锁超时）。该竞态在测试套件里表现为**偶发失败**。已改为换一条全新队列。
5. **`__file__` 深度锚点**（搬迁引入，测试抓到）：`rag_kb` / `tool_runner` 的仓库根、
   `read_paper` 独立运行的 `sys.path` 兜底都因目录变深而指错——不报错，只是静默指向
   错目录（MCP 子进程 cwd、相对配置全部跑偏）。
6. **`main.py` 覆盖 `config` 模块名**：`config = config.Config(...)` 之后任何
   `config.ClassConfig` 都会 `AttributeError`，靠"恰好用不到"才没炸。
7. **`mcp_server` 把启动逻辑写在 `__main__` 里**：兼容层只能转发名字，
   `python -m 旧路径` 静默起不来——MCP 客户端要等 120s 超时才发现。

### 11.4 可选后续（已在阶段 6 全部完成，见 §12）

- ~~拆分剩余大文件：`analysis/find_best_eq.py`(557)、`agents/sampler_agent.py`(538)、
  `analysis/sensitivity_prune.py`(515)。~~
- ~~为 Agent 层补行为测试（当前主要靠结构护栏 + 既有功能测试）。~~
- ~~兼容层按 [`ARCHITECTURE.md`](./ARCHITECTURE.md) §8 的时机逐层删除。~~

---

## 12. 执行记录（阶段 6：收尾三件事）

阶段 5 之后剩下的三项"可选后续"一次性做完：拆大文件、补 Agent 行为测试、清退兼容层。

### 12.1 拆分三个大文件

| 原文件 | 现在 | 切出的模块（各自单一职责） |
|---|---|---|
| `agents/sampler_agent.py` 538 | **181** | `agents/skeleton.py`（骨架提取）、`agents/prompt_injection.py`（经验/残差注入策略） |
| `analysis/find_best_eq.py` 557 | **116** | `analysis/expr_parse.py`（表达式解析）、`analysis/explain.py`（物理解释）、`analysis/expr_viz.py`（可视化） |
| `analysis/sensitivity_prune.py` 515 | **265** | `analysis/expr_evaluation.py`（求值内核）、`analysis/prune_stats.py`（统计）、`analysis/prune_demo.py`（演示） |

拆分后的收益不止"文件变短"：注入策略、表达式改写规则、求值内核现在都能**单独测试**，
不必再借道一个 500 行的文件（见 §12.2）。顺手删掉两处死代码：`SamplerAgent._draw_samples_api`
（无调用点的第二条采样路径）与 `_conversation_ids`（只写不读的状态）。

### 12.2 Agent 层行为测试（+102 项）

| 文件 | 项数 | 覆盖 |
|---|---|---|
| `tests/test_sampler_agent.py` | 51 | 骨架提取三种回退、注入策略（配额/概率/新鲜度/排序/噪声过滤/预算提示/截断/mtime 缓存）、采样编排（批量与逐条、重采样上界、有界重试、注入失败不拖垮采样） |
| `tests/test_agent_behavior.py` | 51 | 工具循环收敛与参数兜底、反思异常降级、残差上下文与归属字段、初次分析落盘、协调者的质量判定/最优追踪/落盘字段/停止条件/失败容忍 |

全部用替身（假 LLM 客户端、假评估器、假缓冲），不触网、不依赖真实模型。

### 12.3 兼容层清退

删掉 24 个转发模块（顶层 20 + `tools/` 下 4），`drsr_420/` 顶层只剩 `__init__.py`；
`tests/` 与库代码内所有旧路径 import 迁移到规范路径；新护栏
`LegacyPathRemovalTest` 反向锁定"旧路径不得复活"。根目录 `main.py` 由转发 shim 降级为
4 行真入口（IDE 运行配置与两个 `.sh` 都靠它），`llm.py` 只再导出公开 API。
判据、保留理由与迁移期踩过的坑见 [`ARCHITECTURE.md`](./ARCHITECTURE.md) §8。

> 这两个根模块随后也在阶段 7 一并删除（见 §13）：`-m` 已经能表达同一个入口，
> 而仓库根留在 `sys.path` 上的代价（包不自包含、`import llm` 可能被同名 PyPI 包劫持）
> 并不值得。

### 12.4 顺带修掉的缺陷

1. **GBK 控制台崩溃 ×3**（同类缺陷第三次出现）：剪枝演示的 `x²`、剪枝 verbose 日志的 `✂`、
   `read_paper` 试运行输出的 emoji——在 Windows 默认代码页下 `print` 直接抛
   `UnicodeEncodeError` 打断流程。新增 `tests/test_console_encoding.py`：以
   `PYTHONIOENCODING=gbk` 跑 5 个典型入口，把这类问题钉在提交前。
2. **`ResidualAnalyzerAgent` 的三个残差统计量（mean/max/std）算了但从未使用**
   （提示词模板只吃残差矩阵）。为避免改变 LLM 输入与产物结构，**保留计算并加注释说明**，
   不做行为改动。

### 12.5 阶段 6 指标

| 指标 | 阶段 5 结束时 | 现在 |
|---|---|---|
| 测试数 | 262 | **372** |
| 最长文件 | 557（`find_best_eq.py`） | 482（`llm/client.py`，本次未拆分） |
| `drsr_420/` 顶层 `.py` | 21（1 `__init__` + 20 shim） | **1** |
| 包内实现行数 | 未统计 | 9,477（8 层） |
| 兼容转发模块 | 24 | **0** |

---

## 13. 执行记录（阶段 7：清退仓库根的顶层模块）

阶段 6 只清退了**包内**的转发层，仓库根还留着最后两个顶层模块——`main.py`（命令行
入口）与 `llm.py`（旧模块名的再导出）。当时把它们判为"项目的对外接口"，但那份接口
完全可以用 `-m` 表达；真正付出的代价是仓库根必须永远待在 `sys.path` 上。本阶段把
这两个文件也删掉，并把所有引用点同步改成模块方式。

### 13.1 改了什么

| 曾经的用法 | 现在 | 同步修改的引用点 |
|---|---|---|
| `python main.py …` | `python -m drsr_420.cli.main …` | 4 个 `.idea/runConfigurations/*.xml`、`example.sh`、`MRFCompress-3.sh`、`tests/test_console_encoding.py` |
| `import llm` / `python llm.py` | `from drsr_420 import llm` / `python -m drsr_420.llm` | 库代码与 `tests/` 全部走规范路径（阶段 6 已完成迁移） |

IDE 运行配置从**脚本模式**改为**模块模式**：`SCRIPT_NAME` 由 `$PROJECT_DIR$/main.py`
改为 `drsr_420.cli.main`，`MODULE_MODE` 由 `false` 改为 `true`；`WORKING_DIRECTORY`
保持 `$PROJECT_DIR$`，因此 `--data_csv ./data/…` 与 `--llm_config llm.config` 这些
相对路径、以及命令行参数与产物文件名**全程未变**。

### 13.2 新护栏：`RootEntrypointRemovalTest`

阶段 6 的 `LegacyPathRemovalTest` 管包内旧路径，本阶段补上管**包外顶层模块**的三条：

1. 仓库根不得再出现任何 `.py` 模块（这才是"没有下一个顶层 shim"的充分条件）；
2. `main` / `llm` 不得从仓库根解析——断言"解析结果不落在仓库根"而非
   "ModuleNotFoundError"，因为环境里若恰好装了同名的 PyPI 包（`llm` 就是这种情况），
   后者会假失败；
3. 库代码与 `tests/` 不得 import 这两个顶层名。

第 1 条比列出两个具体文件名更强：它顺带堵住"多年后再冒出一个根目录脚本"。

### 13.3 顺带清理的过期表述

转发层消失后，几处注释仍在描述"门面模块"：`drsr_420/llm/__init__.py` 里
"`owner_module()` 供兼容层把属性写入转发到定义处"（已无兼容层，改为提醒打桩必须打在
定义处）、`tool_runner.py` / `read_paper.py` / `data_analyzer_agent.py` /
`cli/main.py` 里指向 `main.py`、`llm.py` 的注释，以及 README / ARCHITECTURE / `data/README.md`
中的命令行示例。

### 13.4 阶段 7 指标

| 指标 | 阶段 6 结束时 | 现在 |
|---|---|---|
| 测试数 | 372 | **375**（`RootEntrypointRemovalTest` 新增 3 条） |
| 仓库根 `.py` 模块 | 2（`main.py` 18 行、`llm.py` 33 行） | **0** |
| `drsr_420.tools.*` 等旧路径 | 0 | 0（未复活） |
| `sys.path` 依赖 | 需"仓库根在搜索路径上" | 无（`python -m` + `drsr420` 命令） |

---

## 14. 执行记录（阶段 8：配置文件三层分离）

阶段 7 之后用户指出"配置文件的管理和使用比较混乱"。诊断结果不是"命名乱"，而是
结构缺陷：**「哪个 Agent 用哪套配置」在代码里没有任何一处声明**，且旧结构把
「角色 → 配置」做成一对一内嵌（一份 `*.config` 同时装密钥、生成参数和一列角色覆盖
`tasks`），**在原理上无法表达"给某个角色换一个模型"**。完整设计与取舍见
[`CONFIG_PLAN.md`](./CONFIG_PLAN.md)（§1–§8 是设计稿，§9 是落地记录与差异说明），
架构侧摘要见 [`ARCHITECTURE.md`](./ARCHITECTURE.md) §10。

### 14.1 一句话结果

三份正交的东西分开存放：`config/<提供商>_<模型>.config`（密钥+端点+生成参数，
不入库）、`config/agents.config.json`（角色 → 档案 + 参数覆盖，**入库**、无密钥）、
`core/config.py`（实验超参，未动）。扩展名即保密边界。

### 14.2 顺带修掉的真实故障

`analysis/explain.py` 硬编码加载 `deepseek_deepseek-v4-flash.config`——该文件**不存在**
→ `FileNotFoundError` 被吞 → `client=None` → `explain_re_act` 直接返回 `None`
→ **`explain.txt` 长期是空的**，全程仅一行 `[WARN]`。原因是旧结构下"给 explain
换模型"无路可走，于是有人建了 `llm_explain.config`（deepseek-v4-pro）与
`llm_summary.config`（本地 ollama）——**两个文件都无人读取**，正是结构缺陷的产物。
新结构下这个意图可以表达了（注册表里加一行 `config`），孤儿文件已被规范化命名收编。

### 14.3 新增护栏

`tests/test_config_roles.py`（43 项）里最关键的一条：**扫描库代码的字符串字面量，
出现写死的 `*.config` 文件名即失败**。它堵的是故障成因（"某个模块自己决定读哪份配置"），
而不只是修复某一个文件；无需任何豁免名单（`.config` 纯后缀、`agents.config.json`、
文档字符串里的散文都不匹配）。

### 14.4 阶段 8 指标

| 指标 | 阶段 7 结束时 | 现在 |
|---|---|---|
| 测试数 | 375 | **418** |
| 全局最长文件 | 482（`llm/client.py`） | 482（首版 `roles.py` 639 行，已按职责拆成 3 个模块） |
| 根目录 `.config` 文件 | 6 | **0** |
| 入库的配置模板 | 1（还是 legacy 名） | **4**（3 个规范名档案 + rag） |
| 库代码里的硬编码档案名 | 3 处 | **0**（有护栏） |
| 无人读取的孤儿档案 | 2 | **0** |

---

## 15. 执行记录（阶段 9：自定义提供商 + 启用角色绑定）

阶段 8 把 `explain` / `summary` 的档案绑定留成了注释里的开关（理由：跟踪文件引用
gitignored 的档案会让新克隆的 `--check` 失败）。本阶段按用户要求打开它，并要求
`summary` 走**自定义提供商**（校内网关）——于是"接一个新端点"必须从"改代码"变成"改档案"。

结果是 `llm/factory.py` 的封闭提供商表被降级为"内置提供商的默认值表"，不再是白名单：
**未知 provider 段 + 档案给了 `base_url` ⇒ 照常构造**（`OpenAICompatClient`）。端点属于
Q1（连接谁 / 用哪把钥匙），按 §14 的三层分红本就归档案。请求体方言随之从 provider 名
推断改为可声明的 `dialect` 字段，并抽成新模块 `drsr_420/llm/adapt.py`（client.py 因此从
482 行降到 472 行——顺带满足"最长文件 < 500"）。

顺带修掉：`extra_body` 从死字段（工厂不注入、适配层又 pop，模板里写了从没上线）变成
**最后并入请求体**的显式出口；`--check` 的密钥问题不再误导性地提示 `cp`；角色表列宽
改为按内容算（26 字符的档案名会把来源列挤成连字）；端点键名统一为 `base_url`
（LLM 侧去掉 `host` 别名、RAG 侧 `api_host` → `api_base_url`，旧拼写**报错**而非静默
忽略——忽略会让请求打到默认端点，或报出与键名无关的 `MissingSchema`）；端点取值也统一
为**完整 URL**（取消"缺 scheme 自动补 `https://`"，`require_absolute_url` 在客户端构造处
一次守住配置/内置默认值/环境变量三条通道），`deepseek` 档案从裸主机 `api.deepseek.com`
改为 `https://api.deepseek.com/v1`，spec 默认值同步。

### 15.1 阶段 9 指标

| 指标 | 阶段 8 结束时 | 现在 |
|---|---|---|
| 测试数 | 418 | **499**（新增 12 条 `tests/test_batch_scripts.py`：XML / `.sh` / `.bat` 三者逐项等价） |
| 全局最长文件 | 482（`llm/client.py`） | **487**（`llm/client.py`；`rag_kb.py` 一度 517，已拆出 `rag_config.py`） |
| `llm/` 层 | 9 文件 / 1818 行 | 11 文件 / 2331 行（新增 `adapt.py` / `stream.py`） |
| 入库的配置模板 | 4 | **5** |
| 建档即可接入的端点 | 7 个内置 | **任意 OpenAI Chat Completions 兼容端点** |
| 端点的键名/写法 | `host` / `base_url` / `api_host`，且允许裸主机 | **`base_url` / `api_base_url`，必须完整 URL** |
| 一份坏档案的后果 | 整个实验起不来 | **只影响用到它的角色**（启动告警，`--ping` 可提前查出） |

### 15.2 收尾：按可用性实测暴露出的三件事（追加）

逐份档案发真实请求之后，暴露的问题都不是"配置写错"那种，而是**故障可诊断性**：

* 网关把错误包在 HTTP 200 里时，异常只说 `API response missing choices`，
  真正有用的 `404 NOT_FOUND` 留在了 print 里 → `client.gateway_error_detail` 把它带进异常；
* 一份用不到的档案（`summary` 缺密钥）会让整个实验起不来 → `RoleClients` 改为**按需构造**，
  `cli.llm_setup` 启动时只告警；严格闸门仍是 `--check`；
* 只有"结构校验"没有"连通性校验" → 新增 `python -m drsr_420.llm.roles --ping`（按档案去重、
  用首个角色的参数与方言、不重试、输出上限 16 token）。

顺带修掉 `locate_config` 把 `config/x.config` 拼成 `config/config/x.config` 的报错路径；
并把 `rag_kb.py` 顶部的档案部分拆到 `knowledge/rag_config.py`（加完端点校验后 517 行，
破了 500 行预算，拆分后 420 行）。

### 15.3 收尾：4 个 MRF 运行配置补齐 .sh / .bat，并修掉 3 份 XML 写错的 background（追加）

`.idea/runConfigurations/` 下的 4 个配置（`MRFShear-Cuboid` / `MRFShear-Ellipsoid` /
`MRFCompress-Cuboid` / `MRFCompress-Ellipsoid`）原先只能在 IDE 里跑，且第 7 阶段之后
仓库根的启动脚本只剩批量示例。现在每组都有三份等价物：XML（参数的源头）、`<名>.sh`
（Linux/macOS）、`<名>.bat`（Windows，可双击），`tests/test_batch_scripts.py` 守住三者
逐项一致，并额外要求 `background` 与 `data/<名>/train.csv` 的表头相符。

顺带发现 3 份 XML 的 `background` 是复制粘贴写坏的——而 background 会直接进提示词，
描述一个数据里不存在的列会误导采样：

| 配置 | XML 原文 | 实际数据列 | 已改为 |
|---|---|---|---|
| `MRFCompress-Ellipsoid` | …lambda12(L1/L2), **and lambda23(L2/L3)**… | `lambda12,sigma` | 只提 lambda12 与 L1/L2 |
| `MRFShear-Ellipsoid` | …lambda12(L1/L2), **and lambda23(L2/L3)**… | `lambda12,miu` | 只提 lambda12 与 L1/L2 |
| `MRFShear-Cuboid` | …短轴 of the **ellipsoid** particle | `lambda12,lambda23,miu` | `cuboid particle` |

`example.sh` 里这 4 条本来就是对的一份，因此以它为准（新建的 `.sh` / `.bat` 也用这一份）。
另外把两条踩过的坑固化成注释与测试：① 批处理的 `call :label 参数` 会把参数里的脱字符
翻倍（1→2、2→4），而 `echo` 会把它显示回 1 个，于是写坏的值只在真正传给 Python 时才
暴露——含 LaTeX `^{...}` 的 background 必须走变量传递；② `.bat` 必须是 CRLF（cmd 的
`call`/`goto` 按字节偏移定位标签），由新增的 `.gitattributes` 固定。

完整设计与取舍见 [`CONFIG_PLAN.md`](./CONFIG_PLAN.md) §10，架构摘要见
[`ARCHITECTURE.md`](./ARCHITECTURE.md) §10.5。
