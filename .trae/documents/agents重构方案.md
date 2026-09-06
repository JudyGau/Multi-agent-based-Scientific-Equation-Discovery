# DRSR 多 Agent 系统中度重构方案

## Context（背景）

本项目是一个多 agent 科学方程发现系统（DRSR）：LLM 采样方程骨架 → 评估器多起点最小二乘拟合参数评分 → 经验总结/残差分析反馈 → 注入下一轮采样。虽然系统本身是多 agent 协作，但现有代码组织结构松散：

- `drsr_420/sampler.py` 混装了三类角色：`Sampler`（采样 agent）、`SamplingOrchestrator`（协调 agent，`sample()` 约 300 行巨型方法）、经验注入等工具函数；
- `drsr_420/evaluator.py` 混装了 `Evaluator`（评估 agent）与 `Sandbox`/`LocalSandbox`（沙箱基础设施）；
- `pipeline.py` 的 `main()` 约 160 行，包含数据准备、初次分析、线程启动等杂糅逻辑；
- 各 agent 角色无统一命名后缀，难以一眼看出"谁是谁、谁和谁协作"。

用户诉求：**提高代码易懂性，体现多 agent 系统**。已确认三个决策：

1. **中度重构**：新建 `drsr_420/agents/` 子包，各 agent 角色独立成文件，拆解巨型方法，统一命名（`*Agent` 后缀），不过度抽象（不引入消息队列/Agent 基类强约束）。
2. **保持兼容**：旧模块名（`drsr_420.sampler`、`drsr_420.evaluator` 等）保留为 re-export 兼容层，`tests/` 不改也能跑。
3. **保留现状日志**：重构只调结构，不删任何调试 print。

## 目标包结构

```
drsr_420/agents/
├── __init__.py                       # 包 docstring + agent 协作图（无 import，防循环导入）
├── tool_caller_agent.py              # ToolCallerAgent（= ToolCaller）
├── sampler_agent.py                  # LLM(ABC) + SamplerAgent（= Sampler）
├── experience_summarizer_agent.py    # ExperienceSummarizerAgent（= ExperienceSummarizer）
├── residual_analyzer_agent.py        # ResidualAnalyzerAgent（= ResidualAnalyzer）
├── evaluator_agent.py                # EvaluatorAgent + Sandbox(ABC) + LocalSandbox（= Evaluator 等）
├── data_analyzer_agent.py            # DataAnalyzerAgent（= DataAnalyzer）
└── coordinator_agent.py              # CoordinatorAgent（= SamplingOrchestrator）+ SampleBatch
```

角色对应关系（`= 旧名` 表示兼容别名，旧名与新类是同一对象，`isinstance`/`Type[]` 注解均成立）：

| agents/ 文件 | 新类名 | 兼容别名 |
|---|---|---|
| tool_caller_agent.py | `ToolCallerAgent` | `ToolCaller` |
| sampler_agent.py | `SamplerAgent(LLM)`、`LLM(ABC)` | `Sampler` |
| experience_summarizer_agent.py | `ExperienceSummarizerAgent` | `ExperienceSummarizer` |
| residual_analyzer_agent.py | `ResidualAnalyzerAgent` | `ResidualAnalyzer` |
| evaluator_agent.py | `EvaluatorAgent`、`Sandbox`、`LocalSandbox` | `Evaluator` |
| data_analyzer_agent.py | `DataAnalyzerAgent` | `DataAnalyzer` |
| coordinator_agent.py | `CoordinatorAgent` + `SampleBatch` dataclass | `SamplingOrchestrator` |

不纳入 agents/：`prompt_config.py`（提示词渲染基础设施）、`buffer.py`（共享记忆）、`profile.py`、`console.py`、`tool_runner.py`、`rag_kb.py`、`find_best_eq.py`（实验收尾工具函数）、`evaluate_on_problems.py`（纯数值工具）。

## 各文件迁移要点

### 1. `agents/sampler_agent.py`

原样搬移 `drsr_420/sampler.py` 中的：`LLM(ABC)`、`Sampler`（改名为 `SamplerAgent`）、`_extract_body`、`_extract_code_fragment`、`_MAX_BODY_RETRIES`。将内部 `from drsr_420.tool_caller import ToolCaller` 改为 `from drsr_420.agents.tool_caller_agent import ToolCallerAgent`。文件末尾 `Sampler = SamplerAgent` 供兼容层 re-export。

### 2. `agents/experience_summarizer_agent.py` / `residual_analyzer_agent.py`

原类原样搬移 + 类名加 `Agent` 后缀（`ExperienceSummarizerAgent`、`ResidualAnalyzerAgent`），签名不变，import 不变。

### 3. `agents/evaluator_agent.py`

搬移 `drsr_420/evaluator.py` 全部内容：`_FunctionLineVisitor`、`_trim_function_body`、`_sample_to_program`、`Sandbox(ABC)`、`LocalSandbox`、`_eval_worker`、`_run_evaluation_task`、`_sample_residuals`、`Evaluator`（改名 `EvaluatorAgent`）。`import profile` 原样保留（注解从不求值，不做额外修正以免扩大 diff）。

### 4. `agents/data_analyzer_agent.py`

搬移 `drsr_420/data_analyse_real.py` 的 `DataAnalyzer`（改名 `DataAnalyzerAgent`），保留 `Port = '5000'` 常量。

### 5. `agents/tool_caller_agent.py`

搬移 `drsr_420/tool_caller.py` 的 `ToolCaller`（改名 `ToolCallerAgent`）。

### 6. `agents/coordinator_agent.py`（核心）

搬移 `SamplingOrchestrator`（改名 `CoordinatorAgent`）+ 模块级 `_SAMPLER_LOCK`、`_atomic_write_json`（改公开名 `atomic_write_json`）、`_clone_llm_client`（改公开名 `clone_llm_client`）。**保留**：

- 类属性 `_global_samples_nums`、`set_global_sample_nums`（pipeline 以类方式调用 `SamplingOrchestrator.set_global_sample_nums(n)`，别名后自动成立）；
- `_get_global_sample_nums` / `_global_sample_nums_plus_one` / `_checkpoint_path` / `_save_checkpoint` / `_append_progress`；
- `__init__` 中的三个 LLM 客户端克隆（sampling/experience/residual 各自独立 temperature，来自 llm.py `clone_for_task`）。

**删除死代码**：`__init__` 中 `if llm_api:` 的 `global API_HOST...` 覆盖块（pipeline 恒传 `llm_api=None`，不可达）。

**巨型方法拆分**：`sample()` 拆为 8 个语义化内部方法，`sample()` 退化为驱动器：

```python
def sample(self, **kwargs) -> None:
    start_time = time.time()
    while not self._should_stop(start_time):
        prompt = self._database.get_prompt()
        island_id = prompt.island_id
        best_score = self._database._best_score_per_island[island_id]   # 评估前捕获，不随循环更新
        batch = self._sample_batch(prompt)
        self._evaluate_batch(batch, **kwargs)
        self._classify_quality(batch, best_score)
        try:  # 分析失败不中断主循环（对应现状 L722-840 外层 try/except）
            self._summarize_experience(batch)
            self._analyze_residual(batch)
            self._persist_experiences(batch)
        except Exception as e:
            print(f"执行分析时出错: {e}"); traceback.print_exc()
        self._save_checkpoint()
        self._append_progress(island_id, start_time)
```

| 方法 | 职责 |
|---|---|
| `_should_stop(start_time) -> bool` | 时长上限 + 全局采样数上限判断 |
| `_sample_batch(prompt) -> SampleBatch` | 调 `self._llm.draw_samples(...)`，计算 sample_time，封装批次 |
| `_evaluate_batch(batch, **kwargs) -> None` | 逐样本：全局计数+1、随机选 evaluator、`analyse(...)`，记录 scores/errors，追踪本轮最优（best_id 为 **1-based**） |
| `_classify_quality(batch, best_score) -> None` | None→'None'；`> best_score`→'Good'；否则 'Bad' |
| `_summarize_experience(batch) -> None` | 调 `self._summarizer.analyze(...)` |
| `_analyze_residual(batch) -> None` | 仅当 `best_sample` 且 `best_residual is not None` 时调 `self._analyzer.analyze(...)` 再持久化 |
| `_persist_residual(batch, analysis) -> None` | 锁内读-改-写 `residual_analyze.json`；`sample_order = _get_global_sample_nums() - len(samples) + best_id` |
| `_persist_experiences(batch) -> None` | 锁内读-改-写 `experiences.json`，按 Good/Bad/None 分类；`sample_order = _get_global_sample_nums() - len(samples) + i + 1` |

`SampleBatch` dataclass 字段：`prompt, samples, thinking_contents, sample_time, scores, errors, qualities, analyses, best_sample=None, best_residual=None, best_id=None, best_score=None`。

**行为保持红线**（拆分最易踩坑）：
- 所有 print / `print_block` 原样保留（用户要求不删日志）；
- 质量分类用**评估前**的 best_score，不是循环内更新值；
- `_persist_experiences` 用 `i+1`（zip 下标），`_persist_residual` 用 `best_id`——两者公式不同，勿统一；
- 现状 L735 `if_best = False` 复位是无效局部操作，拆分后自然消失。

## 兼容层（re-export shim）

6 个旧模块整体替换为仅 re-export，**不保留旧实现**：

- `drsr_420/sampler.py` → `from drsr_420.agents.sampler_agent import LLM, SamplerAgent as Sampler, _extract_body, _extract_code_fragment` + `from drsr_420.agents.coordinator_agent import CoordinatorAgent as SamplingOrchestrator, _SAMPLER_LOCK, atomic_write_json as _atomic_write_json, clone_llm_client as _clone_llm_client`
- `drsr_420/evaluator.py` → 全部名字：`EvaluatorAgent as Evaluator, Sandbox, LocalSandbox, _FunctionLineVisitor, _trim_function_body, _sample_to_program, _eval_worker, _run_evaluation_task, _sample_residuals`（注意 `tests/test_evaluator.py` 依赖 `LocalSandbox, _run_evaluation_task, _sample_residuals`）
- `drsr_420/experience_summarizer.py` → `ExperienceSummarizerAgent as ExperienceSummarizer`
- `drsr_420/residual_analyzer.py` → `ResidualAnalyzerAgent as ResidualAnalyzer`
- `drsr_420/tool_caller.py` → `ToolCallerAgent as ToolCaller`
- `drsr_420/data_analyse_real.py` → `DataAnalyzerAgent as DataAnalyzer`（保留 `Port = '5000'`）

**不需要改动**：`config.py`（保持 `from drsr_420 import sampler/evaluator`，经 shim 指向 agents）、`main.py`、`tests/` 全部文件。

## pipeline.py 重构

`main()` 抽 5 个语义化步骤，本体收敛为约 10 行：

```python
def main(specification, inputs, config, max_sample_nums, class_config, **kwargs):
    db, template, function_to_evolve, function_to_run = _init_experience_buffer(specification, config, kwargs)
    profiler = _init_profiler(inputs, config, kwargs)
    evaluators = _init_evaluators(db, template, function_to_evolve, function_to_run, inputs, config, class_config)
    _run_initial_analysis(inputs, config, kwargs, evaluators, profiler)
    _launch_samplers(db, template, function_to_evolve, function_to_run, inputs, config,
                     max_sample_nums, class_config, kwargs, profiler)
    find_best_eq(results_root)
```

- `_init_experience_buffer`：L75-95（提取函数名、text_to_program、ExperienceBuffer、checkpoint 恢复含 `CoordinatorAgent.set_global_sample_nums`）；
- `_init_profiler`：L96-113（target_variance + Profiler）；
- `_init_evaluators`：L116-126（构造 EvaluatorAgent 列表）；
- `_run_initial_analysis`：L128-184（初始模板 analyse、DataAnalyzerAgent 构造、RAG 注入、analyze；**删除死代码 `_dar.API_*` 覆盖块**）；
- `_launch_samplers`：L186-223（构造 num_samplers 个 CoordinatorAgent + threading.Thread 启动 join）。

import 改为：`from drsr_420.agents.coordinator_agent import CoordinatorAgent`、`from drsr_420.agents.evaluator_agent import EvaluatorAgent`、`from drsr_420.agents.data_analyzer_agent import DataAnalyzerAgent`。`buffer`、`profile`、`code_manipulation`、`find_best_eq` 保持现状，`_extract_function_names` 保留。

## 文档（体现多 agent）

1. `agents/__init__.py` docstring 放协作图（IDE 悬停包名可见）：

```
DataAnalyzerAgent ──初次数据分析──▶ residual_analyze.json (sample_order=0)
         │
CoordinatorAgent (Sampler-i 线程，共享 ExperienceBuffer / _SAMPLER_LOCK / 全局采样计数)
  ├─▶ SamplerAgent ──▶ ToolCallerAgent ──▶ LLMClient ──▶ MCP 工具 (tool_runner)
  ├─▶ EvaluatorAgent ──▶ LocalSandbox(常驻 worker) ──▶ evaluate_on_problems(多起点 least_squares)
  ├─▶ ExperienceSummarizerAgent ──▶ experiences.json
  └─▶ ResidualAnalyzerAgent ──▶ residual_analyze.json
收尾：find_best_eq()（工具函数，非 agent）
```

2. 每个 agent 类用三段式 docstring：**角色一句话** / **职责列表** / **协作（上游、下游）**。示例见 Plan 阶段给出的 CoordinatorAgent docstring。

3. README.md 增补 "v2 agents 架构" 小节（复用同一协作图）+ 旧→新模块映射表。

## 实施步骤

**阶段 0 基线**：项目根目录 `python -B tests/run_tests.py`，记录绿状态。

**阶段 1 建 agents/ 包**（纯新增，不动旧代码，按依赖顺序）：
1. `agents/__init__.py`：docstring + 协作图，**无任何 import**；
2. `agents/tool_caller_agent.py`（叶子）；
3. `agents/sampler_agent.py`；
4. `agents/experience_summarizer_agent.py`、`agents/residual_analyzer_agent.py`；
5. `agents/evaluator_agent.py`；
6. `agents/data_analyzer_agent.py`；
7. `agents/coordinator_agent.py`（含 sample() 拆分 + SampleBatch）。

**阶段 2 兼容层替换**：6 个旧模块重写为 shim。

**阶段 3 pipeline.py 重构**：按上文抽取步骤、改 import。

**阶段 4 文档**：README.md 增补架构小节与映射表；逐个 agent 类补三段式 docstring（阶段 1 中已含，此处复查）。

**阶段 5 收尾**：全量 `git diff` 对照确认 6 个 shim 无残留旧实现；最终验证。

## 验证

```powershell
# 项目根目录
python -B tests/run_tests.py
python -B -c "import drsr_420.pipeline, drsr_420.config, drsr_420.sampler, drsr_420.evaluator; from drsr_420.evaluator import LocalSandbox, _run_evaluation_task, _sample_residuals; from drsr_420.sampler import Sampler, SamplingOrchestrator; from drsr_420.agents.coordinator_agent import CoordinatorAgent; print('imports OK')"
python -B -c "import drsr_420.sampler as s; from drsr_420.agents import sampler_agent as sa; assert s.LLM is sa.LLM and s.Sampler is sa.SamplerAgent; print('shim identity OK')"
```

## 风险与对策

- **循环导入（最高风险）**：现状依赖环 `config ⇄ sampler ⇄ evaluator ⇄ buffer` 靠延迟注解存活。对策：`agents/__init__.py` 零 import；agents 内部一律 `from drsr_420.agents.xxx import ...`，不得反向 import 兼容层；需要 config 时用 `from drsr_420 import config as config_lib`（仅注解使用，环内安全）。
- **全局状态归属**：`_SAMPLER_LOCK` 必须是 `coordinator_agent.py` 模块级单例；shim 替换后旧锁消失，仅剩一把。
- **行为漂移**：逐方法迁移后与旧代码 `git diff -w` 对照；质量分类用评估前 best_score；两个 sample_order 公式不统一。
- **Windows spawn 多进程**：`LocalSandbox` 常驻 worker 依赖 `__main__` 保护；新文件模块顶层不得有副作用代码（建进程/写文件）。
- **`import profile`**：evaluator 迁移时原样保留。

## 涉及文件清单

**新建**：
- `drsr_420/agents/__init__.py`
- `drsr_420/agents/tool_caller_agent.py`
- `drsr_420/agents/sampler_agent.py`
- `drsr_420/agents/experience_summarizer_agent.py`
- `drsr_420/agents/residual_analyzer_agent.py`
- `drsr_420/agents/evaluator_agent.py`
- `drsr_420/agents/data_analyzer_agent.py`
- `drsr_420/agents/coordinator_agent.py`

**重写为 shim**：`drsr_420/sampler.py`、`drsr_420/evaluator.py`、`drsr_420/experience_summarizer.py`、`drsr_420/residual_analyzer.py`、`drsr_420/tool_caller.py`、`drsr_420/data_analyse_real.py`

**重构**：`drsr_420/pipeline.py`

**文档**：`README.md`（增补架构小节）
