# DRSR 多 Agent 系统 —— agents 子包使用手册

本包把 DRSR（Deep Reasoning Symbolic Regression）主循环中的协作角色拆分为独立的
Agent 类，每个文件对应一个角色。所有 Agent 由 `CoordinatorAgent` 统一编排，
体现"感知 → 决策 → 行动 → 反思"的多智能体闭环。

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

| 文件 | Agent | 一句话职责 |
|---|---|---|
| [coordinator_agent.py](./coordinator_agent.py) | `CoordinatorAgent` | 协调者：驱动"采样→评估→反思→持久化"主循环，多线程并行 |
| [sampler_agent.py](./sampler_agent.py) | `SamplerAgent` | 采样者：LLM 生成方程骨架（含经验/残差注入与空骨架重采样） |
| [tool_caller_agent.py](./tool_caller_agent.py) | `ToolCallerAgent` | 工具调用者：与 LLM 多轮对话并执行 MCP 工具调用 |
| [evaluator_agent.py](./evaluator_agent.py) | `EvaluatorAgent` | 评估者：编译骨架、常驻沙箱执行、多起点拟合打分 |
| [experience_summarizer_agent.py](./experience_summarizer_agent.py) | `ExperienceSummarizerAgent` | 经验总结者：分析样本质量，产出改进建议 |
| [residual_analyzer_agent.py](./residual_analyzer_agent.py) | `ResidualAnalyzerAgent` | 残差分析者：残差统计 + 结构化修正方向 |
| [data_analyzer_agent.py](./data_analyzer_agent.py) | `DataAnalyzerAgent` | 数据分析者：初次数据分析（sample_order=0 基线） |

> 旧模块名（`drsr_420.sampler` / `drsr_420.evaluator` 等）由兼容层 re-export，
> 外部代码与测试无需改动。

---

## 1. CoordinatorAgent —— 协调者

**文件**：[coordinator_agent.py](./coordinator_agent.py)

**职责说明**

- 是唯一持有全链路编排权的角色：每轮从共享记忆 `ExperienceBuffer` 取 prompt，
  委托各 Agent 完成 采样 → 评估 → 质量分类 → 经验总结 → 残差分析 → 持久化。
- 支持多线程并行：pipeline 以 `Sampler-i` 线程启动多个实例，实例间共享
  `ExperienceBuffer` 与全局采样计数，所有共享文件读写与计数由可重入锁
  `_SAMPLER_LOCK` 保护。
- 为不同任务维护**独立 LLM 客户端副本**（采样 / 经验 / 残差各一份），
  避免修改同一客户端导致温度等参数互相覆盖。
- 断点续跑：每轮写 `checkpoint.json`，进度写 `round_progress.csv`。

**关键接口**

```python
CoordinatorAgent(
    database: buffer.ExperienceBuffer,          # 共享记忆（多岛经验缓冲）
    evaluators: Sequence[EvaluatorAgent],       # 本 sampler 独享的评估 Agent 列表
    samples_per_prompt: int,                    # 每轮采样数量
    config: config_lib.Config,                  # 全局配置（结果目录/超时/经验注入超参）
    max_sample_nums: int | None = None,         # 全局采样数上限；None 表示不停止
    llm_class: Type[LLM] = LLM,                 # 采样器类（默认 LLM 基类）
    prompt_ctx: pc.PromptContext | None = None, # 动态提示词上下文（变量名/因变量等）
    llm_client: LLMClient | None = None,        # 基础 LLM 客户端（内部按任务克隆）
    llm_api: dict | None = None,
)
coordinator.sample(**kwargs)                    # 运行主循环（如 kwargs={'profiler': profiler}）
coordinator.set_global_sample_nums(n)           # 设置全局采样计数（断点恢复时使用）
```

**调用示例**（见 [pipeline.py](../pipeline.py) `_launch_samplers`）

```python
import threading
from drsr_420.agents.coordinator_agent import CoordinatorAgent

coordinator = CoordinatorAgent(
    database=database,              # 共享 ExperienceBuffer
    evaluators=evals,               # 本线程独享的 EvaluatorAgent 列表
    samples_per_prompt=config.samples_per_prompt,
    max_sample_nums=max_sample_nums,
    llm_class=class_config.llm_class,
    config=config,
    prompt_ctx=prompt_ctx,
    llm_client=llm_client,
    llm_api=None,
)
# 多线程并行：每个 Sampler-i 线程跑一个 CoordinatorAgent
t = threading.Thread(
    target=coordinator.sample, kwargs={'profiler': profiler},
    daemon=True, name=f"Sampler-{i}",
)
t.start()
```

---

## 2. SamplerAgent —— 采样者

**文件**：[sampler_agent.py](./sampler_agent.py)

**职责说明**

- 负责把"指令 + 任务头 + 历史经验/残差注入"拼成最终提示词（`_build_request_content`），
  交给 `ToolCallerAgent` 与 LLM 多轮对话，拿到候选方程骨架。
- 骨架后处理：从 LLM 混合输出（文字 + 代码）中抽取可执行函数体
  （`_extract_body` / `_extract_code_fragment`）；抽不到时自动重采样
  （最多 `_MAX_BODY_RETRIES = 3` 次），仍无效则丢弃该样本（不进评估队列）。
- 经验注入规则：None（失败教训）始终注入；Good/Bad 按概率参与；
  超过新鲜度阈值后只注入近期经验；Good 按 score 降序、Bad 按 score 升序截断。
- 继承 `LLM` 抽象基类（`samples_per_prompt` 决定每批数量）。

**关键接口**

```python
SamplerAgent(
    samples_per_prompt: int,                    # 每个 prompt 生成几个骨架
    batch_inference: bool = True,               # 批量推理（一次问 LLM 拿 N 个答案）
    trim: bool = True,                          # 是否抽取/裁剪函数体
    prompt_ctx: pc.PromptContext | None = None, # 动态提示词上下文
    llm_client: LLMClient | None = None,        # LLM 客户端
)
sampler.draw_samples(prompt: str, config: config_lib.Config)
# -> tuple[list[str], list[str]] | None   返回 (骨架列表, 思考内容列表)
```

**调用示例**

```python
from drsr_420.agents.sampler_agent import SamplerAgent

sampler = SamplerAgent(
    samples_per_prompt=4,
    batch_inference=True,
    prompt_ctx=prompt_ctx,      # 无则用 pc.instruction_prompt 默认提示
    llm_client=llm_client,
)
samples, thinking_contents = sampler.draw_samples(prompt.code, config)
# samples = ['def equation(x1, x2, params):\n    return ...', ...]
```

---

## 3. ToolCallerAgent —— 工具调用者

**文件**：[tool_caller_agent.py](./tool_caller_agent.py)

**职责说明**

- 与 LLM 多轮对话：若模型返回 `tool_calls`，则逐条执行工具并把结果回传给模型，
  直到模型给出最终答复。
- 工具执行默认走 `tool_runner.mcp_call_tool`（MCP 服务器封装，可注入自定义执行器）。
- 轮次上限 `max_tool_rounds`（默认 4）：防止模型无限检索文献拖死采样；
  到达上限时以当前响应强制返回。
- 兜底：`content` 为空时回退 `reasoning_content`，避免模型只思考不输出正文时骨架丢失。

**关键接口**

```python
ToolCallerAgent(
    llm_client: LLMClient,                      # 提供 chat(messages, on_delta=...) -> dict
    tool_executor: callable | None = None,      # (name, args) -> str；默认 mcp_call_tool
    max_tool_rounds: int = 4,                   # 单个样本工具调用轮次上限
)
tool_caller.complete(content: str, repeat: int = 1)
# repeat==1  -> (response_str, think_str)
# repeat>1  -> (responses_list, think_list)
```

**调用示例**

```python
from drsr_420.agents.tool_caller_agent import ToolCallerAgent

tool_caller = ToolCallerAgent(llm_client, max_tool_rounds=4)
content = "Please search papers about magnetorheological elastomer ..."
responses, thinking = tool_caller.complete(content, repeat=4)   # 批量取 4 个骨架
# 单次：resp, think = tool_caller.complete(content, repeat=1)
```

---

## 4. EvaluatorAgent —— 评估者

**文件**：[evaluator_agent.py](./evaluator_agent.py)

**职责说明**

- 把 `SamplerAgent` 产出的骨架编译成可运行程序（`_sample_to_program` + `_trim_function_body`），
  在常驻子进程沙箱 `LocalSandbox` 中执行，多起点 `least_squares` 拟合参数。
- 产出统一三元组 `(score, error, residual)` 供 `CoordinatorAgent` 分类 Good/Bad/None；
  失败时返回 `(None, None, None)`。
- 成功样本注册进 `ExperienceBuffer`（`register_program`），失败样本经 Profiler 记录。
- `LocalSandbox` 常驻 worker 复用进程池，超时/崩溃时销毁重建，避免 Windows spawn 开销。
- numba 可选加速：编译失败自动降级为纯 Python 执行。

**关键接口**

```python
EvaluatorAgent(
    database: buffer.ExperienceBuffer,          # 经验缓冲（注册成功程序）
    template: code_manipulation.Program,        # spec 编译出的程序模板
    function_to_evolve: str,                    # 待进化的函数名（@equation.evolve）
    function_to_run: str,                       # 运行函数名（@evaluate.run）
    inputs: Sequence[Any],                      # 测试数据集
    timeout_seconds: int = 30,                  # 单样本评估超时
    sandbox_class: Type[Sandbox] = Sandbox,     # 默认 Sandbox；常用 LocalSandbox
)
evaluator.analyse(sample, island_id, version_generated, **kwargs)
# -> tuple[float|None, str, Any]   返回 (score, error_msg, residual)
```

**调用示例**（见 [pipeline.py](../pipeline.py) `_init_evaluators`）

```python
from drsr_420.agents.evaluator_agent import EvaluatorAgent, LocalSandbox

evaluator = EvaluatorAgent(
    database=database,
    template=template,
    function_to_evolve=function_to_evolve,
    function_to_run=function_to_run,
    inputs=inputs,                              # 数据实例
    timeout_seconds=config.evaluate_timeout_seconds,
    sandbox_class=LocalSandbox,                 # 常驻 worker 沙箱
)
score, error_msg, residual = evaluator.analyse(
    sample, island_id=0, version_generated=1, profiler=profiler)
# score: 拟合后的负均方误差；error_msg: 失败原因（如 'Execution Error: ...'）
```

---

## 5. ExperienceSummarizerAgent —— 经验总结者

**文件**：[experience_summarizer_agent.py](./experience_summarizer_agent.py)

**职责说明**

- 对一批样本及质量标签（Good/Bad/None）逐个构造分析提示并调用 LLM，
  把"得分高/低/出错"的样本转成结构化分析文本（改进建议）。
- 输出写入 `experiences.json`（由 `CoordinatorAgent` 持久化），
  供下一轮采样注入提示词。
- 有 `prompt_ctx` 时用动态模板渲染提问（变量名/因变量贴合任务），
  无则用默认模板；None 类问题附带"参数预算"告诫，避免模型要更多参数。

**关键接口**

```python
ExperienceSummarizerAgent(
    llm_client: LLMClient,                      # 经验分析客户端（高思考强度）
    prompt_ctx: pc.PromptContext | None = None, # 动态提示词上下文
)
summarizer.analyze(samples, quality_for_sample, error_for_sample, prompt) -> list[str]
# samples: 骨架列表；quality_for_sample: ['Good'|'Bad'|'None']；prompt: 原始提示（含 .code）
```

**调用示例**

```python
from drsr_420.agents.experience_summarizer_agent import ExperienceSummarizerAgent

summarizer = ExperienceSummarizerAgent(llm_client, prompt_ctx=prompt_ctx)
analyses = summarizer.analyze(
    samples=batch.samples,            # 本轮骨架
    quality_for_sample=batch.qualities,  # ['Good', 'Bad', 'None', ...]
    error_for_sample=batch.errors,    # None 类别才有效
    prompt=batch.prompt,              # 有 .code 属性
)
# analyses: 与 samples 一一对应的 LLM 分析文本列表
```

---

## 6. ResidualAnalyzerAgent —— 残差分析者

**文件**：[residual_analyzer_agent.py](./residual_analyzer_agent.py)

**职责说明**

- 对当前最优样本的残差矩阵做统计（均值 / 最大绝对值 / 标准差），
  读取上一次 `residual_analyze.json` 分析作为上下文，拼进提示词让 LLM
  输出结构化修正方向。
- 输出由 `CoordinatorAgent` 追加写回 `residual_analyze.json`
  （含 `sample_order`、`equation`、`best_score` 等字段）。

**关键接口**

```python
ResidualAnalyzerAgent(
    llm_client: LLMClient,                      # 残差分析客户端（高思考强度）
    prompt_ctx: pc.PromptContext | None = None, # 动态提示词上下文
    results_root: str = '.',                    # 结果目录（读取/关联 residual_analyze.json）
)
analyzer.analyze(sample: str, residual: np.ndarray) -> str
# residual: 残差矩阵（最后一列为残差值），返回 LLM 分析文本
```

**调用示例**

```python
import numpy as np
from drsr_420.agents.residual_analyzer_agent import ResidualAnalyzerAgent

analyzer = ResidualAnalyzerAgent(
    llm_client=llm_client, prompt_ctx=prompt_ctx, results_root=results_root)
# residual: shape (N, n_feat + 1)，最后一列 = 预测残差
analysis = analyzer.analyze(best_sample, best_residual)
```

---

## 7. DataAnalyzerAgent —— 数据分析者

**文件**：[data_analyzer_agent.py](./data_analyzer_agent.py)

**职责说明**

- 实验开始时对目标数据集做初次分析：读 CSV 或数据字典（`{'data': {'inputs', 'outputs'}}`），
  随机采样（默认 100 行）+ 保留小数（默认 3 位），拼进提示词请 LLM
  分析"自变量对因变量的影响关系 + 自变量间潜在关系"。
- 分析结果写入 `residual_analyze.json` 的 `sample_order=0` 初始记录，
  是后续残差分析 / 经验注入链路的起点。
- 支持自定义提示（含 `{csv_data}` 占位符）；`verbose=True` 时打印提示词并落盘。

**关键接口**

```python
DataAnalyzerAgent(
    api_url: str = 'http://127.0.0.1:5000/completions',  # 兼容旧接口（主流程用 llm_client）
    timeout: int = 300,             # API 请求超时（秒）
    decimal_places: int | None = None,  # 保留小数位数（默认 3）
    sample_size: int | None = None,     # 随机采样行数（默认 100）
    base_dir: str | None = None,        # 结果目录（写 residual_analyze.json）
    llm_client: LLMClient | None = None,# 注入的 LLM 客户端（优先于 api_url）
    seed: int | None = None,            # 随机种子（可复现采样）
)
analyzer.analyze(data_source, custom_prompt=None, max_rows=None, verbose=True) -> str
# data_source: CSV 路径(str) 或 {'data': {'inputs': X, 'outputs': y}}(dict)
```

**调用示例**（见 [pipeline.py](../pipeline.py) `_run_initial_analysis`）

```python
from drsr_420.agents.data_analyzer_agent import DataAnalyzerAgent

analyzer = DataAnalyzerAgent(
    timeout=600,
    base_dir=results_root,
    llm_client=llm_client,
    seed=seed,
)
result = analyzer.analyze(
    inputs,                                 # 数据字典：{'data': {'inputs': X, 'outputs': y}}
    initial_analysis_prompt,                # PromptContext 动态渲染 + RAG 文献注入后的提示
    verbose=True,
)
```

---

## 线程安全与共享状态约定

- `_SAMPLER_LOCK`（`coordinator_agent.py`，可重入 RLock）保护：全局采样计数
  `CoordinatorAgent._global_samples_nums` 与 `experiences.json` /
  `residual_analyze.json` / `checkpoint.json` 的读-改-写。
- 每个 Sampler 线程独享一份 `EvaluatorAgent` 列表（避免 `LocalSandbox._last_params`
  等实例状态竞态）；`ExperienceBuffer` 由所有线程共享（内部自带上锁）。
- LLM 客户端按任务克隆独立副本（`clone_llm_client`），采样/经验/残差互不影响。
- 包内 `__init__.py` 刻意不做任何 import，防止循环导入；
  旧模块名由兼容层 re-export 到本包。
