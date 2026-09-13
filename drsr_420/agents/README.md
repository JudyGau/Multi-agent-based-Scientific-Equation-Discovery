# DRSR 多 Agent 系统 —— 角色手册

本手册是**每个 Agent 的接口与用法**参考。架构总览（分层结构、依赖规则、协作时序、
共享状态）在 [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md)，请勿在此重复维护。

```
python -m drsr_420.agents            # 打印组织图（由 SPEC 渲染，不会与代码漂移）
python -m drsr_420.agents --check    # 契约自检：上下游引用 / 可达性 / 线程模型一致性
```

| 角色 | 类 | 规范入口 | 一句话职责 |
|---|---|---|---|
| 协调者 | [`CoordinatorAgent`](./coordinator_agent.py) | `sample(profiler)` | 驱动"采样→评估→反思→持久化"主循环，多线程并行 |
| 采样者 | [`SamplerAgent`](./sampler_agent.py) | `draw_samples(prompt, config)` | LLM 生成方程骨架（含经验/残差注入与空骨架重采样） |
| 工具调用者 | [`ToolCallerAgent`](./tool_caller_agent.py) | `complete(content, repeat)` | 与 LLM 多轮对话并执行 MCP 工具调用 |
| 评估者 | [`EvaluatorAgent`](./evaluator_agent.py) | `analyze(EvaluationRequest)` | 编译骨架、委托沙箱执行、多起点拟合打分 |
| 经验总结者 | [`ExperienceSummarizerAgent`](./experience_summarizer_agent.py) | `analyze(...)` | 分析样本质量，产出改进建议 |
| 残差分析者 | [`ResidualAnalyzerAgent`](./residual_analyzer_agent.py) | `analyze(sample, residual)` | 残差统计 + 结构化修正方向 |
| 数据分析者 | [`DataAnalyzerAgent`](./data_analyzer_agent.py) | `analyze(data_source, prompt)` | 初次数据分析（`sample_order=0` 基线） |

Agent 之间传递的数据类型定义在 [`messages.py`](./messages.py)。

---

## 0. 契约：每个 Agent 的「角色卡」

每个 Agent 继承 [`base.BaseAgent`](./base.py) 并以类属性 `SPEC: AgentSpec` 声明：
key / 角色 / 使命 / 入口方法 / 上下游 / 输入输出契约 / 落盘产物 / 线程模型 /
使用的 LLM 客户端副本。

```python
from drsr_420.agents import SamplerAgent, agent_specs

SamplerAgent.SPEC.key          # 'sampler'
SamplerAgent.SPEC.entrypoints  # ('draw_samples',)
SamplerAgent.SPEC.produces     # ('samples: list[str]', 'thinking_contents: list[str]')
agent_specs()                  # {key: AgentSpec}，系统里有几个 Agent 由代码回答
```

`BaseAgent.__init_subclass__` 在**类定义时**校验：漏声明 `SPEC`、`SPEC` 声明的入口
不存在、`thread_model` 取值非法——都会在 import 阶段直接报错。

---

## 1. CoordinatorAgent —— 协调者

**文件**：[coordinator_agent.py](./coordinator_agent.py)

**职责**

- 唯一持有全链路编排权的角色：每轮从共享记忆取 prompt，委托各 Agent 完成
  采样 → 评估 → 质量分类 → 经验总结 → 残差分析 → 持久化。
- 多线程并行：pipeline 以 `Sampler-i` 线程启动多个实例，实例间共享
  `ExperienceBuffer` 与全局采样计数（由可重入锁 `_SAMPLER_LOCK` 保护）。
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
coordinator.sample(profiler=None)               # 运行主循环（profiler 为样本/进度记录器）
CoordinatorAgent.set_global_sample_nums(n)      # 类方法：设置全局采样计数（断点恢复用）
```

**调用示例**（见 [runtime/pipeline.py](../runtime/pipeline.py) `_launch_samplers`）

```python
import threading
from drsr_420.agents import CoordinatorAgent

coordinator = CoordinatorAgent(
    database=database, evaluators=evals, samples_per_prompt=config.samples_per_prompt,
    config=config, max_sample_nums=max_sample_nums,
    llm_class=class_config.llm_class, prompt_ctx=prompt_ctx, llm_client=llm_client,
)
threading.Thread(target=coordinator.sample, kwargs={'profiler': profiler},
                 daemon=True, name=f"Sampler-{i}").start()
```

---

## 2. SamplerAgent —— 采样者

**文件**：[sampler_agent.py](./sampler_agent.py)
（层内部件：[prompt_injection.py](./prompt_injection.py) 提示词装配、
[skeleton.py](./skeleton.py) 骨架提取——两者不是 Agent，不参与组织图）

**职责**

- 把"指令 + 任务头 + 历史经验/残差注入"拼成最终提示词，交给 `ToolCallerAgent`
  与 LLM 多轮对话，拿到候选方程骨架。
- 骨架后处理：从 LLM 混合输出（文字 + 代码）中抽取可执行函数体
  （`skeleton.extract_body` / `extract_code_fragment`）；抽不到时自动重采样
  （最多 `skeleton.MAX_BODY_RETRIES = 3` 次），仍无效则丢弃该样本。
- 经验/残差注入规则（`PromptInjector`）：None（失败教训）始终注入；Good/Bad 按概率参与；
  超过新鲜度阈值后只注入近期经验；Good 按 score 降序、Bad 按 score 升序截断；
  失败经验额外附"参数预算"提示。
- 继承 `LLM` 抽象基类（`samples_per_prompt` 决定每批数量）。

**关键接口**

```python
SamplerAgent(samples_per_prompt: int, batch_inference: bool = True, trim: bool = True,
             prompt_ctx: pc.PromptContext | None = None, llm_client: LLMClient | None = None)
sampler.draw_samples(prompt: str, config: config_lib.Config)
# -> (samples: list[str], thinking_contents: list[str]) | None
```

**调用示例**

```python
from drsr_420.agents import SamplerAgent

sampler = SamplerAgent(samples_per_prompt=4, prompt_ctx=prompt_ctx, llm_client=llm_client)
samples, thinking = sampler.draw_samples(prompt.code, config)
```

---

## 3. ToolCallerAgent —— 工具调用者

**文件**：[tool_caller_agent.py](./tool_caller_agent.py)

**职责**

- 与 LLM 多轮对话：若模型返回 `tool_calls`，逐条执行工具并把结果回传给模型，
  直到模型给出最终答复。
- 工具执行默认走 `knowledge.tool_runner.mcp_call_tool`（MCP 服务器封装，可注入自定义执行器）。
- 轮次上限 `max_tool_rounds`（默认 4）：防止模型无限检索文献拖死采样。
- 兜底：`content` 为空时回退 `reasoning_content`，避免模型只思考不输出正文时骨架丢失。

**关键接口**

```python
ToolCallerAgent(llm_client, tool_executor=None, max_tool_rounds=4)
tool_caller.complete(content: str, repeat: int = 1)
# 恒返回 (responses_list, think_list)，长度均为 max(1, repeat)
# 契约必须恒定返回 list：旧实现在 repeat<=1 时返回标量，被 sampler 的 list(str)
# 逐字符炸开成海量伪样本（1 次请求 → 82 次 LLM 调用）。
```

**调用示例**

```python
from drsr_420.agents import ToolCallerAgent

tool_caller = ToolCallerAgent(llm_client, max_tool_rounds=4)
responses, thinking = tool_caller.complete(content, repeat=4)
```

---

## 4. EvaluatorAgent —— 评估者

**文件**：[evaluator_agent.py](./evaluator_agent.py)
**执行机制**：[evaluation/sandbox.py](../evaluation/sandbox.py)、[evaluation/problems.py](../evaluation/problems.py)

**职责**

- 编排：把骨架编译成可运行程序（`_sample_to_program`），在常驻子进程沙箱
  `LocalSandbox` 中执行，多起点 `least_squares` 拟合参数。
- 产出 `EvaluationOutcome(score, error, residual)` 供 CoordinatorAgent 分类
  Good/Bad/None；失败时 `score=None`。
- 成功样本注册进 `ExperienceBuffer`（`register_program`），失败样本经 Profiler 记录。
- 执行机制（进程池、超时重建、numba 降级）在 `evaluation/sandbox.py`，
  本文件只保留角色逻辑。

**关键接口**

```python
EvaluatorAgent(database: buffer.ExperienceBuffer, template: code_manipulation.Program,
               function_to_evolve: str, function_to_run: str, inputs: Sequence[Any],
               timeout_seconds: int = 30, sandbox_class: Type[Sandbox] | None = None)
evaluator.analyze(EvaluationRequest) -> EvaluationOutcome
evaluator.analyse(sample, island_id, version_generated, ...)   # deprecated：返回旧式三元组
evaluator.close()                                              # 释放沙箱 worker
```

**调用示例**（见 [runtime/pipeline.py](../runtime/pipeline.py)）

```python
from drsr_420.agents import EvaluatorAgent, EvaluationRequest
from drsr_420.evaluation.sandbox import LocalSandbox

evaluator = EvaluatorAgent(
    database=database, template=template, function_to_evolve=function_to_evolve,
    function_to_run=function_to_run, inputs=inputs,
    timeout_seconds=config.evaluate_timeout_seconds, sandbox_class=LocalSandbox)

outcome = evaluator.analyze(EvaluationRequest(
    sample=sample, island_id=0, version_generated=1,
    global_sample_nums=7, sample_time=1.2, profiler=profiler))
outcome.score, outcome.error, outcome.residual
```

---

## 5. ExperienceSummarizerAgent —— 经验总结者

**文件**：[experience_summarizer_agent.py](./experience_summarizer_agent.py)

**职责**

- 对一批样本及质量标签（Good/Bad/None）逐个构造分析提示并调用 LLM，把"得分高/低/出错"
  的样本转成结构化分析文本（改进建议）。
- 返回 `list[ExperienceEntry]`（每条自带样本/质量/错误/分析文本）；CoordinatorAgent
  补齐归属字段后写入 `experiences.json`，供下一轮采样注入。
- 有 `prompt_ctx` 时用动态模板渲染提问（变量名/因变量贴合任务），无则用默认模板。

**关键接口**

```python
ExperienceSummarizerAgent(llm_client, prompt_ctx: pc.PromptContext | None = None)
summarizer.analyze(samples, quality_for_sample, error_for_sample, prompt) -> list[ExperienceEntry]
# 三个平行列表长度不一致时抛 ValueError（旧实现靠 zip 静默错配样本与经验）
```

---

## 6. ResidualAnalyzerAgent —— 残差分析者

**文件**：[residual_analyzer_agent.py](./residual_analyzer_agent.py)

**职责**

- 对当前最优样本的残差矩阵做统计（均值 / 最大绝对值 / 标准差），读取上一次
  `residual_analyze.json` 作为上下文，拼进提示词让 LLM 输出结构化修正方向。
- 返回 `ResidualInsight`；归属字段（岛屿/样本序号/最佳分）由 CoordinatorAgent 补齐后
  追加写回 `residual_analyze.json`。

**关键接口**

```python
ResidualAnalyzerAgent(llm_client, prompt_ctx=None, results_root='.')
analyzer.analyze(sample: str, residual: np.ndarray) -> ResidualInsight
# residual: shape (N, n_feat + 1)，最后一列 = 预测残差
```

---

## 7. DataAnalyzerAgent —— 数据分析者

**文件**：[data_analyzer_agent.py](./data_analyzer_agent.py)

**职责**

- 实验开始时对目标数据集做初次分析：读 CSV 或数据字典（`{'data': {'inputs', 'outputs'}}`），
  随机采样（默认 100 行）+ 保留小数（默认 3 位），请 LLM 分析"自变量对因变量的影响
  关系 + 自变量间潜在关系"。
- 分析结果写入 `residual_analyze.json` 的 `sample_order=0` 初始记录，是后续残差分析 /
  经验注入链路的起点。
- 支持自定义提示（含 `{csv_data}` 占位符）；`verbose=True` 时打印提示词并落盘。

**关键接口**

```python
DataAnalyzerAgent(api_url='http://127.0.0.1:5000/completions', timeout=300,
                  decimal_places=None, sample_size=None, base_dir=None,
                  llm_client=None, seed=None)
analyzer.analyze(data_source, custom_prompt=None, max_rows=None, verbose=True) -> str
```

---

## 8. 线程安全与共享状态

概要（完整表见 [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) §5）：

- `_SAMPLER_LOCK`（可重入 RLock）保护全局采样计数与
  `experiences.json` / `residual_analyze.json` / `checkpoint.json` 的读-改-写；
- 每个 Sampler 线程**独享一份** `EvaluatorAgent` 列表（避免 `LocalSandbox._last_params`
  等实例状态竞态）；`ExperienceBuffer` 由所有线程共享（内部自带上锁）；
- LLM 客户端按任务克隆独立副本（`clone_llm_client`），采样/经验/残差互不影响；
- 包内 `__init__.py` 用 PEP 562 惰性导出，顶层不 import 子模块（避免循环导入）。
