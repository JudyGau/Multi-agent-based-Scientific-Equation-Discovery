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

## LLM 配置（llm.config）

根目录提供 `llm_summary.config`（JSON），用于配置大模型访问与采样参数：

```json
{
  "host": "api.deepseek.com",
  "api_key": "xxx",
  "model": "deepseek/deepseek-v4-pro",
  "max_tokens": 65536,
  "temperature": 0.1,
  "top_p": 1.0,
  "frequency_penalty": 0.0
}
```

- `api_key` 请替换为真实密钥，否则会报"未提供令牌"。
- `model` 使用 `provider/model` 形式。支持提供商：`deepseek`、`siliconflow`、`deepinfra`、`ollama`、`blt`（柏拉图）、`cstcloud`（科技云）、`glm`（智谱）。
- 切换模型直接修改 `llm_summary.config` 相应字段即可；`api_key` 留空时回退读取对应环境变量（如 `DEEPSEEK_API_KEY`、`ZHIPU_API_KEY`、`SILICONFLOW_API_KEY`）。
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

程序运行时会经 MCP 拉起工具服务器并复用；`read_paper` 的总结模型由 `llm_summary.config` 配置。

## 结果产物与目录结构

以 `experiments/oscillator1_20250101-120000/` 为例：

- `run.out` / `run.err`：标准输出/错误输出
- `spec_dynamic.txt`：本次运行的动态 spec（便于复现）
- `experiences.json`：采样过程中的经验/总结
- `residual_analyze.json`：残差分析结果
- `explain.txt`：最终公式的力学解释（若启用）
- `samples/`：每次评分的样本 JSON（`samples_N.json`），含 `score`、`function`、`params`

## 仓库结构

```
main.py                       # 入口（CSV 动态模式）
llm.py                        # LLM 客户端与 ClientFactory（多提供商 + 参数统一注入）
llm.config / llm_summary.config / llm_explain.config / rag.config   # 配置文件（不入库）
example.sh                    # 批量运行示例
drsr_420/
  pipeline.py                 # 调度 Evaluator/Sampler，注入 LLM Client
  sampler.py                  # 采样器（LLM 请求 + 工具调用循环）
  evaluator.py                # 运行候选方程、BFGS 拟合与打分
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
