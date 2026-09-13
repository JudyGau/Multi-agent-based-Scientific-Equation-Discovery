# 配置文件管理方案（阶段 8 设计稿）

> **状态：已全量落地（2026-09-13）。** 落地记录见文末 §9。
>
> 结论先行：**「哪个 Agent 用哪套配置」这件事，当前在代码里没有任何一处声明。**
> 它只能靠 grep `clone_for_task('...')` 与 `load_llm_config("...")` 反推；而现有结构
> （一个 `*.config` 同时装密钥、模型参数和角色覆盖，再用 CLI 字符串选文件）
> **在原理上无法表达"给某个角色换一个模型"**——这正是仓库里那两个孤儿配置文件
> 存在的原因，也是 `explain` 静默失效的原因。
>
> 本方案把配置拆成三层，让「角色 → 档案」的绑定成为**唯一一处、可入库、可校验**的声明。

---

## 1. 现状盘点（实测，非推测）

### 1.1 根目录平铺的 6 个文件

| 文件 | 大小 | `model` | 谁在读它 | 入库 |
|---|---|---|---|---|
| `glm_glm-5.3-flash.config` | 456 B | `glm/glm-5.3-flash` | ①`factory.load_llm_config` 的默认值 ②`read_paper._load_llm_config` 硬编码 ③`cli.main --llm_config` 默认值 ④`example.sh` / `MRFCompress-3.sh` | ✗ 被 `*.config` 忽略 |
| `llm.config` | 199 B | `glm/glm-5.3-flash` | 4 个 IDE 运行配置 + 本地 `workspace.xml` 的 `MRFCompress-3`，均以 `--llm_config llm.config` | ✗ |
| `llm.config.example` | 172 B | 同上 | **无人读**，纯模板 | ✓ **唯一入库的模板** |
| `llm_explain.config` | 231 B | `deepseek/deepseek-v4-pro` | **无人读**（全仓库 0 处引用） | ✗ |
| `llm_summary.config` | 315 B | `ollama/Qwen3.8-27B-UD-IQ4_XS` | **无人读**（全仓库 0 处引用） | ✗ |
| `rag.config` | 386 B | —（嵌入模型，另一套 schema） | `rag_kb._CONFIG_PATH` | ✗ |

入库的模板只有**一个**，而且是 **legacy 名**（`llm.config.example`）的，里面还没有
`tasks` 段；规范名 `glm_glm-5.3-flash.config`、`rag.config` **都没有模板**——
新克隆的仓库拿不到任何能用的配置起点。

### 1.2 代码里硬编码的配置文件名（3 处）

| 位置 | 文件名 | 状态 |
|---|---|---|
| `drsr_420/llm/factory.py:23` | `glm_glm-5.3-flash.config` | 默认值 |
| `drsr_420/analysis/explain.py:166` | `deepseek_deepseek-v4-flash.config` | **文件不存在** |
| `drsr_420/knowledge/tools/read_paper.py:26` | `glm_glm-5.3-flash.config` | 可用，但绕过了 `--llm_config` |

### 1.3 一个正在发生的故障：`explain` 静默产出空文件

```
$ python -c "import drsr_420.llm as llm; llm.load_llm_config('deepseek_deepseek-v4-flash.config')"
FileNotFoundError: [Errno 2] No such file or directory: 'deepseek_deepseek-v4-flash.config'
```

`explain_best_sample()` 把它包在 `try/except Exception` 里 → `client = None` →
`explain_re_act(None, content)` 第 31 行 `if client is None: return None` → 写空
`explain.txt`，全流程只有一行 `[WARN]`。

而仓库里躺着的 `llm_explain.config` 恰好就是为这个角色准备的（`deepseek/deepseek-v4-pro`），
`llm_summary.config` 恰好是为 `read_paper` 摘要准备的（本地 ollama）——**有人按正确的直觉
建了文件，但没有任何代码路径能加载它们**。这不是疏忽，是结构缺陷的必然产物。

### 1.4 命名与选择方式的不一致

| 引用点 | 用的名字 |
|---|---|
| 5 个 IDE 运行配置（4 个入库 + `MRFCompress-3` 本地） | `llm.config`（legacy） |
| `example.sh`、`MRFCompress-3.sh` | `glm_glm-5.3-flash.config`（规范） |
| 库代码默认值 | `glm_glm-5.3-flash.config` |

---

## 2. 问题归类

1. **三种正交数据塞进同一个 JSON**：密钥/端点、模型与生成参数、角色行为覆盖。
   三者的寿命、保密等级、review 需求完全不同。
2. **「角色 → 模型」无处声明**：只能靠 grep。
3. **档案选择靠 CLI 字符串 + 硬编码文件名**：`--llm_config` 管不到 `explain` 与
   `read_paper`，实验快照里记录的 LLM 配置与实际调用 explain 的模型可以完全不同。
4. **子进程拿不到父进程的选择**：`read_paper` 跑在 MCP server 子进程里，
   `mcp` SDK 只透传系统级白名单环境变量，它只能自己再从磁盘读一个固定文件名。
5. **入库边界反了**：唯一入库的模板是 legacy 名，规范名无模板；`rag.config` 无模板。

---

## 3. 第一性原理：一份"配置"其实在回答三个问题

| # | 问题 | 寿命 | 保密 | 现状 | 应否入库 |
|---|---|---|---|---|---|
| Q1 | **连接谁？用哪把钥匙？** | 随部署变，密钥随时轮换 | 高 | 与 Q2 同文件 | ✗ |
| Q2 | **生成参数是什么？** | 随模型变，需要可复现 | 低 | 与 Q1 同文件 | 可（但与 Q1 同文件时只能不入库） |
| Q3 | **哪个角色用哪套？覆盖什么？** | 随系统设计变，需要 code review | 无 | 藏在 Q1/Q2 文件的 `tasks` 字段里 | **应当入库** |

现有方案把 Q1+Q2 放进 `*.config`，把 Q3 放进**同一个文件的 `tasks` 字段**，然后用
CLI 字符串选那个文件。于是 Q3 的语义被钉死在"同一个模型的参数覆盖"上：
`clone_for_task('explain')` 只能改 `reasoning_effort`，**换不了模型**。

Q3 需要的是"角色 → 档案"的**多对一映射**，而现状提供的是一份"档案自带一列角色参数"的
**一对一内嵌**——表达能力差了一个层级。补上这一层，1.3 的故障与两个孤儿文件同时消失。

---

## 4. 方案：三层分离，**扩展名即保密边界**

```
config/
├── agents.config.json              # ★ 入库：唯一回答 Q3 的地方（无密钥）
├── glm_glm-5.3-flash.config        #   Q1+Q2，不入库（*.config 规则）
├── glm_glm-5.3-flash.config.example    # 入库：模板（api_key 留空）
├── deepseek_deepseek-v4-pro.config     # 不入库
├── deepseek_deepseek-v4-pro.config.example
├── ollama_Qwen3.8-27B.config       # 不入库
├── ollama_Qwen3.8-27B.config.example
└── rag.config                      # RAG，独立 schema，不入库
```

命名规则（**保持你提的 `提供商_模型.config`**）：目录统一到 `config/`，
文件名一律 `提供商_模型.config`；模型名里的 `/` 与 `:` 换成 `-`。

> **重要约束：代码不得解析文件名。** 文件名的唯一权威是文件内的 `model` 字段
> （`Normalize` 阶段已校验 `provider/model` 格式），文件名只是给人看的标签。
> 这样文件名写错不会导致静默连错模型。

### 4.1 `config/agents.config.json`（入库，无密钥）

```json
{
  "version": 1,
  "default": "glm_glm-5.3-flash",
  "roles": {
    "sampling":   {"params": {"reasoning_effort": "low"}},
    "analysis":   {"params": {"reasoning_effort": "high"}},
    "experience": {"params": {"reasoning_effort": "high"}},
    "residual":   {"params": {"reasoning_effort": "high"}},
    "explain":    {"config": "deepseek_deepseek-v4-pro", "params": {"temperature": 0.1, "top_p": 1.0}},
    "summary":    {"config": "ollama_Qwen3.8-27B",       "params": {"temperature": 0.4}}
    }
  }
}
```

* `default`：未显式指定 `config` 的角色用它 → 这就是今天"一个档案打天下"的行为，
  **零行为变化**；
* `roles.<role>.config`：换模型（今天做不到）；
* `roles.<role>.params`：参数覆盖 → 取代档案文件里的 `tasks` 字段（语义不变）。

### 4.2 角色词汇表 = 代码里的常量（code as truth）

现有 6 个 task 名已经是事实上的枚举，只是从未被显式声明。新增
`drsr_420/llm/roles.py`，把角色表与解析逻辑放在 **llm 层**（它依赖 `core`，被
`agents`/`analysis`/`knowledge` 依赖，符合现有分层）：

```python
TASKS = ("sampling", "analysis", "experience", "residual", "explain", "summary")

@dataclass(frozen=True)
class RoleResolution:
    role: str
    config_path: str          # 解析后的绝对路径
    params: dict              # 覆盖后的参数
    source: str               # 生效来源（CLI / env / registry.roles / registry.default）

def load_registry(path: str | None = None) -> RoleRegistry: ...
def resolve(role: str, *, cli_overrides=None) -> RoleResolution: ...
def describe_roles() -> str: ...      # 渲染角色 → 档案表
```

配套自检入口（与 `python -m drsr_420.agents` 的组织图同一风格）：

```bash
python -m drsr_420.llm.roles            # 打印「角色 → 模型档案（生效来源）」表
python -m drsr_420.llm.roles --check    # 校验：角色齐全、档案可解析、密钥可达
```

`--check` 的校验项：每个 `TASKS` 角色都有解析结果；每个被引用的档案文件存在
（或提示用户 `cp *.config.example`）；`model` 字段格式合法；api_key 或对应环境变量
至少有一个非空。

### 4.3 解析优先级（显式、可打印）

```
--role-config sampling=xxx.config      (最高，可重复，按角色精确覆盖；'*' 表示所有角色)
  > 环境变量 DRSR_ROLE_CONFIG_<ROLE>  （容器/CI 友好）
  > agents.config.json 的 roles.<role>.config
  > --llm_config                       （CLI 指定的**默认**档案）
  > agents.config.json 的 default
  > 内置常量 DEFAULT_PROFILE           (最低，保证永不"无配置")
```

`--llm_config` **保留**，语义收窄为"改默认档案"——现有 4 个 IDE 配置与 2 个 `.sh`
因此继续可用（只需把 `llm.config` 改成规范名）。每个角色的生效来源都会写进
`config_snapshot.json`，让"这次实验的 explain 到底用了哪个模型"**可查**。

### 4.4 子进程传递（MCP server）

`tool_runner._server_env()` 已经有一个环境变量白名单（用于透传 API key），
在同一个地方加一个：

```python
env["DRSR_ROLE_CONFIG_SUMMARY"] = <resolver 解析出的绝对路径>
```

`read_paper._load_llm_config()` 改为：env → registry → 内置常量。
这样 MCP 子进程用的是**父进程解析出的那份配置**，而不是它自己从磁盘猜的文件名。

### 4.5 与主流多 Agent 方案的对照

| 框架 | 「角色 → 模型」写在哪 | 凭据来源 |
|---|---|---|
| CrewAI | `agents.yaml`（角色定义）+ `llms` 表 | 环境变量 / `LLM(...)` |
| AutoGen | `config_list` + 每个 Agent 的 `llm_config` | 环境变量（OAI_CONFIG_LIST） |
| LangGraph | 每个节点的 model 对象，由图工厂集中构造 | 环境变量 |
| **本方案** | `config/agents.config.json`（入库）+ `--role-config` 覆盖 | 档案文件（兼容现状）或环境变量 |

共同点：**角色与模型的绑定集中在一处、且这一处不含密钥**。本方案对齐这一点，
但保留档案文件内的 `api_key`（本地 ollama 需要占位 key，且现状依赖它），
无密钥的 `agents.config.json` 入库换取可 review、可校验。

---

## 5. 迁移步骤（阶段 8）

| 步 | 内容 | 风险 |
|---|---|---|
| 8.1 | 新增 `drsr_420/llm/roles.py` + `config/agents.config.json`；`--llm_config` 语义收窄为"默认档案"；新增 `--role-config`；每个角色的解析结果进 `config_snapshot.json` | 低（默认行为不变） |
| 8.2 | 修掉两处硬编码：`explain.py` 走 resolver（**顺带修复 1.3 的空 `explain.txt` 故障**）、`read_paper.py` 走 env→registry→常量 | 低 |
| 8.3 | 配置搬家到 `config/`：把 `llm.config` 里的真实密钥并入 `config/glm_glm-5.3-flash.config`；删除 `llm.config`、`llm_explain.config`、`llm_summary.config`（后两者本就无人读）；补齐 **3 个** `.example` 模板（glm / deepseek-v4-pro / **rag.config**，今天完全没有） | **中**：动了含密钥的本地文件，需先备份并逐项确认 |
| 8.4 | 同步引用点：4 个入库 IDE 运行配置 + 本地 `workspace.xml` 的 `MRFCompress-3` + `example.sh` + `MRFCompress-3.sh` + README / ARCHITECTURE / data-README / `drsr_420/agents/README.md` | 低 |
| 8.5 | 护栏 `tests/test_config_roles.py`（见 §6） | 低 |
| 8.6 | 档案文件里的 `tasks` 字段**删除**（迁入 registry）；读到旧字段时打印一次迁移提示 | 低 |

`rag.config` 一并搬进 `config/`（同一目录、同一 `*.config` 忽略规则），schema 不动。

---

## 6. 验收与护栏

| 检查 | 说明 |
|---|---|
| **代码里不得出现配置文件名** | AST/字符串扫描：`drsr_420/` 下除 `roles.py` 的常量与 `*.example` 提示语外，不得再出现 `*.config` 字面量 —— **这条直接锁死 1.2/1.3 那类故障** |
| 角色表完备 | 每个 `AgentSpec.llm_task` 都在 `TASKS` 里；每个 `TASKS` 角色都有解析结果 |
| 解析优先级 | 5 层优先级逐层构造场景断言（含"registry 不写 config → 落到 default → 行为与今天一致"） |
| 子进程传递 | `_server_env()` 必须带上解析出的档案绝对路径；`read_paper` 优先读 env |
| 快照可追溯 | `config_snapshot.json` 必须记录每个角色的生效档案与来源 |
| 模板完整性 | 每个 `X.config` 都要有入库的 `X.config.example`（防"新克隆无模板"复发） |
| 回归 | 375 项既有测试全绿（默认档案行为不变） |

---

## 7. 明确不做

* **不引入配置框架**（hydra / dynaconf / pydantic-settings）：项目刻意保持零重依赖，
  一个 30 行的 resolver 足够；引入框架会让"配置从哪来"更难回答，而不是更容易。
* **不改 `core/config.py` 的实验超参**：那是"实验怎么跑"（迭代数、岛屿、采样数），
  与"用哪个模型"正交，dataclass 已经合适。
* **不做 `${ENV}` 插值**：现有 `api_key` 留空即回退环境变量的机制已够用；
  插值会让 `config_snapshot.json` 里出现未解析的占位符。
* **不让代码解析文件名**（见 §4 的重要约束）。
* **不动 `specs/` 与实验产物目录**。

---

## 8. 附：本方案顺手能修掉的具体问题

1. `explain` 静默写空 `explain.txt`（真实故障，1.3）；
2. 「`--llm_config` 管不到 explain / read_paper」的语义漏洞；
3. `llm.config` legacy 名与规范名并存；
4. 两个孤儿配置文件（它们想表达的意图，新结构里终于能表达）；
5. 唯一入库模板是 legacy 名、`rag.config` 无模板；
6. `.gitignore` 里同时写着 `llm.config` 与 `*.config`（后者已覆盖前者，属冗余）。

> 另注：仓库声明 `license = "Apache-2.0"` 且 `runtime/pipeline.py` 等文件带
> DeepMind 的 Apache-2.0 头，但**根目录没有 LICENSE 文件**（`git ls-files` 无此项）。
> 与本方案无关，单独提一句以免遗漏。

---

## 9. 落地记录（阶段 8 执行完毕）

上面是设计稿原文（保留判断过程），下面是实际做成了什么样、与设计的差异。

### 9.1 与设计稿的差异（3 处，均有理由）

| 设计稿 | 实际落地 | 理由 |
|---|---|---|
| 注册表里给 `explain` / `summary` 绑定各自档案 | **默认只绑定 `params`，全部角色共用 `default`**；绑定写法写在 `_comment` 里 | 跟踪文件若引用 gitignored 的档案，**每个新克隆**的 `--check` 都会失败。默认必须"处处能跑通"，把开关写清楚比替用户预设更合适。用户本机已有这两份档案，加两行即可启用 |
| 新建 `drsr_420/llm/config.py` | 拆成 `roles.py`（声明+解析）/ `role_clients.py`（客户端池）/ `role_diagnostics.py`（渲染+自检） | 首版单文件写到 **639 行**，超过项目"最长文件 < 500 行"的既定指标；按"解析规则 / 怎么用 / 怎么看"三种变化原因切开，最长回到 482 |
| 只修 `explain.py` / `read_paper.py` 的硬编码 | 另把 `coordinator_agent` 里写死的 `temperature=0.0` / `0.4` 与 `read_paper` 里的 `top_p=0.9` 等**全部搬进注册表** | 它们与"角色参数"是同一件事；留在代码里就还是三处硬编码，护栏也守不住 |
| §4.3 把 `registry.default` 排在 `--llm_config` 之前 | 实现里**调换**了这两级 | 注册表恒有 `default`，若它优先则 `--llm_config` 永远失效。显式 CLI 参数应当高于文件默认值；已同步订正 §4.3 |

### 9.2 实际改了什么

* **新增** `config/agents.config.json`（入库、无密钥）—— 6 个角色的档案绑定与参数覆盖；
* **新增** `drsr_420/llm/roles.py` / `role_clients.py` / `role_diagnostics.py`；
* **新增** `drsr_420/cli/llm_setup.py`（`cli/main.py` 因此增长到 509 行，按 500 行预算切出）；
* **新增 CLI**：`--role-config ROLE=FILE`（可重复，`'*'` 表示所有角色）；
* **搬迁**：4 份根目录档案 → `config/`（`llm.config` 的真实密钥并入 `config/glm_glm-5.3-flash.config`）；
* **删除**：`llm.config`、`llm.config.example`、`llm_explain.config`、`llm_summary.config`（后两者本就无人读取）；
* **补齐模板**：`glm_glm-5.3-flash` / `deepseek_deepseek-v4-pro` / `ollama_Qwen3.8-27B-UD-IQ4_XS` / `rag` 共 4 份 `.config.example`（此前**只有 legacy 名的 `llm.config.example`**，规范名与 `rag.config` 都没有模板）；
* **修掉硬编码**：`analysis/explain.py`、`knowledge/tools/read_paper.py`、`llm/factory.py` 默认值、`knowledge/rag_kb.py` 的 `_CONFIG_PATH`；
* **新护栏** `tests/test_config_roles.py`（43 项）：不得硬编码档案名、6 级优先级逐级验证、参数合并且默认值与历史逐位相同、kwargs 隔离、`single(None)` 不建连接、子进程环境传递、explain 走角色解析、模板完整性、CLI 无 runpy 警告。

### 9.3 顺带修掉的真实故障

**`explain` 长期静默产出空的 `explain.txt`。** `analysis/explain.py:166` 硬编码加载
`deepseek_deepseek-v4-flash.config`，该文件在仓库中**不存在** → `FileNotFoundError`
被 `try/except Exception` 吞掉 → `client=None` → `explain_re_act` 第 31 行直接
`return None` → 写空文件，全程只有一行 `[WARN]`。现在 explain 经角色解析取客户端，
并新增 `ExplainRoleWiringTest` 三条回归锁死。

另外：`cli.main` 原先在 `--llm_config` 指向的文件不存在时会**静默写出一份占位配置**
（哪怕路径叫 `no_such.config`），随后必然因 api_key 为空而失败——已改为明确报错并列出
可用的档案与模板（含可直接复制的 `cp` 命令）。

### 9.4 指标

| 指标 | 阶段 7 结束时 | 现在 |
|---|---|---|
| 测试数 | 375 | **418** |
| 全局最长文件 | 482（`llm/client.py`） | 482（未变；首版 `roles.py` 639 行已拆） |
| 根目录 `.config` 文件 | 6 | **0**（全部进 `config/`） |
| 入库的配置模板 | 1（legacy 名） | **4**（规范名 + rag） |
| 库代码里的硬编码档案名 | 3 处 | **0**（有护栏） |
| 无人读取的孤儿档案 | 2 | **0** |
