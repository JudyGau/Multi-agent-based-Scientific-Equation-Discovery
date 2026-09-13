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

---

## 10. 追加（阶段 9）：自定义提供商，并把两个角色绑定真正启用

阶段 8 落地时把 `explain` / `summary` 的档案绑定**留成注释里的开关**（§9.1 第一条），
理由是"跟踪文件引用 gitignored 的档案，新克隆的 `--check` 会失败"。本阶段按用户要求
把这个开关打开，需求是：`explain` 用官方 DeepSeek 端点，`summary` 用**自定义提供商**
（校内网关 `https://api.llm.ustc.edu.cn/v1`）——并把"自定义提供商"做成能力而不是
一次性硬编码。

> 阅读提示：§1–§9 是阶段 8 的设计稿与落地记录，其中出现的 `deepseek_deepseek-v4-pro`
> 是**当时**的目标档案名；阶段 9 起它已下线，`explain` 改用 `deepseek_deepseek-v4-flash`
> （官方端点）。历史段落保留原样，以免抹掉判断过程。

### 10.1 为什么"自定义提供商"是个能力问题，不是加一行配置

`_PROVIDER_SPECS` 原先是**封闭表**：provider 段不在其中就抛"不支持的提供商"。
于是"给某个角色换一个自建/校内/第三方网关"只能改代码。而角色注册表里那行
`roles.<role>.config` 存在的意义恰恰是**不改代码就能换**——表达力缺口正好卡在
"档案能不能自带端点"这一步。

按 §4 的三层分工，答案其实早就有了：**端点属于 Q1（连接谁 / 用哪把钥匙），本就归档案**。
所以自定义提供商的实现只有一句话——**未知 provider 段 + 档案给了 `base_url` ⇒ 照常构造**。
代码表只留给"内置提供商的经验默认值"，不再兼作白名单。

同时暴露出第二个缺口：**请求体方言**原先隐含地由 provider 名决定（`_adapt_payload`
里的 if/elif）。自定义提供商的 provider 段（`ustc`）不是方言名，必须能显式声明。
于是把它抽成 `llm/adapt.py` 的 `DIALECTS` + 档案字段 `dialect`（缺省 `openai`：
不认识的一律不发，避免 400）。

### 10.2 设计决定

| 决定 | 理由 |
|---|---|
| 未知 provider 段 + **无** `base_url` ⇒ 报错，且报错里教人怎么加 | 沉默会退化成"连到猜出来的地址"；报错信息里的内置列表由 `_PROVIDER_SPECS` **生成**而非手写（手写必漂移） |
| `dialect` 拼错 ⇒ **报错**，不静默降级成 `openai` | 与"角色名拼错必须报错"同一条原则：静默降级 = "配了却没生效" |
| 密钥解析固定为 `api_key` > `api_key_env` > 按 provider 段派生（`ustc` → `USTC_API_KEY`） | 自定义提供商没有内置表可查，派生规则让部署方**有确定的名字可设**；报错里出现的永远是具体变量名，绝不会是 `None` |
| 自定义提供商默认 `api_key_required: true` | 校内网关要密钥；本地免鉴权服务显式写 `false`——不靠"localhost 就免密钥"这类魔法 |
| `extra_body` 改为**最后并入**请求体 | 它原本是**死字段**：工厂不注入、适配层又 pop 掉，`config/ollama_*.config.example` 里那份 `enable_thinking` 从未上线。它的语义（沿用 openai SDK）就是"把键并入请求体"，而它正是自定义端点接私有能力（vLLM 的 `chat_template_kwargs`）的唯一出口；放在方言适配**之后**并入，方言规则才删不掉调用方的显式声明 |
| 模板从"存在即可"升级为"**能构造出客户端**" | 绑定一旦变具体，"模板可用"就成了新克隆能不能起步的前提。这条同时守住 `model` 拼写、自定义提供商漏写 `base_url`、`dialect` 拼错三类漂移 |

### 10.3 新克隆的开销（阶段 8 的顾虑如何被消化）

绑定打开后，新克隆的 `--check` 会报"缺档案"——这是**预期**的，且是可操作的：
报错直接给出 `cp <模板> <档案>`；模板齐备由
`test_registry_bound_profiles_have_shipped_templates` 守着（注册表引用谁，就得有谁的模板）。
零配置可跑的性质由 `default`（`glm_glm-5.3-flash`）保留：删掉某角色的 `config` 行即回退。

### 10.4 顺带修掉的两处

* **`--check` 的问题描述不再一律含 `cp`**：问题分两类、修复动作不同——档案不存在才给
  `cp`，密钥不可达给的是"填哪个字段 / 设哪个环境变量"。给后者塞 `cp` 是误导（文件明明在）。
  测试相应从"每条都含 `cp `"改成"含 `cp ` 或指出环境变量"，并**新增**一条断言缺密钥时
  报出具体变量名。
* **角色表列宽按内容算**：`deepseek_deepseek-v4-flash`（26 字符）正好顶满旧的 26 宽列，
  来源列被挤成 `...flashregistry:roles.explain.config` 连字。表头/表体列宽改为
  由数据算出，档案名再长也不会连字（有回归测试）。

### 10.6 端点键名与写法统一为完整 base_url（追加）

同一件事曾有三个键名：LLM 档案里 `host` 与 `base_url` 并存（`base_url` 优先），
RAG 侧是 `api_host`。现在统一为 `base_url` / `api_base_url`，**旧拼写一律报错**：

* `normalize_llm_config`：出现 `host` → `ValueError`，错误信息里带"改名为 base_url"；
* `rag_config.load_config`：出现 `api_host` → `ValueError`（`_RENAMED_KEYS` 表驱动）。

**为什么是报错而不是静默兼容？** 因为静默忽略旧键的后果不是"少个功能"，而是
**连到别的地方去**：内置提供商自带默认端点，档案里的 `host` 被忽略后请求会悄悄打到
默认地址；RAG 侧则更隐蔽——`api_base_url` 留空使 URL 变成 `/embeddings`，报一个与
"键名写错"毫无关系的 `MissingSchema`。既然改名成本只有一行，就让它在启动瞬间响亮地
失败，并把改法写在错误里。

第二层是**写法**。`deepseek_deepseek-v4-flash.config` 一直写的是裸主机
`api.deepseek.com`（它当初是 `host: api.deepseek.com`，改名那轮只动了键名、刻意没动
取值，靠"缺 scheme 自动补 `https://`"跑通），于是同一批档案里出现了两种形状。现在：

* **取值必须是完整 URL**（`http(s)://` 开头）：`require_absolute_url`（`llm/client.py`）
  在**客户端构造处**守一次——配置、内置默认值、环境变量兜底（`ZHIPU_API_BASE` /
  `BLT_API_BASE`）三条通道全部覆盖；
* **不再替用户补 scheme**。这段"宽容"正是两种写法长期并存的成因：`api.deepseek.com`
  既没有 scheme 也没有路径，从配置上看不出会打到哪个端点；
* 空串仍是"用内置默认端点"（老写法，保留）；`backend="api"` 的 RAG 端点缺失或裸主机
  同样报错（`rag_config._validate_endpoint`）；
* 内置默认端点与随仓库分发的档案一并统一成带路径的形状，`deepseek` 的 spec 默认值改为
  `https://api.deepseek.com/v1`（官方两种都收，项目内只保留一种）。

护栏：`test_config_roles.py` 扫描随仓库分发的模板，禁止两个旧键名（只查键名，不做子串
匹配——`base_url` 自己就含 "url"），并要求**模板里的端点带 scheme 且带路径**；运行时只
强制"带 scheme"（根路径挂载的自建网关是合法的），分发出去的文件才要求 `/v1` 这种形状。
`test_llm_custom_provider.BaseUrlNamingTest` 覆盖裸主机被拒、空串回退、环境变量通道；
`test_mcp_and_rag_cli.RagConfigNamingTest` 覆盖 RAG 侧键名迁移与端点校验。

### 10.7 可用性：网关错误、`--ping`、按需构造（追加）

起因是逐份档案**实测**（真实请求，而不是只看配置）：读到了两个真问题。

**① 端点写错时，最有用的那句话进不了异常。** 智谱对错误路径返回 **HTTP 200** +
`{"code":500,"msg":"404 NOT_FOUND","success":false}`；旧实现只因"响应里没有 choices"而抛
`API response missing choices`，网关那句原话只留在一行 `print` 里——排查时会先去怀疑模型名
或密钥。现在 `client.gateway_error_detail` 兼容 `{code,msg,success}` / `{message}` /
`{detail}` / `{error:{...}}` 四种信封，把原话带进异常；**取不到就不编造**（保持原样，
不在异常里塞空括号）。

**② 一份用不到的档案会拦住整个实验。** `cli.llm_setup` 原本一启动就构造所有档案并
`SystemExit`；`summary` 绑定的档案缺密钥时，一次完全不读文献的实验连启动都起不来。
现在 `RoleClients` **按需构造**：`from_registry` 只解析，首次 `get(role)` 才建该角色所在
档案的客户端并按档案缓存。解析期错误仍然立即致命；单份档案不可用只在真正用到时报错，
信息含角色名与档案路径。CLI 启动改为跑一遍离线自检并逐条 `[WARN]`——"错误推迟"不等于
"看不见"，严格的闸门仍是 `--check`（退出码 1）。

**③ 两级自检。** `--check` 离线（档案在不在、model 合法、密钥可达），`--ping` 联网
（每份档案一次真实请求）。`--ping` 的设计：按**档案**去重（6 个角色常共用一份档案，发 6 次
纯属浪费）、用该档案**首个角色**的参数与方言（顺带验证 requests 体与方言）、`max_retries=0`
+ 明确超时（探测要的是"现在通不通"，指数退避只会让一条命令卡几分钟）、输出上限压到 16
token。诊断文本经 `_clip` 强制 GBK 可编码：诊断工具不该因为"要诊断的东西"里有个 emoji 而
在 Windows 控制台上崩掉。

顺带修掉：`locate_config` 找不到档案时会把 `config/x.config` 拼成 `config/config/x.config`，
报错里的路径本身就是错的（会先把排查带偏）。

另外，加完端点校验后 `knowledge/rag_kb.py` 涨到 **517 行**，破了 500 行预算，于是把档案
部分（`DEFAULT_CONFIG` / 定位 / 读取 / 校验）拆成 `knowledge/rag_config.py` 并在 `rag_kb`
转发，历史导入路径（`rag_kb.load_config` / `rag_kb.DEFAULT_CONFIG` / `rag_kb._REPO_ROOT`）
全部照旧可用；这也与 LLM 侧（配置在 factory、运行在 client）形成对称。

### 10.8 指标

| 指标 | 阶段 8 结束时 | 现在 |
|---|---|---|
| 测试数 | 418 | **499**（新增 12 条 `tests/test_batch_scripts.py`：XML / `.sh` / `.bat` 三者逐项等价） |
| 全局最长文件 | 482（`llm/client.py`） | **487**（`llm/client.py`；`rag_kb.py` 一度 517，已拆出 `rag_config.py`） |
| `llm/` 层 | 9 文件 / 1818 行 | 11 文件 / 2331 行（`adapt.py` / `stream.py` 拆出） |
| `knowledge/` 层 | 8 文件 / 1359 行 | 9 文件 / 1435 行（拆出 `rag_config.py`） |
| 入库的配置模板 | 4 | **5**（新增 USTC 自定义提供商；`deepseek-v4-pro` 按用户要求下线、换成 `deepseek-v4-flash`） |
| 建档无需改代码即可接入的端点 | 7（内置） | **任意 OpenAI 兼容端点** |
| 端点的键名/写法 | `host` / `base_url` / `api_host`，且允许裸主机 | **`base_url` / `api_base_url`，必须完整 URL**（旧拼写报错） |
| 一份坏档案的后果 | 整个实验起不来 | **只影响用到它的角色**（启动告警） |
