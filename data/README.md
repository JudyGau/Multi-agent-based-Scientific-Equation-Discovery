# data/ —— 基准数据集

本目录存放符号回归（SR）的**社区标准基准数据集**与课题组自有的**磁流变（MRF）**数据。**每个子目录对应一个"问题"，至少包含 `train.csv`。**

非 MRF 数据并非本项目独有，而是沿用学界通用 benchmark（可按需增删或替换）；其来源见下方「数据来源」小节。

## 使用约定

- CSV 必须带表头：**前 n-1 列为自变量（特征），最后一列为因变量**。
- `python -m drsr_420.cli.main --data_csv <路径>` 读取该 CSV；列名会作为提示词中的自变量/因变量名传给模型。
- `--problem_name` 仅用于结果目录命名（`experiments/{problem_name}/{problem_name}_{timestamp}/`），
  匿名实验中不会把该名字暴露给模型（见 `prompt_config.PromptContext.problem`）。

## 文件命名（各数据集不完全一致，历史原因）

| 文件名 | 含义 |
|--------|------|
| `train.csv` | 训练集（**必需**，采样/打分/样本选择唯一读取的文件） |
| `test.csv` | 同分布测试集（`--test_csv`，缺省自动探测；**只用于收尾报告的样本外验证**） |
| `test_id.csv` | 同分布（in-distribution）测试集 |
| `test_ood.csv` / `ood_test.csv` | 分布外（out-of-distribution）测试集，两种拼写并存 |
| `train_noise.csv` | 加噪训练集（仅 oscillator2） |

> 采样、参数拟合、打分与最佳样本选择**只用 `train.csv`**；`test.csv` 由收尾分析读取
> （`--test_csv` 显式指定，缺省自动探测训练数据同目录的同名文件，`none` 关闭），
> 只把样本外指标写进 `run.out` 与 `report.md`，不参与任何选择。
> 注意 `MRFShear-3` 与 `MRFCompress-3` 的 `test.csv` 与 `train.csv` **曾经**逐行相同
> （现已互异），但它们与 MRF 其它系的 `test.csv` **都只有 2 行**：它们不是统计意义上
> 的 held-out 集——工具会标出"与训练集重合"，而即便不重合，2 个点也算不出可信的
> 泛化指标，不要拿它论证泛化能力。

## 数据来源

非 MRF 数据均取自学界公开 benchmark，**非本项目独有**。逐类来源如下：

| 子目录 | 类别 | 来源 | 原始文献 |
|--------|------|------|----------|
| `BPG0` / `CRK0` / `PO0` / `MatSci0` | LSR-Synth 四域 | **LLM-SRBench** | Shojaee et al., *LLM-SRBench*, arXiv:2504.10415 |
| `I.37.4_0_1` / `I.48.2_1_0` / `II.6.15b_3_0` / `III.4.33_3_0` | LSR-Transform（Feynman 方程变换） | **LLM-SRBench**（目标函数源自 **AI Feynman**） | Udrescu & Tegmark, *Science Advances*, 2020；Shojaee et al., arXiv:2504.10415 |
| `oscillator1` / `oscillator2` / `stressstrain` / `bactgrow` | 非线性振子 / 铝材应力-应变 / 细菌生长 | **LLM-SR** | Shojaee et al., *LLM-SR*, arXiv:2404.18400 |
| `MRFCompress-*` / `MRFShear-*` | 磁流变本构（小样本） | 课题组自有数据 | — |

> 说明：上述非 MRF 数据亦被 DrSR（arXiv:2506.04282）等后续工作复用，但其原始来源是 LLM-SR / LLM-SRBench / AI Feynman，并非任何单一后续工作独有。本仓库收录的题目范围甚至**大于** DrSR 实验所用子集（其 LSR-Transform 仅取 `I.37.4_0_1` 与 `III.4.33_3_0` 两题）。

## 数据集清单

行数为 `train.csv` 的**数据行数**（不含表头）。"最后列"为因变量。

### 物理 / 力学

| 数据集 | 行数 | 列（自变量… → 因变量） | 说明 |
|--------|------|------------------------|------|
| `oscillator1` | 10000 | `x, v → a` | 含驱动的阻尼非线性振子的加速度 |
| `oscillator2` | 10000 | `t, x, v → a` | 同上，额外给出时间 t |
| `PO0` | 4000 | `x, t, v → dv_dt` | 非线性谐振子的加速度 |
| `I.37.4_0_1` | 80000 | `Int, I2, delta → I1` | 两波源干涉：由合强度/第二波源强度/相位差求第一波源强度 |
| `I.48.2_1_0` | 59071 | `E_n, m, c → v` | 相对论：由总能量/相对论质量/光速求速度 |
| `II.6.15b_3_0` | 44348 | `Ef, epsilon, p_d, theta → r` | 电偶极子场：由场强/介电常数/偶极矩/夹角求距离 |
| `III.4.33_3_0` | 80000 | `E_n, h, omega, kb → T` | 量子谐振子：由第 n 模能量/普朗克常数/角频率/玻尔兹曼常数求温度 |

### 化学 / 生物 / 材料

| 数据集 | 行数 | 列（自变量… → 因变量） | 说明 |
|--------|------|------------------------|------|
| `CRK0` | 4000 | `t, A → dA_dt` | 化学反应动力学：浓度变化率 |
| `BPG0` | 4000 | `t, P → dP_dt` | 种群增长率 |
| `bactgrow` | 7500 | `b, s, temp, pH → db` | 大肠杆菌生长率（底物浓度/温度/pH） |
| `MatSci0` | 4000 | `epsilon, T → sigma` | 应力（应变/温度） |
| `stressstrain` | 2161 | `strain, temp → stress` | 铝棒弹塑性区应力-应变-温度 |

### 磁流变（MRF）—— 小样本

`lambda12 = L1/L2`、`lambda23 = L2/L3` 为颗粒长短轴比；`alpha` 控制超椭球表面曲率。

| 数据集 | 行数 | 列（自变量… → 因变量） | 模式 |
|--------|------|------------------------|------|
| `MRFShear-3` | 19 | `alpha, lambda12, lambda23 → miu` | 剪切 |
| `MRFShear-Cuboid` | 8 | `lambda12, lambda23 → miu` | 剪切 |
| `MRFShear-Ellipsoid` | 7 | `lambda12 → miu` | 剪切 |
| `MRFCompress-3` | 19 | `alpha, lambda12, lambda23 → sigma` | 压缩 |
| `MRFCompress-Cuboid` | 8 | `lambda12, lambda23 → sigma` | 压缩 |
| `MRFCompress-Ellipsoid` | 7 | `lambda12 → sigma` | 压缩 |

> MRF 系列样本量极小（7–19 行），参数拟合易过拟合，适合考察小样本下的方程发现能力。

## 批量运行

根目录 `example.sh` 提供批量运行示例（可用 `LLM_CONFIG` 环境变量指定配置文件）：

```bash
LLM_CONFIG=deepseek_deepseek-v4-flash.config bash example.sh
```

Windows 上用等价的 `example.bat`，可直接双击（会自动切到仓库根目录，并优先使用
`.venv2\Scripts\python.exe`）：

```bat
set LLM_CONFIG=config/deepseek_deepseek-v4-flash.config
example.bat
```

4 个单问题运行配置（`MRFShear-Cuboid` / `MRFShear-Ellipsoid` / `MRFCompress-Cuboid` /
`MRFCompress-Ellipsoid`）同样各有 `.sh` 与 `.bat`，参数与 `.idea/runConfigurations/`
下的同名 XML 一致。上述三份的逐项等价（以及 `background` 必须与数据表头相符）由
`tests/test_batch_scripts.py` 守住。
