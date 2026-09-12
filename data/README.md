# data/ —— 基准数据集

本目录存放 DrSR 的基准数据集。**每个子目录对应一个"问题"，至少包含 `train.csv`。**

## 使用约定

- CSV 必须带表头：**前 n-1 列为自变量（特征），最后一列为因变量**。
- `main.py --data_csv <路径>` 读取该 CSV；列名会作为提示词中的自变量/因变量名传给模型。
- `--problem_name` 仅用于结果目录命名（`experiments/{problem_name}_{timestamp}/`），
  匿名实验中不会把该名字暴露给模型（见 `prompt_config.PromptContext.problem`）。

## 文件命名（各数据集不完全一致，历史原因）

| 文件名 | 含义 |
|--------|------|
| `train.csv` | 训练集（**必需**，主流程唯一读取的文件） |
| `test.csv` | 同分布测试集 |
| `test_id.csv` | 同分布（in-distribution）测试集 |
| `test_ood.csv` / `ood_test.csv` | 分布外（out-of-distribution）测试集，两种拼写并存 |
| `train_noise.csv` | 加噪训练集（仅 oscillator2） |

> 当前主流程只使用 `train.csv`；测试集供外部评估脚本使用。

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
