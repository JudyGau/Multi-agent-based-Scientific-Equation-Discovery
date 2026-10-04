# 投稿期刊调研（路线 A：MRF 本构的可信发现）

> **背景**：力学硕士；毕业需 SCI；暂不考虑纯计算机类期刊/会议。
> **论文定位（路线 A）**：应用型 —— 数据稀缺条件下磁流变弹性体（MRE/MRF）本构方程的**可信数据驱动发现**；benchmark 数据仅作泛化性旁证。
> **数据现实**：MRF 仅 6 个体系、7–19 点，评估只能走 LOO（留一法），不声称泛化/OOD。
> **整理日期**：2026-10-04
> **数据来源**：期刊官网、LetPub、科研通(ablesci)、SciRev、出版社 APC 页面等公开信息。
> **重要提醒**：JCR 分区与中科院分区**逐年变动**，APC 与审稿周期各来源常不一致 → **投稿前必须以 Clarivate JCR、中科院文献情报中心分区表、以及期刊官网 APC 页实时复核**。本文件仅供初步筛选用。

---

## 0. 结论速览

- **最对口 + 免版面费**：`Smart Materials and Structures`（MRE 娘刊）、`Mechanics of Materials`（本构主战场）。
- **想冲分区高一点且免版面费**：`Int. J. Mechanical Sciences`（工程技术 1 区 Top，可走订阅免费通道）。
- **偏 AI/交叉且免版面费**：`Engineering Applications of AI`、`Advanced Engineering Informatics`（均中科院 1 区 Top，订阅免费）。
- **彻底 OA 且必付费**（除非有经费，否则优先不选）：`npj Computational Materials`、`Machine Learning: Science & Technology`、`J. Computational Design and Engineering`、`Digital Discovery`。
- **方向未定时最稳妥（A/B 都能投）**：`EAAI`、`Advanced Engineering Informatics`（免 APC、1 区 Top）、`npj Computational Materials`、`Machine Learning: Science & Technology`、`J. Computational Design and Engineering` —— 详见第 8 节。
- **强烈不建议**：`Mechanics of Advanced Materials and Structures`（2025 年被 WOS 列入 On Hold）。

> 币值概算：1 USD ≈ 7.1 CNY，1 EUR ≈ 7.7 CNY，1 GBP ≈ 9.0 CNY。

---

## 路线 A / 路线 B 详细说明（选题与定位）

> 本文档的期刊清单服务于**路线 A**。此处完整记录两条路线的内容，便于与导师对照、以及日后切换。

### 路线 A：应用型 —— MRF 本构方程的可信发现（**已选定**）

- **一句话定位**：在数据极度稀缺（n ≤ 19）且存在一维不可辨识山脊的条件下，用**可信性判据**（数值病理 + 可辨识性）替代纯 goodness-of-fit 选解，得到**物理可解释的磁流变弹性体（MRE/MRF）本构方程**。
- **论文主题**：数据稀缺条件下 MRF 本构方程的可信数据驱动发现。
- **核心交付物**：一条（或一组）覆盖 6 个 MRF 体系的**本构方程** + 其可信性论证。
- **主实验与数据**：
  - MRF 六体系（压缩：Cuboid 8 点 / Ellipsoid 7 点 / -3 17 点；剪切：同构）——**主贡献与主实验**；
  - benchmark 12 题（LLM-SR / LLM-SRBench）——仅作**方法可迁移性旁证**。
- **评价口径**：**LOO（留一法）**，报 median / 95th 分位 LOO NMSE + 逐点误差；6 体系**跨体系一致性**（压缩 vs 剪切、Cuboid/Ellipsoid/-3）作为外部证据；**不声称泛化 / 不做 OOD**。
- **卖点（创新叙事）**：把"可信性"讲成主张——**病理体检（拦下尖峰/门控/大系数抵消）+ 可辨识性（声明秩亏方向）**，回答"为什么这条本构可信"，而非"我们用了 LLM"。
- **论文骨架**：
  1. 绪论：MRF 本构建模现状（超弹性 / 唯象模型）+ 小样本建模难点；
  2. 方法：LLM-SR + 病理体检 + 可辨识性约束（讲成"可信手段"）；
  3. 案例：6 个 MRF 体系的方程发现（LOO + 跨体系一致性）；
  4. 可信性论证：病理 / 可辨识 / 物理解释 + 与已有 MRF 本构的定性对照；
  5. benchmark 旁证：方法可迁移；
  6. 局限：n 小、不作泛化声明。
- **模块取舍**：
  - **强**：数值病理体检、可辨识性、LOO、物理解释、跨体系一致性；
  - **必补**：`holdout.py` 的 LOO 通道、一条可信 MRF 本构；
  - **砍 / 降级**：可归因编排、元控制器、bandit、多种子协议。
- **期刊落点**：Mechanics of Materials / Smart Materials and Structures / JIMSS / Computational Materials Science / Acta Mechanica Sinica 等（详见第 1–3 节）。
- **优点**：契合"力学硕士 + 非纯 CS SCI"约束；直接服务课题组刚需；MRF 数据与产物已具备；证据链缺口最小。
- **风险 / 局限**：
  - n 太小 → 只能 LOO，**不能声称泛化/OOD**；
  - 审稿人必问"为何不用已有 MRF 本构模型（超弹性/Ogden/唯象）"与"没有新数据凭什么信"→ 靠可信性判据 + LOO + 跨体系一致性作答。
- **前置工作**：① `holdout.py` 加 LOO 通道（train < 30 自动留一法）；② 用现有 MRF 产物出一版力学刊 framing。

### 路线 B：方法型 —— 可信性感知的符号回归智能体

- **一句话定位**：现有 LLM-SR 方法（含 DrSR、DE、A-SR 同属此列）系统性地忽略**数值病理与可辨识性**；本文把"可验证的可信性判据"从收尾诊断提升为**一等奖励与诊断信号**。
- **论文主题**：可信性感知的符号回归智能体（Trustworthy / Verifiability-aware Symbolic Regression）。
- **核心交付物**：一套方法（病理判据 + 可辨识性约束 + 诊断定向反馈）+ 在标准 benchmark 上的收益证明。
- **主实验与数据**：12 题 benchmark（LLM-SR / LLM-SRBench）为**主表**；MRF 为案例研究。
- **评价口径**：SA / Acc@0.1 / median NMSE / ID 与 OOD 分开报；**固定 3–5 种子 + mean±std**；报 token 与墙钟成本。
- **卖点（创新叙事）**：方法增量本身——病理作为"第三类失败模式"、可辨识性进入闭环、可归因编排与细粒度可验证信用分配。
- **论文骨架**：绪论（三代演进 + 三类失败模式）→ 相关工作（正面区分 A-SR / DE）→ 方法 → 可归因编排 → 实验（主表 + 消融 + 效率 + 稳健性）→ 讨论。
- **模块取舍**：**必须有** baseline（PySR / gplearn / LLM-SR）+ 消融（E2 六项）+ 多种子 + 成本；可归因编排（创新 3）为加分项。
- **期刊落点**：MLST / EAAI / Advanced Engineering Informatics / npj Computational Materials / Nature 子刊（天花板）。
- **优点**：影响力上限更高；方法叙事"新"；benchmark 现成。
- **风险 / 局限**：**证据链缺口最大**——无 baseline、无消融、且已实证"n=3 多种子对照在本方差下结构性无效"；需与 LLM-SR 社区正面比拼；投入周期长；且更偏 AI，触及"非纯 CS"约束的边界。

### A / B 对照

| 维度 | 路线 A（应用型） | 路线 B（方法型） |
|---|---|---|
| 主题 | MRF 本构的可信发现 | 可信性感知的符号回归方法 |
| 主实验 | MRF（6 体系，LOO） | 12 题 benchmark + baseline + 消融 |
| benchmark 角色 | 泛化性旁证 | 主表 |
| 基线压力 | 低 | 高（PySR/gplearn/LLM-SR + 多种子） |
| 证据链缺口 | 小 | 大 |
| 对口刊 | 力学 / 材料 / 智能材料 | AI4Science / AI+工程 |
| 与"非纯 CS"约束 | ✅ 完全契合 | ⚠️ 偏 AI |
| 建议 | **当前选定** | 备选（需补方法证据） |

### 决策依据

- 项目**本源**是服务课题组 MRF 本构拟合 → 天然是"应用型"工作；
- 学生为**力学硕士**、需**非纯 CS 的 SCI** → 路线 A 更契合；
- 项目既有差异化机制（病理 + 可辨识性）**恰好为"小样本 + 一维不可辨识 + 数值病理"设计**，正是 MRF 的痛点 → 路线 A 能直接复用，路线 B 反而要额外补大实验。

### 若要切换 / 确认方向，先问导师这一个问题

> **"毕业论文是要给出一条可用的 MRF 本构方程，还是要做一套通用的符号回归方法？"**
> 答"要本构" → 路线 A；答"要方法" → 路线 B。

---

## 1. 智能材料 / 力学本构（路线 A 核心）

| 期刊 | JCR 分区（小类） | 中科院分区（大类 / 小类 / Top） | OA 类型 | 费用 | 投稿周期 |
|---|---|---|---|---|---|
| **Smart Materials and Structures** (IOP) | IF 3.8；仪器仪表 **Q2**(26/81)、材料综合 **Q2**(234/472) | 材料科学 **3区**；仪器仪表 2区、材料综合 4区；**非 Top** | 混合 OA | 订阅**免费**；金色 OA **£2655 / €3050 / $3665** | ~3 个月 |
| **Mechanics of Materials** (Elsevier) | IF 4.1；力学 **Q1**(37/172)、材料综合 Q2 | 材料科学 **3区**；材料综合 3区、力学 3区；**非 Top** | 混合 OA | 订阅**免费**；金色 OA **$3590** | 官网初审 ~10.1 周；网友 ~7 月 |
| **Int. J. Solids and Structures** (Elsevier) | IF 4.6；力学 **Q1**(30/172) | 工程技术 **2区**；力学 2区；Top **存疑** | 混合 OA | 订阅**免费**；金色 OA **$3590** | 初审 9 天、送审后 63 天、投稿→接收 123 天 |
| **J. Intelligent Material Systems and Structures** (SAGE) | IF 2.2；材料综合 **Q3**(305/472) | 材料科学 **4区**；材料综合 4区；**非 Top** | 订阅制 | 投稿/发表**均免费**；SAGE Choice APC 未查到 | 3–8 周 |
| ⚠️ **Mechanics of Advanced Materials and Structures** (T&F) | **2025 被 WOS 列入 On Hold，2025 JCR 无 IF/分区**（2024: IF 3.6） | 材料科学 3区；表征与测试 2区；非 Top | 混合 OA | 订阅无 APC；OA 金额未查到 | 初审 28 天 |

---

## 2. 计算材料 / 计算力学

| 期刊 | JCR 分区（小类） | 中科院分区（大类 / 小类 / Top） | OA 类型 | 费用 | 投稿周期 |
|---|---|---|---|---|---|
| **Int. J. Mechanical Sciences** (Elsevier) | IF 11.4；工程机械 **Q1**(4/184)、力学 **Q1**(5/172) | 工程技术 **1区**；工程机械 1区、力学 1区；**Top 是** | 混合 OA | 订阅**免费**；金色 OA **$4980** | submission→final 6.4 周；网友 ~2.7 月 |
| **Computational Mechanics** (Springer) | IF 4.5；数学跨学科 **Q1**(13/135)、力学 **Q1**(33/170) | 工程技术 **2区**；数学跨学科 2区、力学 2区；**非 Top** | 混合 OA | 订阅**免费**；金色 OA **£2590 / $3990 / €2990** | 4–8 周 |
| **Acta Mechanica Sinica** (Springer / 中国力学学会) | IF 4.6；工程机械、力学 **双 Q1**（主办方称） | 工程技术 **2区**；工程机械 2区、力学 2区；**非 Top** | 混合 OA | 订阅**免费**；APC 三版不一致（€3190 / $4190 / £2790 等） | 官方 2023：平均 47 天；网友 ~5 月 |
| **Computational Materials Science** (Elsevier) | IF 3.7；材料综合 **Q2**(227/460) | 材料科学 **3区**；材料综合 3区；**非 Top** | 混合 OA | 订阅**免费**；金色 OA **$3470** | 首轮 1.7 月、总 2.0 月 |
| **npj Computational Materials** (Nature) | IF ~9；物理化学 **Q1**(24/191)、材料综合 **Q1**(54/472) | 材料科学 **1区**；物理化学 1区、材料综合 1区；**Top 是** | **完全 OA** | **必付**：Research **£2790 / $3590 / €3090**；Brief/Review ~£1355；低收入可豁免 | 6 周；中位 146 天 |

---

## 3. AI + 工程交叉（偏 AI，非纯 CS）

| 期刊 | JCR 分区（小类） | 中科院分区（大类 / 小类 / Top） | OA 类型 | 费用 | 投稿周期 |
|---|---|---|---|---|---|
| **Engineering Applications of AI (EAAI)** (Elsevier) | **Q1×4**：自动化控制 8/88、CS-AI 30/210、工程电子电气 27/369、工程综合 6/178 | 计算机科学 **1区**；工程综合 1区、自动化控制 2区、计算机 AI 2区；**Top 是** | 混合 OA | 订阅**免费**；金色 OA **$3040** | 官网投稿→接收 207 天（~6.9 月）；作者 ~9 月 |
| **Advanced Engineering Informatics** (Elsevier) | **Q1**：CS-AI 17/210、工程综合 4/178 | 工程技术 **1区**；计算机 AI 1区、工程综合 1区；**Top 是** | 混合 OA | 订阅**免费**；金色 OA **$3380**（第三方） | 初审 9.3 周；第三方投稿→接收 3–4.5 月 |
| **Machine Learning: Science & Technology** (IOP) | CS-AI Q2；**综合性期刊 Q1**(26/140) | 物理与天体物理 **2区**；综合性期刊 2区、计算机 AI 3区；**非 Top** | **完全 OA** | **必付** **£2680 / €3215 / $3350** | 初审 ~5 周 |
| **J. Computational Design and Engineering** (OUP) | **Q1**：CS-跨学科 34/185、工程综合 11/178 | 工程技术 **2区**；计算机跨学科 3区、工程综合 3区；**非 Top** | **完全 OA** | **必付** **$2715** | 投稿→初审 6 周；投稿→接收 15 周 |
| ⚠️ **Digital Discovery** (RSC) | Q1 但**仅 ESCI（非 SCIE）** | 计算机科学 **2区**（2024 版无记录）；化学综合 2区 | **完全 OA** | **必付** **£2200**；机构协议可免 | 初审 40 天 |

---

## 4. 警戒项（投稿前必须处理）

1. **Mechanics of Advanced Materials and Structures**：2025 年被 WOS 列入 **On Hold**，2025 JCR 已无 IF 与分区 → **不建议投**，存在检索风险。
2. **Digital Discovery**：**仅 ESCI，不在 SCIE** → 若毕业要求"SCIE 收录"，可能不被认可，先确认学校规定。
3. **Machine Learning: Science & Technology**：检索状态第三方说法矛盾（一说"未被最新 JCR 收录"，LetPub 称 SCIE）→ **投稿前务必核实收录状态**。
4. **APC 数值来源不一**（官网 vs LetPub vs 第三方）：npj、Acta Mechanica Sinica、IJMS、MLST、JCDE、AEI 均存在差异 → **以期刊官网为准**。
5. **分区版本混用**：JCR 多为 2025 版（2026-06 发布，部分站点仍混用 2024 版）；中科院为 2025 升级版（2025-03 发布）；另有非官方的"2026 新锐版"被部分站点标注，勿混用。
6. **"是否收取审稿费/超页费/彩图费"**：多数期刊官网未提及，本表标注为"未见/未查到"，**不等于确认无费用**。

---

## 5. 相对路线 A 的适配度与选刊建议

- **不想付版面费** → 选**混合 OA 期刊走订阅通道**：`Smart Materials and Structures`、`Mechanics of Materials`、`Int. J. Solids and Structures`、`Int. J. Mechanical Sciences`、`Computational Materials Science`、`Acta Mechanica Sinica`、`Computational Mechanics`、`EAAI`、`Advanced Engineering Informatics`。
- **最对口（MRE 本构）**：`Smart Materials and Structures`、`Mechanics of Materials`、`J. Intelligent Material Systems and Structures`。
- **冲分区**：`Int. J. Mechanical Sciences`（工程技术 1 区 Top，订阅免费）、`npj Computational Materials`（1 区 Top，但必付 APC）。
- **偏 AI/交叉**：`EAAI`、`Advanced Engineering Informatics`（1 区 Top，订阅免费）；若求快且不介意付费，`J. Computational Design and Engineering`（投稿→接收 ~15 周，必付 APC）。
- **JIM（Journal of Intelligent Manufacturing）**：与本路线"材料本构"scope 不匹配（JIM 面向制造），除非工作本身带制造/工艺属性（如磁场固化制备、增材制造、器件制造），否则不建议。

---

## 6. 待核实清单（投稿前逐项确认）

- [ ] 目标刊最新一年的 **JCR 分区**（Clarivate 官方）。
- [ ] 目标刊最新版 **中科院分区（大类 + 小类 + Top）**（中科院文献情报中心官方）。
- [ ] **收录状态**：是否仍在 SCIE（尤其 MLST、Digital Discovery）。
- [ ] **APC / 订阅免费通道**：以期刊官网 APC 页为准。
- [ ] **是否收取审稿费 / 超页费 / 彩图费**：向编辑部或作者指南确认。
- [ ] **审稿周期**：以期刊官网"review speed"页或近期作者经验为准。

---

## 7. 路线 B 推荐期刊（方法型：可信性感知的符号回归）

> 路线 B 是 AI4S / AI 方法型，期刊偏 AI 与交叉。
> **重要张力**：路线 B 与"暂不考虑纯计算机类"的约束存在冲突——其中 `Nature Machine Intelligence`、`Nature Computational Science` 在中科院分区中计为**计算机科学 1 区**，部分单位视为计算机类；投稿前请确认学校/学院的认定口径。
> 币值概算：1 USD ≈ 7.1 CNY、1 EUR ≈ 7.7、1 GBP ≈ 9.0。

### 7.1 天花板（要求真实科学发现 / 方法极强）

| 期刊 | JCR 分区（小类） | 中科院分区（大类 / 小类 / Top） | OA 类型 | 费用 | 投稿周期 |
|---|---|---|---|---|---|
| **Nature Machine Intelligence** | IF 29.8；计算机:AI **Q1**(2/210)、计算机:跨学科 **Q1**(1/185) | 计算机科学 **1区**；计算机:AI 1区、计算机:跨学科 1区；**Top 是** | 混合 OA | 订阅**免费**；金色 OA **~US$12,690–12,850**（≈¥9万） | 首次编辑决定中位 10 天；投稿→接收中位 ~231 天 |
| **Nature Computational Science** | IF 20.3；计算机:跨学科 **Q1**(3/185)、计算机:理论方法 **Q1**(2/146)、综合性期刊 **Q1**(7/140) | 计算机科学 **1区**；小类均 1区；**Top 是** | 混合 OA | 订阅**免费**；金色 OA **€9,500**（≈¥7.4万） | 初审 ~9 天；投稿→接收 ~4–6 个月 |

### 7.2 AI4Science 主力

| 期刊 | JCR 分区（小类） | 中科院分区（大类 / 小类 / Top） | OA 类型 | 费用 | 投稿周期 |
|---|---|---|---|---|---|
| **Machine Learning: Science & Technology** (IOP) | CS-AI Q2；**综合性期刊 Q1**(26/140) | 物理与天体物理 **2区**；综合性期刊 2区、计算机 AI 3区；**非 Top** | **完全 OA** | **必付** **£2680 / €3215 / $3350**（≈¥2.4万） | 初审 ~5 周 |
| **npj Computational Materials** (Nature) | IF ~9；物理化学 **Q1**(24/191)、材料综合 **Q1**(54/472) | 材料科学 **1区**；物理化学 1区、材料综合 1区；**Top 是** | **完全 OA** | **必付**：Research **£2790 / $3590 / €3090**（≈¥2.5万）；Brief/Review ~£1355 | 6 周；中位 146 天 |
| **Computer Physics Communications** (Elsevier) | IF 3.9；物理:数学物理 **Q1**(2/61)、计算机:跨学科 Q2(81/185) | 物理与天体物理 **2区**；物理:数学物理 2区、计算机:跨学科 3区；**非 Top** | 混合 OA | 订阅**免费**；金色 OA **~US$3,410–3,700**（≈¥2.5万） | 初审 ~11 天–6 周；投稿→接收 5.3–12 个月（源差异大） |
| **Scientific Reports** (Nature) | IF 4.9；综合性期刊 **Q1**(21/140) | 综合性期刊 **3区**；综合性期刊 3区；**非 Top** | **完全 OA** | **必付** **US$2,850**（≈¥2.1万）；可申请豁免/折扣 | 投稿→初审 56 天、→接收 133 天 |
| ⚠️ **Digital Discovery** (RSC) | Q1 但**仅 ESCI（非 SCIE）** | 计算机科学 **2区**（2024 版无记录）；化学综合 2区 | **完全 OA** | **必付** **£2200**；机构协议可免 | 初审 40 天 |

### 7.3 AI + 工程交叉（写法与路线 A 兼容，免 APC 可选）

| 期刊 | JCR 分区（小类） | 中科院分区（大类 / 小类 / Top） | OA 类型 | 费用 | 投稿周期 |
|---|---|---|---|---|---|
| **Engineering Applications of AI (EAAI)** (Elsevier) | **Q1×4**：自动化控制 8/88、CS-AI 30/210、工程电子电气 27/369、工程综合 6/178 | 计算机科学 **1区**；工程综合 1区、自动化控制 2区、计算机 AI 2区；**Top 是** | 混合 OA | 订阅**免费**；金色 OA **$3040** | 官网投稿→接收 207 天（~6.9 月） |
| **Advanced Engineering Informatics** (Elsevier) | **Q1**：CS-AI 17/210、工程综合 4/178 | 工程技术 **1区**；计算机 AI 1区、工程综合 1区；**Top 是** | 混合 OA | 订阅**免费**；金色 OA **$3380**（第三方） | 初审 9.3 周；投稿→接收 3–4.5 月 |
| **J. Computational Design and Engineering** (OUP) | **Q1**：CS-跨学科 34/185、工程综合 11/178 | 工程技术 **2区**；计算机跨学科 3区、工程综合 3区；**非 Top** | **完全 OA** | **必付** **$2715**（≈¥2.0万） | 投稿→初审 6 周；投稿→接收 15 周 |

### 7.4 备选（更偏材料/化学，或相对易中）

| 期刊 | JCR 分区（小类） | 中科院分区（大类 / 小类 / Top） | OA 类型 | 费用 | 投稿周期 |
|---|---|---|---|---|---|
| **Advanced Intelligent Systems** (Wiley) | IF 6.7；自动化控制 **Q1**(15/88)、计算机:AI **Q1**(43/210)、机器人学 **Q1**(10/48) | 计算机科学 **3区**；小类均 3区；**非 Top** | **完全 OA** | **必付** **US$3,970–4,410**（≈¥2.8–3.2万，来源不一） | ~4 周（作者反馈） |
| **Advanced Theory and Simulations** (Wiley) | IF 3.2；综合性期刊 **Q2**(39/140) | 工程技术 **3区**；综合性期刊 4区；**非 Top** | 混合 OA | 订阅**免费**；金色 OA **~US$4,220 / €3,520**（≈¥3万） | 投稿→初审中位 20 天 |
| **J. Chemical Information and Modeling** (ACS) | IF 6.4；药物化学/计算机:信息系统/计算机:跨学科 **Q1**、化学:综合 Q2（2024 版小类） | 化学 **2区**；化学:综合 2区、药物化学 3区、计算机:跨学科 3区；**Top 是** | 混合 / Transformative | 订阅**免费**；专属 APC **未查到**（ACS AuthorChoice 一般 ~$4000 档） | ~3 个月（作者反馈） |

### 7.5 明确排除（按"暂不考虑纯计算机领域"的约束）

以下为纯计算机/机器学习类期刊与会议，**按你的约束排除**、不列入路线 B：

- 期刊：**TMLR、JMLR、IEEE TPAMI、IEEE TNNLS、Artificial Intelligence (AIJ)、Machine Learning (Springer)**。
- 会议：**NeurIPS / ICML / ICLR / AAAI / IJCAI**。
- （备注：`Nature Machine Intelligence`、`Nature Computational Science` 虽为综合/AI 顶刊，但中科院计为计算机科学 1 区，是否算"纯计算机"由学校认定为准。）

### 7.6 路线 B 选刊建议与警戒

- **能付 APC + 想贴 AI4S 身份** → `Machine Learning: Science & Technology`、`npj Computational Materials`。
- **兼容路线 A 写法 + 免 APC** → `Engineering Applications of AI`、`Advanced Engineering Informatics`（均中科院 1 区 Top，订阅免费）。
- **快速见刊** → `Scientific Reports`（完全 OA，~133 天）、`J. Computational Design and Engineering`（~15 周）。
- **天花板** → `Nature Machine Intelligence`、`Nature Computational Science`（需真实科学发现 + APC 高昂 ~¥7–9 万）。

**警戒项**：
1. **APC 高昂**：NMI ~¥9万、NCS ~¥7.4万；且二者被视为"计算机类"，未必符合你的约束。
2. **Digital Discovery**：**仅 ESCI，不在 SCIE** → 毕业认定可能不认。
3. **Advanced Intelligent Systems**：APC 三处来源不一致（$3,190 / $3,970 / $4,410）→ 以官网为准。
4. **JCIM**：专属 APC 未查到，需向 ACS 确认。
5. **Scientific Reports**：中科院综合性期刊 **3区**，部分单位认可度一般。
6. **Computer Physics Communications**：投稿→接收周期来源差异极大（5.3–12 个月）。

---

## 8. 路线 A / B 通用期刊（两条路线都可投）

> **判据**：期刊 scope 同时接受"应用优先（MRF 本构发现）"与"方法优先（可信性感知的符号回归）"两种写法。这些是**方向未最终定**时最稳妥的落点。

| 期刊 | JCR 分区（小类） | 中科院分区（大类 / Top） | OA 类型 | 费用 | 投稿周期 | 两条路线如何摆 |
|---|---|---|---|---|---|---|
| **Engineering Applications of AI (EAAI)** (Elsevier) | **Q1×4**：自动化控制 8/88、CS-AI 30/210、工程电子电气 27/369、工程综合 6/178 | 计算机科学 **1区**；**Top 是** | 混合 OA | 订阅**免费**；金色 OA **$3040** | 投稿→接收 207 天（~6.9 月） | A：讲"AI 用于本构建模"；B：讲"可信 SR 方法" |
| **Advanced Engineering Informatics** (Elsevier) | **Q1**：CS-AI 17/210、工程综合 4/178 | 工程技术 **1区**；**Top 是** | 混合 OA | 订阅**免费**；金色 OA **$3380**（第三方） | 初审 9.3 周；投稿→接收 3–4.5 月 | 同 EAAI，工程/信息学接口 |
| **npj Computational Materials** (Nature) | IF ~9；物理化学 **Q1**(24/191)、材料综合 **Q1**(54/472) | 材料科学 **1区**；**Top 是** | **完全 OA** | **必付**：Research **£2790 / $3590 / €3090**；Brief/Review ~£1355 | 6 周；中位 146 天 | A：讲"材料本构发现"；B：讲"ML 用于计算材料" |
| **Machine Learning: Science & Technology** (IOP) | CS-AI Q2；**综合性期刊 Q1**(26/140) | 物理与天体物理 **2区**；**非 Top** | **完全 OA** | **必付** **£2680 / €3215 / $3350** | 初审 ~5 周 | AI4S 主刊，两种写法都收 |
| **J. Computational Design and Engineering** (OUP) | **Q1**：CS-跨学科 34/185、工程综合 11/178 | 工程技术 **2区**；**非 Top** | **完全 OA** | **必付** **$2715** | 投稿→初审 6 周；→接收 15 周 | 计算设计 + AI，两种写法都收 |
| ⚠️ **Digital Discovery** (RSC) | Q1 但**仅 ESCI（非 SCIE）** | 计算机科学 **2区**（2024 版无记录） | **完全 OA** | **必付** **£2200**；机构协议可免 | 初审 40 天 | 两种写法都收；**收录风险**（仅 ESCI） |

### 8.1 通用表的选刊取舍

- **免 APC 优先** → `EAAI`、`Advanced Engineering Informatics`（均中科院 1 区 Top，可走订阅免费通道）。
- **愿付 APC 且想贴 AI4S 身份** → `npj Computational Materials`、`Machine Learning: Science & Technology`。
- **快速见刊** → `J. Computational Design and Engineering`（~15 周）。
- **避开** → `Digital Discovery`（仅 ESCI，毕业认定可能不认）。

### 8.2 边界情况（口径宽、两路线也能沾，但有取舍）

| 期刊 | 中科院分区 | 说明 |
|---|---|---|
| **Scientific Reports** (Nature) | 综合性期刊 **3区** | 口径最宽，两种写法都收；但分区较低、部分单位认可度一般；完全 OA 必付 **US$2,850** |
| **Advanced Theory and Simulations** (Wiley) | 工程技术 **3区** | 计算/模拟 + ML，两路线都能沾；订阅**免费**，金色 OA ~US$4,220 |
| **Advanced Intelligent Systems** (Wiley) | 计算机科学 **3区** | AI + 智能系统，智能材料 + AI 可两者；但**完全 OA 必付 US$3,970–4,410** |
| **J. Chemical Information and Modeling** (ACS) | 化学 **2区**（Top） | **偏化学**：路线 B 的 ML 方法契合，路线 A 的 MRE 本构为弱契合 → 更适合作 B |

---

## 附：主要数据来源

- Smart Materials and Structures — IOP About / 作者指南：https://publishingsupport.iopscience.iop.org/journals/smart-materials-and-structures/about-smart-materials-structures/
- Int. J. Solids and Structures — Elsevier 期刊页：https://www.journals.elsevier.com/international-journal-of-solids-and-structures
- Int. J. Mechanical Sciences / Computational Materials Science — ScienceDirect OA options：
  - https://www.sciencedirect.com/journal/international-journal-of-mechanical-sciences/publish/open-access-options
  - https://www.sciencedirect.com/journal/computational-materials-science/publish/open-access-options
- npj Computational Materials — Nature APC 页：https://www.nature.com/npjcompumats/about/apc
- Acta Mechanica Sinica — Springer 投稿页：https://www.springer.com/journal/10409/how-to-publish-with-us
- Computational Mechanics — Springer 投稿页：https://www.springer.com/journal/466/how-to-publish-with-us
- Machine Learning: Science & Technology — IOP About：https://publishingsupport.iopscience.iop.org/journals/machine-learning-science-and-technology/about-machine-learning-science-technology/
- Digital Discovery — RSC：https://www.rsc.org/publishing/journals/digital-discovery
- J. Computational Design and Engineering — OUP：https://academic.oup.com/jcde
- LetPub 期刊选择器：https://www.letpub.com/journal-selector
- 科研通 ablesci 期刊库：https://www.ablesci.com/journal/index
- SciRev（审稿周期）：https://scirev.org/
- Springer Nature 混合期刊 APC 价目（Couperin 2025）：https://www.couperin.org/wp-content/uploads/2024/07/Springer_Tarifs_APC_oct2025.pdf

路线 B 补充来源：
- Scientific Reports — 官网/APC：https://www.nature.com/srep/
- Computer Physics Communications — Elsevier：https://www.elsevier.com/journals/computer-physics-communications/0010-4655
- Advanced Intelligent Systems — Wiley OA：https://advanced.onlinelibrary.wiley.com/hub/journal/26404567/open-access
- Advanced Theory and Simulations — Wiley：https://advanced.onlinelibrary.wiley.com/journal/25130390
- JCIM — ACS About / 定价：https://pubs.acs.org/page/jcisd8/about.html ；https://acsopenscience.org/researchers/oa-pricing/
- Wiley 混合期刊 APC 价目（2024）：https://www.uv.es/investsbd/openaccess/Wiley-Journal-APCs-OnlineOpen.pdf
- Nature Portfolio APC 说明：https://www.springernature.com/gp/article-processing-charges-faqs/14238042