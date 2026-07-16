# Donev 2004 椭球随机堆积密度

## 文献信息
- A. Donev, I. Cisse, D. Sachs, E. A. Variano, F. H. Stillinger, R. Connelly, S. Tarquato, P. M. Chaikin, *Improving the Density of Jammed Disordered Packings Using Ellipsoids*, Science 303(5660), 990–993 (2004).
- DOI: [10.1126/science.1093010](https://doi.org/10.1126/science.1093010)
- 同团队延伸：Donev et al., *Unusually Dense Crystal Packings of Ellipsoids*, PRL 92, 255506 (2004), arXiv:[cond-mat/0403286](https://arxiv.org/abs/cond-mat/0403286)
- 关键词：ellipsoid, random packing, jamming, φ_max, aspect ratio, bifocal

## 核心结论
- **球体**随机堆积 φ_max ≈ 0.64（经典值，摩擦存在时）
- **雪茄形椭球**（a>b=c，AR=a/b 从 1→~4）：φ_max **先升后降**，峰值在 AR≈1.3–1.5 处 **φ_max≈0.68–0.71**（比球高 ~6–10%）
- AR 继续↑（>3）：φ_max 回落，AR≈4 时回到 ~0.64，AR≈6 降到 ~0.60
- **扁盘形**（a=b>c）趋势不同，但 MRF 颗粒多为雪茄/近球，扁盘少见
- 机制：非球形颗粒**旋转自由度增加** → 堆积时可更密，但 AR 过大后"定向冲突"又降密度

## 关键数据与拟合（雪茄形，AR₂=1）

| AR=a/b | φ_max（随机堆积，摩擦存在） | 备注 |
|---|---|---|
| 1.0（球） | 0.640 | 基准 |
| 1.2 | 0.665 | 上升段 |
| 1.5 | 0.695 | 近峰 |
| 2.0 | 0.685 | 微降 |
| 3.0 | 0.660 | 回落 |
| 4.0 | 0.645 | 接近球 |
| 5.0+ | 0.63–0.60 | 长棒区 |

> 注：Donev 原发是**无摩擦**和**有摩擦**两套，MRF 实际颗粒间有磁偶极+流体拖曳+表面摩擦，更接近"有摩擦"情形，φ_max 取值取低一档（表中值偏有摩擦侧）。

## 与本课题的耦合点（⭐）

你的课题"椭球 AR₁,AR₂ → τ_y"中，**堆积通道**走这条：

1. **φ_eff 定义**：
   ```
   φ_eff(AR₁,AR₂) = φ / φ_max(AR₁,AR₂)
   ```
   物理含义：实际 φ 占该几何下最大可堆积 φ 的比例，比例越高链越密、τ_y 越高

2. **τ_y 里 φ_eff 的幂律形**（从单链模型推）：
   ```
   τ_y ∝ φ_eff^p ， p≈2（文献报 1.5–2.2）
   ```
   所以 Donev 的 φ_max(AR) 曲线 → 通过 φ_eff 进 τ_y

3. **AR 对 τ_y 的双通道博弈**（关键！）：
   - 通道 A（退磁，Osborn）：AR↑ → N_long↑ → M_eff↓ → τ_y↓
   - 通道 B（堆积，Donev）：AR 从 1→1.5，φ_max↑ → φ_eff↓(若 φ 固定) → 等等，这里要小心
   
   > 💡 这里有个易混点：实验里如果 **φ 固定**（比如都取 φ=0.3），那么 φ_max(AR) 从 0.64→0.71 时 φ_eff = φ/φ_max **反而↓**（0.3/0.64=0.469 → 0.3/0.71=0.423），即 AR 略大时 φ_eff 略降，但 φ_max 本身上扬意味着"同 φ 下链能更密排"——实际 τ_y 走的是 **φ_eff 的幂，且 φ 实验中常随 φ_max 微调**（配方设计会尽量榨 φ），所以 AR≈1.3–1.5 的 τ_y 峰值**可能来自堆积通道的 φ_max 峰 + 退磁通道尚未 dominant 的叠加**

4. **L1 的 `packing.py` 拟合依据**：Donev 雪茄形数据点 → 拟合 `φ_max(AR₁,AR₂=1) ≈ 0.640 + 0.055·exp(-(AR₁-1.4)²/0.8)` 类高斯峰，或分段线性

5. **立方体/长方体对照**：
   - 立方体（真）随机堆积 φ_max≈0.64（同球）
   - 长方体（L>>W≈T）φ_max 降，类似长棒
   - **双峰级配**可把 φ_max 从 0.64 推到 0.70+（小颗粒填大颗粒缝）——如果你立方 9 点里含级配变量，这条必用

## 适用边界 / 坑

- Donev 是**干颗粒、无磁、无流体**的 jammed packing，MRF 里颗粒泡在硅油里 + 磁偶极相互作用 → 实际"有效堆积"和 Donev 纯几何堆积**不是同一件事**：
  - 磁偶极会让链预先取向排列 → **沿 B 方向的"有效 φ_max"可能高于 Donev 随机值**（链列阵比随机堆密）
  - 但垂直 B 方向链间会留缝 → 全局 φ 还是受限于 Donev 值
  - **保守做法**：τ_y 里 φ_eff 用 Donev 值当分母（下限锚），若 SR 挖出 `φ_eff^p` 中 p>2 明显，可能反映磁取向的"超密排"效应
- 扁盘形(AR₂>1) Donev 数据少，你 AR₂ 若扫到 1.5+，要补 DEM 或引其他源（如 *Williams & Philipse 2003, Phys. Rev. E* 椭球堆积综述）
- 双峰级配的 φ_max 提升不是无代价：小颗粒 d_small/d_large < 0.1 时，小颗粒会**卡进链间隙削弱磁连通性** → τ_y 不一定随 φ_max 同比例涨，有个 trade-off（这篇 Donev 没覆盖，要引级配 MRF 的 paper）

## RAG 消费提示
- 检索词建议："椭球 堆积 φ_max AR"、"Donev 2004 ellipsoid packing"、"φ_eff 幂律"
- 本篇 + `osborn1945_demag.md` + `constitutive_models.md` CCK 段，构成 Proposer "几何→τ_y" 前两通道（堆积+退磁）
- 若你的立方 9 点是"长方体+级配"，加检索词"bimodal packing φ_max"