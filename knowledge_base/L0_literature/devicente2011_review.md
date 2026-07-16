# de Vicente et al. 2011 MRF Review（几何影响节）

## 文献信息
- J. de Vicente, D. J. Klingenberg, R. Hidalgo-Alvarez, *Magnetorheological fluids: a review*, Soft Matter 7, 3701–3710 (2011).
- DOI: [10.1039/c0sm01221a](https://doi.org/10.1039/c0sm01221a)
- 关键词：MRF, review, particle shape, aspect ratio, yield stress

## 几何影响核心结论（你课题的主锚）

### AR 对 τ_y 的非单调（⭐⭐⭐）
- 椭球 AR 从 1→~3：τ_y **先升后降**，峰值在 AR≈2–4（不同配方略有漂移）
- 升段机制：AR↑ → 链更易沿 B 取向 → 有效链长↑ → τ_y↑
- 降段机制：AR↑ → 退磁 N↑(Osborn) + φ_max↓(Donev) 占主导 → τ_y↓
- **立方/非球对照**：轻微非球(AR≈1.2–1.5) 在某些 φ 区间 τ_y 反略高于球（取向赢退磁），但 AR>3 后必低于球

### 颗粒形状类别对比（综述里提的）
- 球：基准，φ_max≈0.64，τ_y 最低（同 φ 下）
- 椭球/雪茄：AR≈2–3 峰值
- 立方体/棱角：τ_y 主项 ≤ 同 AR 椭球（M_eff↓+φ_max↓），但**滞后环面积/A_hys 可能反超**（棱角咬合滑移）
- 表面粗糙/包覆：降摩擦 → 屈服后 η↓、滞后环缩

### 级配
- 双峰级配抬 φ_max → τ_y 可涨，但 d_small/d_large < 0.1 时小颗粒卡链间隙 → τ_y 不按比例涨

## 与本课题耦合点（⭐）
- **Proposer 的"AR 非单调"趋势锚**：LLM 提 `τ_y(AR)` 时，应允许 AR∈[1,5] 有峰，峰位 ~2–4
- **立方体 vs 椭球的因变量选择**：
  - τ_y 主项：立方 ≤ 椭球（此篇锚定）
  - A_hys/滞后：立方可能 ≥ 椭球（棱角效应，此篇提但数据少，可挖）
- **级配若在你立方 9 点里**：此篇给 φ_max 抬升 + 小颗粒卡链 的 trade-off 锚

## RAG 消费提示
- 检索词："de Vicente 2011 颗粒形状 AR τ_y 非单调"、"MRF 立方体 棱角 滞后"
- 本篇 + 三锚(Osborn/Donev/Klingenberg) = Proposer "AR 趋势全貌"