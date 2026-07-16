# Osborn 1945 三轴椭球退磁因子

## 文献信息
- J. A. Osborn, *Demagnetizing Factors of the General Ellipsoid*, Phys. Rev. 67(11), 351–357 (1945).
- DOI: [10.1103/PhysRev.67.351](https://doi.org/10.1103/PhysRev.67.351)
- 关键词：ellipsoid, demagnetizing factor, magnetic anisotropy, AR

## 核心结论
- 给出**任意三轴椭球**三个主轴方向的退磁因子 N_a, N_b, N_c 的**闭式表达式**（椭圆积分形），满足 N_a+N_b+N_c=1。
- 球 (a=b=c): N_a=N_b=N_c=1/3
- 长旋转椭球 (a>b=c, 雪茄形): N_a 随 a/b 增大而增大，N_b=N_c 减小
- 扁旋转椭球 (a=b>c, 盘形): 反之

## 公式与参数物理含义

三轴椭球半轴 a≥b≥c>0，定义：
```
e² = 1 − (c/a)²   （第一偏心率平方，沿长轴 a 方向）
```

沿长轴 a 的退磁因子：
```
N_a = (1−e²)/(2e³) · [ ln((1+e)/(1−e)) − 2e ]
```

沿中轴 b、短轴 c 的 N_b, N_c 由对称轮换得（Osborn 原文式 (18)-(20)），且：
```
N_a + N_b + N_c = 1
```

| 符号 | 单位 | 物理含义 |
|---|---|---|
| a,b,c | m | 椭球三半轴，L≥W≥T 对应 a,b,c |
| AR₁ = a/b, AR₂ = b/c | 无量纲 | 你课题的自变量 |
| N_a,N_b,N_c | 无量纲 | 退磁因子，值越大表示该方向磁化越被"自身场"抵消 |
| e | 无量纲 | 第一偏心率，e²=1−(c/a)² |

## 与本课题的耦合点（⭐）

你的课题是"椭球 AR₁,AR₂ → 磁流变效应(τ_y)"，Osborn 这篇是**退磁通道的理论源头**：

1. **沿链取向取 N_long**：MRF 中颗粒链沿 B 场取向，通常取长轴方向为链轴 → 用 N_a(AR₁,AR₂)
2. **有效磁化 M_eff 的稀释形**（线性磁化近似）：
   ```
   M_eff = M_s · χ / (1 + N_long·χ)  或  M_eff ∝ (1 − N_long·χ/(1+N_long·χ))
   ```
   即 AR↑ → N_long↑ → M_eff↓ → τ_y↓（退磁稀释通道）
3. **L1 的 `demag.py` 直接实现 N_a(AR₁,AR₂)`**：Osborn 式是代码的理论依据，AR₁∈[1,5], AR₂∈[1,2] 可预算查表
4. **Proposer 候选式里的 `f_N(AR)` 项**：从这篇来，形为 `[1 − N_long(AR)]^{α}` 或 `[M_s·χ/(1+N_long·χ)]^{α}`

## 适用边界 / 坑

- Osborn 式假设**均匀磁化 + 线性磁导率**，MRF 颗粒（羰基铁）在 B>0.3T 已进入**磁饱和**，此时 N 的"稀释"意义变弱 → 高 B 段 τ_y 饱和形 `B^β/(c+B^β)` 要另挂（Ginder-Davis 1994）
- 长方体没有 Osborn 闭式，只能用近似或数值 → 立方体 N=1/3 精确、长方体需查表（Pillai 1978 类）
- AR₁>>1 且 AR₂≈1（长雪茄）时 N_a→1、N_b=N_c→0，数值稳定性注意（e→1 时 ln 项发散趋势，Osborn 式本身稳定但实现时注意）

## RAG 消费提示
- 检索词建议："椭球 退磁因子 N_a AR"、"Osborn 1945 demagnetizing factor"
- 本篇与 `constitutive_models.md` 的 CCK 段 + `donev2004_packing.md` 的 φ_max 段，构成 Proposer "几何→τ_y" 三通道（退磁+堆积+棱角）的前两通道