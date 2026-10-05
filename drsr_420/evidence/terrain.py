"""历史采样架构地形：把"已经试过哪些函数结构"从 LLM 的自述收回到代码。

角色
----
评测层的**确定性诊断计算**（与 :mod:`drsr_420.evidence.facts` 同类），但对象
不是数据集而是**本实验已经评估过的方程**。它回答三个只有代码能回答的问题：

* 当前最好分由哪个架构取得、此后多少个样本没再改进；
* 已评估样本里有多少是**同一架构的重新参数化**（换个记号/平移不算新架构）；
* 当前架构的"删一项"邻域里，哪些**从未被评估过**。

为什么需要它
------------
实测 ``MRFCompress-Cuboid_20260926-110809``（同数据第二次运行）的 87 个有评分样本：
52 个（60%）是**同一个完整二阶响应面**（两个变量的平方项 + 交叉项），彼此只差记号
与参数化（``(u,v)`` 平移、``(log u, log v)``、``(λ-1)/(λ+1)``…）；而**删掉一个平方项**
的 5 参数非对称形式一次都没被提出——它恰是该实验里分数最高的干净骨架。采样器把
"换个记号"当成了探索，整期锁死在同一架构。（post-mortem 时用文本模式匹配的粗估计是
45 个，偏低；漏项的来源正是下面"和式别名"那条——多行右值只截到第一行。）

注入"已试过哪些架构 + 最小的未试邻域"这类机器事实，比只暴露罚分更准：罚分只说明
当前解不好，不说明该往哪个结构走。

光说"没试过"还不够（2026-09-26 实测补充）
-----------------------------------------
``MRFCompress-Cuboid`` 的同数据 A/B（``110809`` → ``151008``）显示：只列"未试邻域"
时，点名的 term set **一次都没被采纳**（88 个样本里 0 个用 ``drop λ12²`` 的非对称二次），
因为模型把"结构性事实"读成了"别回那个家族"。同时最优分从 NMSE 2.1e-4 退化到 8.4e-3。
故本模块补两件事（都由代码算出、固定种子可复现）：

* :func:`measure_term_set` —— 对每个未试邻域用**评估器同口径**（同 bounds / 多起点
  least_squares / 同残差清洗）真拟合一遍，把实测 NMSE 与该参数化的体检罚分写进提示
  （"这个 term set 从没评估过；确定性拟合 NMSE=2.65e-3，你当前最好干净解 8.4e-3"）。
  无法机械还原的标签显式降级，绝不静默当成"没试过"；
* 分数分解 —— 提示里给出"最好干净解（罚分=0）MSE=…"与"全局最好拟合 MSE=… 但罚分=…
  → 分被吃掉"的对照，避免模型把"拟合 MSE 0.197 + 罚分 36.06"当成胜利（本轮 order 34
  就是这种样本，观测到的"低 MSE 高罚分 ↔ 高 MSE 零罚分"两模态振荡即源于此）。

2026-09-28 补充：闸门被抖动锁死 + 加项方向缺失 + 归并标签整族无实测
------------------------------------------------------------------
多种子协议（6 跑，``ab-terrain-{only,measured}``）暴露本模块三处会让处置**在最需要它的
时候静默失效**的设计：

* **"自最优以来"用错了最优**。``max(records, key=score)`` 会让 refit 抖动（同一模型
  重新拟合出的 1e-10 级差异）成为全局最大，把 ``best_order`` 顶到样本前沿。实测
  T2-s33 撞地板后 order 49/62/65/73 的 4 次"刷新"累计只改善 8.6e-10，于是块内自报的
  停滞被重置成 ``samples since that best: 8``（若锚在真正的最后进步应为 33）——闸门
  恰好在该注入时关闭。现在最优由 :func:`significant_best` 定：改善必须超过
  :data:`SIGNIFICANT_IMPROVEMENT_REL` / :data:`SIGNIFICANT_IMPROVEMENT_ABS`。
* **未试邻域只枚举"删项"**。实测唯一成功的逃逸动作是**加项**（非对称二次 + λ23³，
  T2-s22 的 order 79 → MSE 0.271565），而只列删项时该方向既不在列表里、也不可能被
  点名——同一臂的 T2-s33 全程 85 个样本里 ``λ23³`` 一次都没出现，卡在地板上。
  现在 :func:`addition_candidates` 从目标自己的项**升一阶**产生候选，与删项同口径
  实测、同段渲染（``add λ23^3 -> …``）。为此结构指纹把三次单项式**单列标签**
  （``λ23^3`` / ``λ12^2·λ23`` / ``λ12·λ23^2`` …），不再并进 ``higher``——否则
  "这个 term set 从未试过"既认不出来、也无法从标签还原代表元去实测。
* **归并标签（``higher`` / ``power(...)``）的邻域一条实测都给不出**。标签把多个形状并成
  一个，代表元无从唯一还原 → 整族降级成"NOT measured here"。实测 T2-s33 的采样器整期
  落在参数指数幂律分支：20 个 term set 里 ``power(...)`` 族占样本约 1/3，前 6 大族的
  未试邻域覆盖率只有 21/24，该臂的处置**恰好在自己最需要的那一支上静默关闭**。
  现在 :func:`term_texts` 从**目标方程自己的项文本**取出该项（别名按原式展开、参数保留
  ``params[k]``），交给 :func:`representative_from_text` 的 AST 白名单求值器——**不是猜，
  是读**；越界（白名单外的构造）仍降级成"未测量"并写明原因。覆盖率 21/24 → **24/24**
  （T2-s22 同样 19/25 → 25/25，无回归）。

2026-09-28（午后）补充：一阶邻域点不到"改两项"，镜像项被漏掉
--------------------------------------------------------------
第一个带全部修复的 run（``20260928-141337``）从 order 42（0.271565）卡到 order 83 才
改善到 **0.257394**（当时最好的干净解，已写进已知解清单）。它的逃逸动作
``-λ12·λ23 +λ12²`` 距离当时最优**两项**——而一阶邻域（删项/加项）里**没有一个**比
当时最好更好，模型收到的信息等于"你已到顶"。用同口径枚举"删一加一"对角邻域后，该族
另有 4 个干净解优于当时最好（最好的 ``-const +λ12³`` 实测 −0.002248，已接近那次实验的
早停目标 −0.00201208）。另有一处更便宜：加项生成器漏**镜像项**（目标里有 ``λ23³`` 却
没有 ``λ12³`` 时不生成 ``λ12³``），而 ``+λ12³`` 实测 −0.000058——旧规则因为目标里没有
``λ12²`` 从不生成它。故本模块补三件事：

* 加项候选加**镜像规则**（``(a,b) → (b,a)``），并**候选全测**、渲染前按**实测分**取前
  几名（旧的"按次数排序只测前 4 个"会把有用的那项挤掉）；
* 新增**对角邻域**（同时删一 + 加一），与删/加项同口径真拟合；一阶全都不如当前最好时
  它往往是唯一还有改进的地方；
* 分数分解的 "best" 与地形块标题的 order 对齐（同容差档位时按显著最优取），消掉同一段
  文字里出现两个 ``sample_order`` 的双口径（实测 42 vs 47 差 1e-11）。

⚠️ 闭环约束**不得**写成"禁止该维引入 |指数|>k 的幂律"：本轮达到干净地板 NMSE 2.65e-3
的形式 ``a*(λ23+p·λ12)^b+d·λ12^e+f`` 本身带参数指数，一刀切会把最优干净形式一起禁掉
（且本仓已记录该手段不可泛化）。约束只能落在**可验证的量**上：输出跨度 / 局部斜率 /
系数比 / 实测 NMSE——即本模块与 :mod:`drsr_420.equations.pathology` 给出的东西。

结论必须由代码给出
------------------
"从未评估过"是对**已评估集合**的陈述，必须逐样本核对。本模块对每个样本做**结构
指纹**（表达式拆成加性项，归到"输入变量的一维变换的幂次"），再对指纹集合做差集。
解析不出的样本计入 ``n_unparsed`` 并显式披露，绝不静默当成"没试过"。

指纹口径（刻意粗、但对记号不变）
--------------------------------
* 先把函数体里的**别名**解析成三类原子：``U0``/``U1``（只依赖某一个输入变量的一维
  变换，含平移 ``λ-1``、缩放、``log``、``(λ-1)/(λ+1)``）与 ``W``（同时依赖两者）；
  含两个以上含变量项的**和式**别名（``sigma = p0 + p1*λ12 + …`` 再 ``return sigma``）
  保留文本结构、不压成一个原子——否则 6 个项会被记成 1 个；
* 每个加性项按"幂次对"归到标签：``(1,0)``→线性、``(2,0)``→平方、``(1,1)``→交叉、
  其余高阶→``higher``；指数是**参数**（``x**params[1]``）或 ``np.power(x, p)`` 的项
  另记 ``power(...)``（那是幂律，不是多项式）；
* 于是"平移/缩放/取对数后的二次型"与"原坐标二次型"落进**同一指纹**——这正是要抓的
  东西（换个记号不是换架构）；而"只对乘积 ``λ12·λ23`` 做二次"落进另一个指纹
  （那是一维山脊模型，不是双变量响应面）。

范围
----
只支持**两个自变量**（本项目当前的多变量实验都落在二元）：更多自变量时无法用
"平方项/交叉项"这一对概念刻画邻域，函数直接返回 ``None``/空串（调用方不注入）。

代表元拟合的局限（必须在提示里说清）
------------------------------------
未试邻域是从**标签**还原的（标签对记号不变，故还原必然丢掉"平移/取对数"这一层）。
多项式族无妨：仿射重参数化（``u = λ12-1``）与恒等记号张成同一线性空间，最小二乘最优
NMSE 精确相同；但 ``log`` 记号不是同一空间，且 ``higher`` / ``power(λ12,λ23)`` 这类标签
把多个不同形状归并在一起、无法唯一还原——这些一律**不猜**，显式记为未测量。

分层
----
依赖 core（:mod:`~drsr_420.equations.pathology`、:mod:`~drsr_420.equations.records`）与
同层的 :mod:`~drsr_420.execution.problems` / :mod:`~drsr_420.evidence.facts`
（拟合口径必须与打分完全一致，另起一套就不可比）。供 agents 层的采样提示注入与残差
分析提示共用。
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from drsr_420.equations.pathology import dynamic_range_check
from drsr_420.equations.records import load_sample_records, score_breakdown
from drsr_420.evidence.facts import BASELINE_SEED, load_facts
from drsr_420.execution.problems import evaluate
from drsr_420.equations.text_algebra import (   # noqa: F401  （转发：对外契约不变）
    architecture_fingerprint,
    atom_meanings,
    features_from_equation,
    representative_from_text,
    term_texts,
)
from drsr_420.evidence.neighborhood import (   # noqa: F401
    FIT_SEED,
    measure_term_set,
    template_from_terms,
)

#: 判定"已触底"所需的最少"自最优以来的样本数"。更短时该判断证据不足，不注入。
MIN_STAGNANT_SAMPLES = 5

#: 判定"已试过足够多架构"所需的最少已解析样本数。
MIN_PARSED_SAMPLES = 10

#: 汇总块最多列出的"删一项"未试邻域个数（按被删项由高阶到低阶排序）。
#: 每个邻域都要真拟合一遍，故这也是"每份地形最多几次 least_squares"的上界。
MAX_UNTRY_DELETIONS = 3

#: 汇总块最多列出的"加一项"未试邻域个数（按被加项由高阶到低阶排序）。
#: 加项与删项是**两个方向**，不能只做一个：实测本仓唯一成功的逃逸动作就是**加项**
#: （``{const, λ12, λ12·λ23, λ23, λ23²}`` 再加 ``λ23³``，T2-s22 的 order 79 →
#: MSE 0.271565），而只枚举删项时该方向既不在列表里、也不可能被点名（同一臂的
#: T2-s33 全程 85 个样本里 ``λ23³`` 一次都没出现，卡在 0.867473 的地板上）。
#: 取 4 而不是 3：二次目标"升一阶"的全部候选正好是 4 个三次单项式
#: （``λ12³`` / ``λ12²·λ23`` / ``λ12·λ23²`` / ``λ23³``），取 3 会按标签字典序
#: 砍掉其中一个——那正是"按实现细节而不是按结构决定看哪个方向"的老毛病。
MAX_UNTRY_ADDITIONS = 4

#: 汇总块最多列出的"删一 + 加一"（**同时改两项**）未试邻域个数，按实测分取前几名。
#: 为什么需要这一族：一阶邻域给不出比当前最好更优的候选时，模型收到的信息等于"你已到顶"
#: ——实测 ``20260928-141337`` 正是这样从 order 42 卡到 83（41 个样本），而它唯一有效的
#: 逃逸动作 ``-λ12·λ23 +λ12²``（order 83 → 0.257394）**距离当时的最优两项**，一阶列表
#: 结构上点不到；用同口径枚举对角邻域后，该族另有 **4 个干净解**优于当时最好——最好的
#: ``-const +λ12³`` 实测 **−0.002248**（MSE 0.002248、罚分 0），已接近该实验的早停目标
#: （−0.00201208）。
MAX_UNTRY_DIAGONALS = 3

#: "实质改善"的容差（相对 / 绝对，取二者较大者）。
#: refit 微抖动不是进步：实测 T2-s33（``20260928-092926``）撞地板后 order 49/62/65/73
#: 的 4 次"刷新"累计只改善 **8.6e-10**（相邻两点差 8.2e-10 / 2.1e-11 / 1.3e-11 /
#: 2.5e-12），却把"最优"一路顶到样本前沿，于是地形块**自报的**停滞被重置成
#: ``samples since that best: 8``（真值应为 33）——闸门恰好在该注入时关闭。
#: 容差取 1e-6，与渲染精度（``round(..., 6)``）同量级：显示上完全相同的两个分数
#: 不构成一次"刷新"。
SIGNIFICANT_IMPROVEMENT_REL = 1e-6
SIGNIFICANT_IMPROVEMENT_ABS = 1e-6


#: 结构指纹的原子记号：U0/U1 = 只依赖某个输入变量的一维变换，W = 同时依赖两者，
#: P = 不含输入变量的量（参数/常数）。
def family_of(terms: Sequence[str], names: Sequence[str]) -> str:
    """粗粒度族标签：用于"同一架构被反复重试"的计数。

    判序刻意先看**平方项的个数**再看交叉项：只保留一个平方项的 5 参数非对称形式
    （本实验里分数最高的干净骨架）必须与"只有一个交叉项"的乘积模型分开计——前者
    是"二阶不足以支撑两个平方项"的候选，后者是一维山脊模型。
    """
    tags = set(terms)
    n0, n1 = names
    square0, square1 = f"{n0}^2", f"{n1}^2"
    cross = f"{n0}*{n1}"
    if {square0, square1, cross} <= tags:
        return "second_order_surface"
    if square0 in tags or square1 in tags:
        return "partial_second_order"
    if cross in tags:
        return "coupled_not_second_order"
    if "higher" in tags:
        return "higher_order"
    if any(tag.startswith("power(") for tag in tags):
        return "power_law"
    return "other"


def _family_phrase(family: str) -> str:
    return {
        "second_order_surface": ("a second-order surface with BOTH squared terms and "
                                 "the cross term"),
        "partial_second_order": ("a second-order surface that keeps only ONE of the two "
                                 "squared terms"),
        "coupled_not_second_order": "a coupled but not full second-order form",
        "higher_order": "a purely higher-order form",
        "power_law": "a power-law form (at least one parametric exponent)",
        "other": "a form with no squared or coupled terms",
    }.get(family, family)


def _cubic_labels(names: Sequence[str]) -> set[str]:
    """三次单项式标签集合（供"高阶项"排序与加项候选共用）。"""
    n0, n1 = names
    return {f"{n0}^3", f"{n1}^3", f"{n0}^2*{n1}", f"{n0}*{n1}^2"}


def _term_rank(label: str, names: Sequence[str]) -> int:
    """删项/加项建议的优先级：先动高阶/结构性项，最后才动线性项与常数项。"""
    if label in ("higher",) or label.startswith("power(") or label in _cubic_labels(names):
        return 3
    if label.endswith("^2") or label == f"{names[0]}*{names[1]}":
        return 2
    if label == "const":
        return 0
    return 1


def _monomial_degree(label: str, names: Sequence[str]) -> tuple[int, int] | None:
    """标签对应的 ``(λ12 的次数, λ23 的次数)``；不是单项式标签时返回 ``None``。

    ``higher`` / ``power(...)`` 本身归并了多个形状，"升一阶"无从谈起 → 不产生加项候选。
    """
    n0, n1 = names
    return {
        "const": (0, 0),
        n0: (1, 0), n1: (0, 1),
        f"{n0}^2": (2, 0), f"{n1}^2": (0, 2), f"{n0}*{n1}": (1, 1),
        f"{n0}^3": (3, 0), f"{n1}^3": (0, 3),
        f"{n0}^2*{n1}": (2, 1), f"{n0}*{n1}^2": (1, 2),
    }.get(label)


def _monomial_label(degree: tuple[int, int], names: Sequence[str]) -> str | None:
    """次数对 → 标签；三次以上没有独立标签，返回 ``None``（不猜）。"""
    n0, n1 = names
    return {
        (0, 0): "const",
        (1, 0): n0, (0, 1): n1,
        (2, 0): f"{n0}^2", (0, 2): f"{n1}^2", (1, 1): f"{n0}*{n1}",
        (3, 0): f"{n0}^3", (0, 3): f"{n1}^3",
        (2, 1): f"{n0}^2*{n1}", (1, 2): f"{n0}*{n1}^2",
    }.get(degree)


def addition_candidates(terms: Sequence[str], names: Sequence[str]) -> list[str]:
    """目标 term set 的"升一阶"加项候选，按次数降序 + 标签排序。

    两条规则，都在"目标自己的项"上长候选（不枚举任意高阶单项式）：

    * **升一阶**：``λ23²`` → ``λ23³``、``λ12`` → ``λ12²``、``λ12·λ23`` →
      ``λ12²·λ23`` / ``λ12·λ23²``；
    * **镜像**：把已有的项**交换两个自变量的指数**（``(a,b) → (b,a)``，``λ23³`` → ``λ12³``）。

    镜像规则是 2026-09-28 实测补的：``20260928-141337`` 的 order-42 族
    ``{const, λ12, λ12·λ23, λ23, λ23², λ23³}`` 在 λ23 上一路升到三次、在 λ12 上只到一次，
    而"加 ``λ12³``"的实测分是 **−0.000058（罚分 0）**——已越过那次实验的早停目标
    （``score ≥ −0.00201208``）；旧的"只升一阶"规则因为目标里没有 ``λ12²``，**从不生成
    ``λ12³``**，模块当时给出的三个加项全是罚分陷阱（−29.0 / −293.9 / −228.8）。漏的不是
    "高阶"本身，而是**两个自变量上的不对称**，镜像恰好补这一维。

    常数项在次数上无从升、镜像又是它自己，故不产生候选。排序按次数降序（高次优先）→ 与
    本仓"逃逸动作 = 给最高次项升阶"的实测一致；渲染前会再按**实测分**重排
    （见 :func:`_best_measured`），故这里的顺序只在预算截断时起作用。
    """
    found: list[str] = []

    def push(label: str | None) -> None:
        if label and label not in terms and label not in found:
            found.append(label)

    for label in terms:
        degree = _monomial_degree(label, names)
        if degree is None or degree == (0, 0):
            continue
        for raised in ((degree[0] + 1, degree[1]), (degree[0], degree[1] + 1)):
            push(_monomial_label(raised, names))
        push(_monomial_label((degree[1], degree[0]), names))              # 镜像
    return sorted(found, key=lambda label: (-sum(_monomial_degree(label, names)), label))


def _best_measured(items: list, limit: int) -> list:
    """按**实测分**取前 ``limit`` 个；没测出分数的排最后，同分保持原有顺序（稳定排序）。

    为什么按实测分而不是按次数或字典序：候选"次数高"不等于"更值得试"——实测
    ``20260928-141337`` 里被机械排到前面的三个三次项全是罚分陷阱（−29.0 / −293.9 /
    −228.8），而漏掉的那个三次项实测 −0.000058、已越过早停目标。分数由评估器按同一口径
    算出，比任何先验排序都更接近"这一步值不值"。
    """
    def key(item):
        score = (item.get("measurement") or {}).get("score")
        return (score is None, -(score if score is not None else 0.0))

    return sorted(items, key=key)[:limit]




# ── 输入装载（两条注入通道共用） ────────────────────────────────
def load_terrain_inputs(results_root: str | None) -> dict:
    """读地形计算所需的两个外部输入：事实表与样本记录（缺失即降级为空）。

    * ``facts``：``data_facts.json``（提供训练数据表，未试邻域的实测 NMSE 要用它复现
      评估器口径）；
    * ``records``：``samples/*.json``（提供 ``mse``/``penalty``/``score`` 三元组，
      分数分解要用）。

    任一缺失时下游只输出纯结构事实（这正是旧行为），注入本身绝不能让主流程失败。
    """
    root = results_root or "."
    try:
        facts = load_facts(root)
    except Exception:
        facts = {}
    try:
        records = load_sample_records(root)
    except Exception:
        records = []
    return {"facts": facts if isinstance(facts, dict) else {}, "records": records}


def with_score_breakdown(terrain: dict, records: Sequence[dict]) -> dict:
    """返回带上"分数是怎么来的"分解的**地形副本**（不就地改：地形本身会被整轮缓存复用）。

    单独一步的原因：结构地形（含未试邻域的拟合）只随**已试集合**变化、可以整轮缓存；
    而分数分解随每个新样本变化，必须每轮刷新。
    """
    merged = dict(terrain or {})
    breakdown = score_breakdown(list(records or []))
    # 两套"最优"口径的对齐**只在抖动级别**做：原始 argmax 与"显著最优"若落在同一个分数档
    # （差异在容差内），就按显著最优那个 order 渲染——否则同一段文字里会出现两个
    # sample_order（实测 ``20260928-141337`` 的块：标题说 42、分解行说 47，差 1e-11）。
    # 差异是真的时（典型情形：分数更高但结构解析不出、因而进不了显著最优的样本）**保留**
    # 原始 argmax——"存在一个分数更好但读不出结构的样本"是必须披露的信息，不能抹掉。
    significant_order = merged.get("best_order")
    significant_score = merged.get("best_score")
    best = breakdown.get("best") or {}
    raw_score = best.get("score")
    if (significant_order is not None and significant_score is not None
            and raw_score is not None
            and abs(float(raw_score) - float(significant_score))
            <= significant_improvement_tolerance(float(significant_score))):
        breakdown = score_breakdown(list(records or []), best_order=significant_order)
    merged["breakdown"] = breakdown
    return merged


def significant_improvement_tolerance(reference_score: float) -> float:
    """判定"实质改善"的容差：``max(绝对, 相对 × |参考分|)``。

    取二者较大者：本仓实测的分数跨 0.27~10³，纯绝对容差在小分上偏松、纯相对容差
    在 0 附近恒为 0。与渲染精度（6 位小数）同量级 ⇒ 显示上分不出差别的两个分数
    不会被当成一次"刷新"。
    """
    try:
        magnitude = abs(float(reference_score))
    except (TypeError, ValueError):
        magnitude = 0.0
    return max(SIGNIFICANT_IMPROVEMENT_ABS, SIGNIFICANT_IMPROVEMENT_REL * magnitude)


def significant_best(records: Sequence[dict]) -> dict:
    """按 ``order`` 走一遍，只在**实质**改善时推进"最优"（不做全局 ``max``）。

    为什么不能用 ``max(records, key=score)``：refit 抖动会让**同一个模型**以 1e-10 的
    优势成为全局最大，于是 ``best_order`` 被顶到样本前沿、"自最优以来" 归零——恰好在
    真正卡住时把地形块关掉（实测见 :data:`SIGNIFICANT_IMPROVEMENT_REL` 的文档）。
    """
    ordered = sorted(records, key=lambda r: (r["order"], -r["score"]))
    best = ordered[0]
    for record in ordered[1:]:
        if record["score"] > best["score"] + significant_improvement_tolerance(best["score"]):
            best = record
    return best


def sampling_terrain(entries: Sequence[dict], features: Sequence[str],
                     *, target: str | None = None, facts=None) -> dict:
    """汇总已评估样本的架构地形。

    Args:
        entries: 经验条目（``experiences.json`` 的元素），需要 ``equation``、
            ``score``（无数字分值的条目被跳过）与可选的 ``sample_order``。
        features: 自变量名（长度必须为 2，否则返回空地形）。
        target: 要刻画其"删一项邻域"的方程文本；``None`` 时用**分数最高**的样本
            （采样通道用默认，残差通道传当前被分析的方程）。
        facts: ``data_facts.json`` 的内容（可选）。给了就用评估器同口径把每个未试
            邻域真拟合一遍并把实测 NMSE 写进地形（见 :func:`measure_term_set`）；
            没给/表不可用则只列 term set、附上显式降级原因。

    Returns:
        dict（结构见 :func:`render_terrain`；缺数据时各项为空/0，渲染器自行决定是否注入）。
    """
    names = list(features or [])
    empty = {"ok": False, "n_scored": 0, "n_parsed": 0, "n_unparsed": 0,
             "target_terms": None, "untried_deletions": [], "untried_additions": [],
             "untried_diagonals": []}
    if len(names) != 2:
        return empty

    records = []
    for index, entry in enumerate(entries or []):
        if not isinstance(entry, dict):
            continue
        score = entry.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        order = entry.get("sample_order")
        order = int(order) if isinstance(order, (int, float)) and not isinstance(order, bool) \
            else index + 1
        records.append({"score": float(score), "order": order,
                        "equation": entry.get("equation"),
                        "terms": architecture_fingerprint(entry.get("equation"), names)})
    parsed = [r for r in records if r["terms"]]
    if not parsed:
        return empty

    best = significant_best(parsed)
    current_order = max(r["order"] for r in records)
    since = [r for r in parsed if r["order"] > best["order"]]
    best_family = family_of(best["terms"], names)

    target_terms = tuple(architecture_fingerprint(target, names)) if target else best["terms"]
    target_is_best = bool(target_terms) and tuple(target_terms) == tuple(best["terms"])
    if not target_terms:
        target_terms = best["terms"]
        target_is_best = True
    target_terms = tuple(target_terms)
    target_family = family_of(target_terms, names)

    tried = {r["terms"] for r in parsed}
    # 未试邻域的代表元还原以**目标方程自己的项文本**为准（缺陷 2）：``higher`` /
    # ``power(...)`` 归并了多个形状，从标签猜是不诚实的，但它们的项在方程里写得很清楚。
    target_texts = term_texts(target if target else best.get("equation"), names)
    deletions = []
    for label in sorted(target_terms, key=lambda x: (-_term_rank(x, names), x)):
        candidate = tuple(sorted(t for t in target_terms if t != label))
        if candidate in tried:
            continue
        # 每个未试邻域真拟合一遍：光说"没试过"实测会被读成"别回那个家族"，
        # 配上可复现的 NMSE 才成为"可验证的改进方向"（见模块 docstring）。
        deletions.append({"dropped": label, "terms": list(candidate),
                          "measurement": measure_term_set(candidate, names, facts,
                                                          texts=target_texts)})
        if len(deletions) >= MAX_UNTRY_DELETIONS:
            break

    # 加项：**候选全测**、再按**实测分**取前几名渲染。先前是"按次数排序后只测前 4 个"，
    # 会把有用的那项挤掉：实测 ``20260928-141337`` 的 order-42 族，被机械排在前面的三个
    # 加项全是罚分陷阱（−29.0 / −293.9 / −228.8），而漏掉的 ``+λ12³`` 实测 −0.000058。
    additions = []
    for label in addition_candidates(target_terms, names):
        candidate = tuple(sorted(list(target_terms) + [label]))
        if candidate in tried:
            continue
        # 与删项同口径：加项邻域也要真拟合一遍，"加这项会变成什么分数"必须可验证。
        additions.append({"added": label, "terms": list(candidate),
                          "measurement": measure_term_set(candidate, names, facts,
                                                          texts=target_texts)})
    additions = _best_measured(additions, MAX_UNTRY_ADDITIONS)

    # 对角（**同时删一项 + 加一项**）：一阶邻域全都不如当前最好时，模型收到的信息等于
    # "你已到顶"——实测 ``20260928-141337`` 就这样从 order 42 卡到 83，而它唯一有效的
    # 逃逸动作 ``-λ12·λ23 +λ12²`` 距离当时最优**两项**，一阶列表结构上点不到。
    diagonals = []
    for label in sorted(target_terms, key=lambda x: (-_term_rank(x, names), x)):
        for added in addition_candidates(target_terms, names):
            candidate = tuple(sorted([t for t in target_terms if t != label] + [added]))
            if len(candidate) != len(target_terms) or candidate in tried:
                continue
            diagonals.append({"dropped": label, "added": added, "terms": list(candidate),
                              "measurement": measure_term_set(candidate, names, facts,
                                                              texts=target_texts)})
    diagonals = _best_measured(diagonals, MAX_UNTRY_DIAGONALS)

    return {
        "ok": True,
        "features": names,
        "n_scored": len(records),
        "n_parsed": len(parsed),
        "n_unparsed": len(records) - len(parsed),
        "best_score": round(best["score"], 6),
        "best_order": best["order"],
        "current_order": current_order,
        "stagnant_samples": current_order - best["order"],
        "since_samples": len(since),
        "since_same_family": sum(1 for r in since if family_of(r["terms"], names) == best_family),
        "since_term_sets": len({r["terms"] for r in since}),
        "best_terms": list(best["terms"]),
        "best_family": best_family,
        "best_family_count": sum(1 for r in parsed
                                 if family_of(r["terms"], names) == best_family),
        "best_family_variants": len({r["terms"] for r in parsed
                                     if family_of(r["terms"], names) == best_family}),
        "target_terms": list(target_terms),
        "target_is_best": target_is_best,
        "target_family": target_family,
        "target_family_count": sum(1 for r in parsed
                                   if family_of(r["terms"], names) == target_family),
        "target_exact_count": sum(1 for r in parsed if r["terms"] == target_terms),
        "untried_deletions": deletions,
        "untried_additions": additions,
        "untried_diagonals": diagonals,
    }


# ── 渲染 ────────────────────────────────────────────────────
def _fmt_number(value) -> str:
    """4 位有效数字；``None``/非数值 → ``n/a``（跨量级的数字必须留有效位数，不取整）。"""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return "n/a"


def _render_measurement(measurement: dict, best_score) -> list[str]:
    """未试邻域的"实测口径 + 实测分数"（或显式降级原因）。

    为什么比较的是**分数**而不是 NMSE：NMSE 只反映拟合质量，"拟合得好但被罚分吃掉"
    与"拟合一般却干净"在评分层是同一条轴上的两个解。实测 110809 的完整二次型拟合
    NMSE 2.11e-4 却带 0.6877 罚分（score −1.113），而"删 λ12²"的非对称二次测下来
    拟合 NMSE 4.31e-4、罚分 0（score −0.867）——**后者分数更好**；只报 NMSE 会把
    这个可验证的改进方向说反。

    只说"没试过"实测会被读成"别回那个家族"（110809→151008 的 88 个样本里被点名的
    term set 0 次采纳），配上"它比你现在最好的分还高"才成为可执行的方向。
    """
    if not measurement:
        return []
    if measurement.get("nmse") is None:
        reason = measurement.get("reason") or "no NMSE was produced"
        return [f"      NOT measured here ({reason}) — still never evaluated"]
    head = (f"      measured on THIS run's training data with the same evaluator "
            f"(deterministic fit, seed {FIT_SEED}, representative parameterization "
            f"`{measurement.get('template')}` using {measurement.get('n_params')} parameters): "
            f"fit MSE {_fmt_number(measurement.get('mse'))} "
            f"(NMSE {_fmt_number(measurement['nmse'])})")
    if measurement.get("flagged"):
        hits = "; ".join(measurement.get("criteria") or []) or "dynamic-range pathology"
        return [head + f", BUT penalty {_fmt_number(measurement.get('penalty'))} ({hits})"
                f" -> implied score {_fmt_number(measurement.get('score'))}",
                "      -> that fit is a localized device: its low NMSE would be paid for by "
                "the penalty, so do not chase the fit quality alone"]
    lines = [head + ", penalty 0 (clean)"]
    score = measurement.get("score")
    if score is not None and best_score is not None:
        if score > best_score:
            lines.append(f"      -> implied score {_fmt_number(score)} WOULD BEAT your current "
                         f"best score ({_fmt_number(best_score)}): an unexplored term set with "
                         "a verifiable improvement on the same training data")
        else:
            lines.append(f"      -> implied score {_fmt_number(score)} is below your current "
                         f"best score ({_fmt_number(best_score)}): still unexplored, but it is "
                         "not the missing step")
    return lines


def _penalty_share(row: dict) -> float:
    """该样本的分数里罚分占的比例（``|score| = mse + penalty``）；算不出时返回 0。"""
    total = (row.get("mse") or 0.0) + (row.get("penalty") or 0.0)
    return (row.get("penalty") or 0.0) / total if total else 0.0


def _render_breakdown(terrain: dict) -> list[str]:
    """分数分解段：把"拟合 MSE"与"体检罚分"拆开。

    实测 20260926-151008 的 order 34 拟合 MSE 0.197 却带 36.06 罚分，模型会把 0.197
    当胜利——观测到的"低 MSE 高罚分 ↔ 高 MSE 零罚分"两模态振荡正是这么来的。
    故这里点名两个"看起来很好但分被吃掉"的样本：带罚分里**分数最高**的那个（干净解
    最强的对手）与**拟合 MSE 最低**的那个（"拟合得好"的真正上界）。

    计数只说"**已落盘**的样本记录"：旧实验只留 top-K（``persist_all_samples`` 之前），
    此时记录数远小于头行的 ``scored samples``，两者不混。
    """
    breakdown = terrain.get("breakdown") or {}
    if breakdown.get("n_scored", 0) <= 0:
        return []
    if not breakdown.get("penalty_known"):
        return ["score decomposition unavailable for this run: its sample records do not "
                "carry the pathology penalty separately from the fit (older format, where "
                "the stored MSE already contained the penalty)"]
    best = breakdown.get("best") or {}
    clean = breakdown.get("best_clean")
    penalized = breakdown.get("best_penalized")
    lowest = breakdown.get("best_fit")
    shown = {best.get("sample_order")} | {other.get("sample_order")
                                         for other in (clean, penalized) if other}
    lines = ["how the score is built - every number below was measured by this run's own "
             "evaluator: score = -(fit MSE + pathology penalty), so a high fit quality "
             "with a penalty still loses"]
    lines.append(f"  best score overall: sample_order {best.get('sample_order')}, fit MSE "
                 f"{_fmt_number(best.get('mse'))}, penalty {_fmt_number(best.get('penalty'))}, "
                 f"score {_fmt_number(best.get('score'))}")
    if clean and clean.get("sample_order") != best.get("sample_order"):
        lines.append(f"  best sample with ZERO penalty: sample_order {clean.get('sample_order')}, "
                     f"fit MSE {_fmt_number(clean.get('mse'))}, "
                     f"score {_fmt_number(clean.get('score'))}")
    if penalized and penalized.get("sample_order") != best.get("sample_order"):
        tail = (": this is the strongest penalized competitor"
                + (", and its score is worse than the clean one" if clean else ""))
        lines.append(f"  best sample that still carries a penalty: sample_order "
                     f"{penalized.get('sample_order')}, fit MSE "
                     f"{_fmt_number(penalized.get('mse'))}, penalty "
                     f"{_fmt_number(penalized.get('penalty'))}, score "
                     f"{_fmt_number(penalized.get('score'))}{tail}")
    if lowest and lowest.get("sample_order") not in shown:
        lines.append(f"  lowest fit MSE seen: sample_order {lowest.get('sample_order')}, fit MSE "
                     f"{_fmt_number(lowest.get('mse'))} but penalty "
                     f"{_fmt_number(lowest.get('penalty'))} -> score "
                     f"{_fmt_number(lowest.get('score'))} "
                     f"({100.0 * _penalty_share(lowest):.0f}% of the score is penalty), so the "
                     "fit MSE alone overstates how good that solution is")
    summary = (f"  Of the {_fmt_number(breakdown.get('n_penalty_known'))} sample records "
               f"persisted for this run, {_fmt_number(breakdown.get('n_penalized'))} carry a "
               "nonzero penalty")
    if not clean:
        summary += " and NONE is clean"
    summary += (". The penalty is charged by the dynamic-range check on verifiable quantities "
                "(output span vs the data range, local slope inside the box, coefficient "
                "cancellation); a low fit MSE WITH a large penalty is a WORSE solution than a "
                "higher fit MSE with zero penalty.")
    lines.append(summary)
    return lines


def render_terrain(terrain: dict, features: Sequence[str], title: str) -> str:
    """把地形渲染成注入提示词的文本块；证据不足或无法判定时返回空串。

    两个闸门（任一不满足就不注入）：已解析样本数 ≥ :data:`MIN_PARSED_SAMPLES`、
    自最优以来 ≥ :data:`MIN_STAGNANT_SAMPLES` 个样本（"自最优"由
    :func:`significant_best` 定，refit 抖动不算进步）。"该架构已触底"只在有足够
    样本时才说得出口。

    四段内容（各段缺数据时自行省略）：结构事实（已试家族 / 这个 term set 被评估过几次）、
    未试邻域的**实测分数**（**两个方向**：删项与加项），以及分数的分解（拟合 MSE ↔ 体检罚分）。
    """
    if not terrain or not terrain.get("ok"):
        return ""
    if terrain.get("n_parsed", 0) < MIN_PARSED_SAMPLES:
        return ""
    if terrain.get("stagnant_samples", 0) < MIN_STAGNANT_SAMPLES:
        return ""
    parsed = terrain["n_parsed"]
    since = ""
    if terrain["since_samples"]:
        since = (f" ({terrain['since_same_family']} of them in the best sample's term-set "
                 f"family, {terrain['since_term_sets']} distinct term sets in total)")

    lines = [f"scored samples: {terrain['n_scored']} (structurally parsed: {parsed}"
             + (f", unparsed: {terrain['n_unparsed']}" if terrain["n_unparsed"] else "")
             + f") | best score: {terrain['best_score']} (higher is better) at sample_order "
             f"{terrain['best_order']} | samples since that best: "
             f"{terrain['stagnant_samples']}{since}"]

    where = ("the best sample's term set" if terrain["target_is_best"]
             else "the term set of the equation under analysis")
    lines.append(f"{where}: {{{', '.join(terrain['target_terms'])}}}")
    lines.append(
        f"{terrain['target_family_count']} of {parsed} scored samples "
        f"({100.0 * terrain['target_family_count'] / parsed:.0f}%) are in that family - "
        f"{_family_phrase(terrain['target_family'])}; this exact term set has already been "
        f"evaluated {terrain['target_exact_count']} time(s), and a different notation or "
        "parameterization of it is the SAME architecture, not a new one")

    lines.extend(_render_breakdown(terrain))

    deletions = terrain["untried_deletions"]
    additions = terrain.get("untried_additions") or []
    diagonals = terrain.get("untried_diagonals") or []
    if deletions or additions or diagonals:
        lines.append("term sets NEVER evaluated that are ONE term away from it:")
        for item in deletions:
            lines.append(f"  drop {item['dropped']} -> {{{', '.join(item['terms'])}}}")
            lines.extend(_render_measurement(item.get("measurement"),
                                             terrain.get("best_score")))
        for item in additions:
            lines.append(f"  add {item['added']} -> {{{', '.join(item['terms'])}}}")
            lines.extend(_render_measurement(item.get("measurement"),
                                             terrain.get("best_score")))
        if diagonals:
            lines.append("term sets NEVER evaluated that are TWO terms away (ONE dropped AND "
                         "ONE added at the same time) - listed because every one-term step "
                         "above was measured, so the useful move may live here:")
            for item in diagonals:
                lines.append(f"  drop {item['dropped']} + add {item['added']} -> "
                             f"{{{', '.join(item['terms'])}}}")
                lines.extend(_render_measurement(item.get("measurement"),
                                                 terrain.get("best_score")))
        lines.append(
            "Another re-parameterization of the term set above stays at the same score "
            "floor; the deletions, additions and two-term moves listed above are "
            "structurally different and unexplored.")
    else:
        lines.append(
            "Every one-term deletion, every one-term addition AND every two-term (drop one + "
            "add one) combination of that term set has already been evaluated: the next "
            "structural step must change the functional form of an existing term, or leave "
            "this family.")
    return title + "\n".join(lines) + "\n"