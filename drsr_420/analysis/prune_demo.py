"""敏感度剪枝的直观演示：6 个典型/反例场景，打印剪枝前后与统计。

用法::

    python -m drsr_420.analysis.prune_demo

为什么单独一个模块
------------------
演示是"给人看行为"的脚本，不是库代码：它不该混在算法文件里占用阅读路径，
也不该被 import 触发。放在这里后 ``sensitivity_prune.py`` 只剩算法本体，
而 ``python -m drsr_420.analysis.sensitivity_prune`` 仍然可用（转发到本模块）。
"""
from __future__ import annotations

import sympy as sp

from drsr_420.analysis.sensitivity_prune import sensitivity_prune


def main() -> None:
    x, y, z = sp.symbols("x y z", real=True)
    eps = sp.Rational(1, 1000)
    SEP = "═" * 65

    cases = [
        (
            "案例 1 · 多项式：含极小系数项",
            x**3 + 2*x**2 + eps * x + eps**2,
            [x], 0.01,
            "预期：eps·x 和 eps^2 被剪枝，保留 x^3 + 2x^2",
        ),
        (
            "案例 2 · 三角函数：含微小高频项",
            sp.sin(x) + sp.cos(x) + eps * sp.sin(50*x),
            [x], 0.05,
            "预期：eps·sin(50x) 被剪枝，保留 sin(x)+cos(x)",
        ),
        (
            "案例 3 · 多变量：含可忽略交叉项",
            x**2 + y**2 + z**2 + eps * x*y + eps**2 * x*y*z,
            [x, y, z], 0.02,
            "预期：两个交叉项被剪枝，保留 x^2+y^2+z^2",
        ),
        (
            "案例 4 · 乘积：含接近 1 的小扰动因子",
            (1 + eps * x) * (x**2 + y**2) * sp.exp(-eps * y),
            [x, y], 0.05,
            "预期：(1+eps·x) 和 exp(-eps·y) 被剪枝，保留 x^2+y^2",
        ),
        (
            "案例 5 · 嵌套：sin 内部含小扰动",
            sp.sin(x + eps * y) + x**2 + eps**2 * z,
            [x, y, z], 0.01,
            "预期：eps^2·z 被剪枝；sin 内部的 eps·y 视阈值可能被剪",
        ),
        (
            "案例 6 · 负面：所有项均重要，不应剪枝",
            x**2 + 2*x + 1,
            [x], 0.01,
            "预期：无项被剪枝",
        ),
    ]

    for title, expr, syms, thr, hint in cases:
        print(f"\n{SEP}\n  {title}\n{SEP}")
        print(f"  表达式 : {expr}")
        print(f"  阈值   : {thr}   ← {hint}\n")

        pruned, stats = sensitivity_prune(
            expr, syms, threshold=thr, verbose=True,
        )

        print(f"\n  原始 : {expr}")
        print(f"  剪枝 : {pruned}")
        print(f"  节点 : {stats.nodes_pruned} 已剪 / {stats.nodes_visited} 已访问  "
              f"({stats.prune_rate:.0%})\n")


if __name__ == "__main__":
    main()
