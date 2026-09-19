"""表达式可视化：最优公式的预览图与表达式树图。

角色归属
--------
收尾分析（analysis）阶段的**可选输出**：只画图，不改表达式、不参与判分。

降级策略
--------
两项依赖都是可选的，缺任何一个都只告警、不中断实验（产物是"给人看的"，
不该让整条收尾流程失败）：

* 预览图需要 LaTeX 工具链（``sp.preview``）；
* 表达式树图需要 ``graphviz``——它在顶层可选导入，未安装时 ``Source`` 为 ``None``
  （历史上是硬导入，导致缺 graphviz 的环境连 ``pipeline`` 都无法 import）。
"""
from __future__ import annotations

import sympy as sp

# graphviz 为可选依赖：未安装时 Source=None，表达式树渲染会跳过（已有 try/except 兜底）。
# 此前顶层硬导入会使 pipeline.py（依赖本模块）在缺 graphviz 环境下整体无法 import。
try:
    from graphviz import Source
except ImportError:  # pragma: no cover - 缺 graphviz 时降级
    Source = None


def graphviz_available() -> bool:
    """表达式树渲染是否可用（供调用方决定要不要提示用户装 graphviz）。"""
    return Source is not None


def safe_preview(expr, filename: str) -> None:
    """容错地保存表达式预览图；缺 latex/工具链时仅告警，不中断流程。"""
    try:
        sp.preview(expr, output='png', filename=filename, viewer='file')
    except Exception as e:
        print(f"[WARN] 保存表达式图片失败（{filename}）: {e}")


def render_expr_trees(results_root: str, expr, pruned_expr) -> None:
    """表达式树可视化（依赖 graphviz，缺失或失败时仅告警，不中断流程）。

    ``pruned_expr`` 为 ``None`` 表示**本次没有剪枝后的表达式**（没真剪掉项，或剪枝
    失败/未执行）：此时只画原始树，不再画一张与它相同的"剪枝后"树。
    """
    for name, e in (("original_expr_tree", expr), ("pruned_expr_tree", pruned_expr)):
        if e is None:
            # 旧实现在这里 sp.dotprint(None) 会抛异常，再被下面的 except 当成
            # "缺 graphviz" 报出来——噪声且误导（真实原因是本次没有剪枝结果）。
            print(f"[INFO] 跳过表达式树图（{name}）：本次没有剪枝后的表达式")
            continue
        if Source is None:
            print(f"[WARN] 跳过表达式树图（{name}）：未安装 graphviz")
            continue
        try:
            src = Source(sp.dotprint(e))
            src.render(f'{results_root}/{name}', view=True)
        except Exception as ex:
            print(f"[WARN] 生成表达式树图失败（{name}，可能缺少 graphviz 环境）: {ex}")
