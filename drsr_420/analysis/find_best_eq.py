"""收尾分析编排：最佳样本 → 敏感度剪枝与可视化 → 物理解释。

位置
----
``runtime/pipeline.main`` 在实验结束前调用一次 :func:`find_best_eq`（非 Agent，
是普通工具函数——架构图中的"收尾"环节）。

编排（各步骤都拆成了单一职责的模块）
------------------------------------
::

    find_best_eq(results_root)
      ├── find_best_sample()      扫描 samples/*.json 取最高分样本
      ├── prune_and_visualize()   **先剪枝**（解释要覆盖剪枝结果与剪枝过程）
      │     ├── expr_parse.expr_substitution()   骨架字符串 → SymPy 表达式
      │     ├── sensitivity_prune.SensitivityPruner.prune()  敏感度剪枝
      │     ├── prune_report.compare_fits()      剪枝前后在训练数据上的拟合对比
      │     ├── expr_viz.safe_preview() / render_expr_trees()  预览图与树图
      │     └── expr_curves.plot_data_curves()  剪枝前后曲线 + 数据点（可失败，仅告警）
      │     └── 返回剪枝摘要 dict（剪掉了哪些项 + 敏感度 + 拟合变化）
      └── explain.explain_best_sample(pruning=摘要)
            把剪枝前/后表达式、被移除项与拟合数值一起交给解释 LLM → explain.md
            （含参考文献清单：知识库检索结果 + 解释过程中的工具检索结果）

**顺序不能反过来**：explain.md 必须解释"剪枝后的表达式"并讲清"剪掉了哪些项、为什么
合理"，这两件事都要求剪枝结果先算出来。

本模块只做"取样本 + 步骤编排 + 兜底告警"，具体逻辑见上表各自的模块。
"""
import glob
import json
import os
import re

import sympy as sp

from drsr_420.analysis.expr_parse import expr_substitution
from drsr_420.analysis.expr_viz import render_expr_trees, safe_preview
from drsr_420.analysis.explain import explain_best_sample
from drsr_420.analysis.prune_report import compare_fits, format_fit_summary, load_training_data
from drsr_420.analysis.sensitivity_prune import SensitivityPruner


def find_best_sample(results_root: str):
    """扫描 samples 目录，返回分数最高的样本 (score, path, func, params)；无则 None。"""
    best = None
    for p in glob.glob(os.path.join(results_root, "samples", "*_samples_*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            s = d.get("score")
            if s is None:
                continue
            if best is None or s > best[0]:
                best = (s, p, d.get("function", ""), d.get("params"))
        except Exception:
            continue
    return best


def _parse_symbols(func: str) -> tuple[str, list[str]] | None:
    """从样本函数头解析 (因变量名, 自变量名列表)；解析失败返回 None。

    兼容逗号 / 中文逗号 / 空白分隔的自变量列表。
    """
    dependent_match = re.search(r'Dependent:\s*(\w+)', func)
    independent_match = re.search(r'Independents:\s*(.*)', func)
    if not dependent_match or not independent_match:
        return None
    sym_names = [v.strip() for v in re.split(r'[,，\s]+', independent_match.group(1)) if v.strip()]
    if not sym_names:
        return None
    return dependent_match.group(1), sym_names


def prune_and_visualize(results_root: str, func: str, params,
                        threshold: float, sample_range: tuple) -> dict | None:
    """基于敏感度分析剪枝最优公式，保存表达式预览图与表达式树图，返回剪枝摘要。

    返回值是给 ``explain`` 用的剪枝摘要（含剪枝前/后表达式、被移除项及其敏感度、
    剪枝统计、剪枝前后在训练数据上的拟合对比）；解析/剪枝失败时返回 ``None``，
    调用方据此让解释 LLM 知道"本次没有剪枝结果"。
    """
    parsed = _parse_symbols(func)
    if parsed is None:
        print("[WARN] 无法从样本中解析 Dependent/Independents，跳过剪枝。")
        return None
    dependent, sym_names = parsed
    symbols = sp.symbols(sym_names)

    pruner = SensitivityPruner(symbols=symbols, threshold=threshold, sample_range=sample_range)
    expr = expr_substitution(func, params)
    if expr is None:
        print("[WARN] 表达式解析失败，跳过剪枝。")
        return None

    print(f"剪枝前的表达式为 {dependent} =")
    sp.pprint(expr)
    safe_preview(expr, f'{results_root}/expr.png')

    try:
        pruned_expr = pruner.prune(expr, verbose=True)
    except Exception as e:
        print(f"[WARN] 剪枝失败: {e}")
        pruned_expr = None

    if pruned_expr is not None:
        # n(2) 仅用于控制台打印/预览图的观感；曲线绘制与解释必须用全精度表达式
        # （2 位有效数字会在 ~2000 量级的项上引入 ±30 的偏差，见 expr_parse 的教训）
        display_expr = pruned_expr.n(2)
        print(f"剪枝后的表达式为 {dependent} =")
        sp.pprint(display_expr)
        safe_preview(display_expr, f'{results_root}/prunedExpr.png')

    render_expr_trees(results_root, expr, pruned_expr)

    # 剪枝前后在训练数据上的拟合对比：解释 LLM 要靠它论证"剪掉这些项是否合理"
    data = load_training_data(results_root)
    fit = compare_fits(dependent, sym_names, data, expr, pruned_expr)
    print(f"[PRUNE] {format_fit_summary(fit)}")

    # 剪枝完成 → 剪枝前后表达式曲线 + 数据点（每个自变量一幅，人工检查贴合度）。
    # 与 expr.png 同级别的"给人看"产物：失败只告警，不拖垮收尾流程。
    try:
        from drsr_420.analysis.expr_curves import plot_data_curves
        plot_data_curves(results_root, dependent, sym_names, expr, pruned_expr)
    except Exception as e:
        print(f"[WARN] 剪枝前后曲线图生成失败（跳过）: {e}")

    return {
        "dependent": dependent,
        "sym_names": list(sym_names),
        "threshold": threshold,
        "sample_range": tuple(sample_range),
        "substituted_expr": sp.sstr(expr),
        "pruned_expr": sp.sstr(pruned_expr) if pruned_expr is not None else None,
        "nodes_visited": pruner.stats.nodes_visited,
        "nodes_pruned": pruner.stats.nodes_pruned,
        "prune_rate": pruner.stats.prune_rate,
        "removed": [
            {
                "kind": r.node_type,
                "term": sp.sstr(r.removed),
                "sensitivity": r.sensitivity,
                "depth": r.depth,
            }
            for r in pruner.stats.records
        ],
        "fit": fit,
    }


def find_best_eq(results_root: str, threshold: float = 0.1,
                 sample_range: tuple = (1, 14), role_clients=None):
    """收尾：寻找最优样本 → 敏感度剪枝与可视化 → 生成物理解释（含剪枝分析）。

    主函数仅做扁平编排，具体逻辑拆分到 find_best_sample / prune_and_visualize /
    explain_best_sample，避免原先 try-with-for-if-try 的深嵌套。

    Args:
        role_clients: ``llm.roles.RoleClients``；物理解释按其中的 ``explain`` 角色
            取客户端。省略时由 ``explain`` 模块自行按注册表解析（因此直接调用
            本函数也能拿到正确档案，不再依赖硬编码文件名）。
    """
    best = find_best_sample(results_root)
    if best is None:
        print("没有找到有效样本。")
        return

    score, path, func, params = best
    print(f"[BEST] score={score} file={path}")

    # 先剪枝：explain.md 要解释"剪枝后的表达式"与"剪掉了哪些项、为什么合理"，
    # 剪枝摘要（含剪枝前后在训练数据上的拟合对比）必须先算出来。
    pruning = prune_and_visualize(results_root, func, params, threshold, sample_range)

    # 物理解释（按 sample_order 匹配 Good 经验，含 RAG 文献注入与剪枝分析）
    order_match = re.search(r"samples_(\d+)", path)
    if not order_match:
        print("[WARN] 无法从样本文件名解析 sample_order，跳过物理解释。")
    else:
        explain_best_sample(results_root, func, order_match.group(1),
                            role_clients=role_clients, pruning=pruning)


def _latest_run_dir(root: str = "experiments") -> str | None:
    """返回最近修改的实验目录，供手工排查时"不传路径"使用。

    兼容两种布局：新布局 ``experiments/<问题>/<问题>_<时间戳>/`` 与旧布局
    ``experiments/<问题>_<时间戳>/``；判据是目录里确实有实验根目录的标志产物
    （``run.out`` 或 ``checkpoint.json``），避免误取 ``samples/`` 之类的子目录。
    """
    candidates = [p for p in glob.glob(os.path.join(root, "*", "*")) if os.path.isdir(p)]
    candidates += [p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
    runs = [p for p in candidates
            if os.path.exists(os.path.join(p, "run.out"))
            or os.path.exists(os.path.join(p, "checkpoint.json"))]
    if not runs:
        return None
    return max(runs, key=os.path.getmtime)


if __name__ == "__main__":
    # 手工排查用：`python -m drsr_420.analysis.find_best_eq [实验目录]`
    # 不给路径时取 experiments/ 下最近修改的一次 run（见 _latest_run_dir）。
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else _latest_run_dir()
    if target is None:
        print("[WARN] 未找到任何实验目录，请显式给出路径。")
    else:
        print(f"[INFO] 收尾分析目标: {target}")
        find_best_eq(target)
