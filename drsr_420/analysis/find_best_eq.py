"""收尾分析编排：最佳样本 → 物理解释 → 敏感度剪枝与可视化。

位置
----
``runtime/pipeline.main`` 在实验结束前调用一次 :func:`find_best_eq`（非 Agent，
是普通工具函数——架构图中的"收尾"环节）。

编排（各步骤都拆成了单一职责的模块）
------------------------------------
::

    find_best_eq(results_root)
      ├── find_best_sample()      扫描 samples/*.json 取最高分样本
      ├── explain.explain_best_sample()      物理解释 → explain.txt（可失败，仅告警）
      └── prune_and_visualize()
            ├── expr_parse.expr_substitution()   骨架字符串 → SymPy 表达式
            ├── sensitivity_prune.SensitivityPruner.prune()  敏感度剪枝
            └── expr_viz.safe_preview() / render_expr_trees()  预览图与树图

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
                        threshold: float, sample_range: tuple) -> None:
    """基于敏感度分析剪枝最优公式，并保存表达式预览图与表达式树图。"""
    parsed = _parse_symbols(func)
    if parsed is None:
        print("[WARN] 无法从样本中解析 Dependent/Independents，跳过剪枝。")
        return
    dependent, sym_names = parsed
    symbols = sp.symbols(sym_names)

    pruner = SensitivityPruner(symbols=symbols, threshold=threshold, sample_range=sample_range)
    expr = expr_substitution(func, params)
    if expr is None:
        print("[WARN] 表达式解析失败，跳过剪枝。")
        return

    print(f"剪枝前的表达式为 {dependent} =")
    sp.pprint(expr)
    safe_preview(expr, f'{results_root}/expr.png')

    try:
        pruned_expr = pruner.prune(expr, verbose=True)
    except Exception as e:
        print(f"[WARN] 剪枝失败: {e}")
        return

    pruned_expr = pruned_expr.n(2)
    print(f"剪枝后的表达式为 {dependent} =")
    sp.pprint(pruned_expr)
    safe_preview(pruned_expr, f'{results_root}/prunedExpr.png')

    render_expr_trees(results_root, expr, pruned_expr)


def find_best_eq(results_root: str, threshold: float = 0.1,
                 sample_range: tuple = (1, 14), role_clients=None):
    """收尾：寻找最优样本 → 生成物理解释 → 敏感度剪枝与可视化。

    主函数仅做扁平编排，具体逻辑拆分到 find_best_sample / explain_best_sample /
    prune_and_visualize，避免原先 try-with-for-if-try 的深嵌套。

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

    # 物理解释（按 sample_order 匹配 Good 经验，含 RAG 文献注入）
    order_match = re.search(r"samples_(\d+)", path)
    if not order_match:
        print("[WARN] 无法从样本文件名解析 sample_order，跳过物理解释。")
    else:
        explain_best_sample(results_root, func, order_match.group(1),
                            role_clients=role_clients)

    # 敏感度剪枝 + 表达式预览/树图
    prune_and_visualize(results_root, func, params, threshold, sample_range)


if __name__ == "__main__":
    # 手工排查用：对指定实验目录跑一遍收尾（默认取最近一次 MRF 压缩实验）
    find_best_eq("..\\experiments\\MRFCompress-Ellipsoid_20260813-133229")
