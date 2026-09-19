"""剪枝前后表达式曲线 + 数据点可视化（每个自变量一幅图）。

用途
----
实验收尾后人工检查：最优公式在剪枝前后的行为差异，以及它们与训练数据的贴合
程度。多元公式的曲线是投影：对每个自变量画一幅图——横轴 = 该自变量，纵轴 =
因变量。**曲线沿训练数据路径求值**：把数据按当前自变量排序，在每个数据点的
完整坐标处代入模型再连线。这很重要——训练数据的自变量往往强耦合（例如 MRF
数据里 lambda12*lambda23 ≈ 常数，数据躺在一条一维流形上），若改为把其它变量
固定在某个中位数切片上，求值点根本不在数据流形附近，曲线会与散点严重脱节、
产生"拟合很好但画出来对不上"的错觉。

用法
----
::

    python -m drsr_420.analysis.expr_curves <results_root> [--threshold 0.1]

产物：``<results_root>/expr_curve_<自变量名>.png``（每个自变量一幅）。
**本次没有真剪掉项时只画一条曲线**并在图注里注明未剪枝——画两条完全重合的曲线
（甚至"剪枝后"比"剪枝前"更复杂，那只是 simplify 通分）会误导读者。

依赖：matplotlib（缺失时只告警跳过，不影响实验）；公式解析复用
``expr_substitution`` / ``SensitivityPruner``，与剪枝流程的参数完全一致，
因此"剪枝后"曲线与 run.out 里打印的剪枝表达式同源。
"""
from __future__ import annotations

import os

import numpy as np
import sympy as sp

from drsr_420.analysis.expr_parse import expr_substitution
from drsr_420.analysis.find_best_eq import _parse_symbols, find_best_sample
from drsr_420.analysis.holdout import load_test_data
from drsr_420.analysis.prune_report import (_warn_once, load_training_data,
                                            resolve_columns,
                                            resolve_csv as _resolve_csv)
from drsr_420.analysis.sensitivity_prune import SensitivityPruner

__all__ = ["plot_data_curves", "plot_expr_curves", "_resolve_csv"]


def plot_data_curves(results_root: str, dependent: str, sym_names: list[str],
                     expr, pruned=None, test_csv: str | None = None) -> list[str]:
    """核心绘图：按训练数据路径画剪枝前/后表达式曲线 + 数据散点（含 held-out 点）。

    ``pruned=None`` 表示**本次没有剪枝后的表达式**（没真剪掉项，或剪枝失败/未执行）：
    此时只画一条曲线并在图注里注明"未剪枝"，不画一条与它完全重合的"剪枝后"曲线。

    ``test_csv`` 给定（或自动探测到）时，把 held-out 点用**另一种标记**画上去——
    它们没有参与参数拟合与样本选择，是肉眼判断"公式是不是只在训练点上插值"的唯一
    可视化证据。曲线本身仍只沿**训练数据路径**求值（原因见模块开头），不向 held-out
    点连线。

    供两处调用：``prune_and_visualize``（剪枝完成后自动触发）与本模块的
    ``plot_expr_curves``（对既有实验目录独立补跑）。返回成功写出的图片路径；
    失败只告警，不抛异常——曲线是"给人看的"产物，不该拖垮收尾流程。
    """
    import matplotlib
    matplotlib.use("Agg")            # 无头环境；必须在 pyplot 之前
    import matplotlib.pyplot as plt

    # 训练数据（CSV 列名 = 自变量名 + 因变量名）；定位与读取复用 prune_report
    data = load_training_data(results_root)
    if data is None:
        return []
    # 列名对齐：函数头里的名字是 LLM 写的，可能和 CSV 表头不完全一样（如 um vs miu）
    try:
        dep_col, ind_cols, note = resolve_columns(data, dependent, sym_names)
    except KeyError as e:
        print(f"[WARN] {e}，跳过曲线绘制。")
        return []
    if note:
        _warn_once(f"变量名与数据列不完全一致：{note}")

    # held-out 点（没参与拟合与选择）：只画散点，不参与曲线求值
    test_data = load_test_data(results_root, test_csv)
    test_cols = None
    if test_data is not None:
        try:
            test_dep, test_ind, test_note = resolve_columns(test_data, dependent, sym_names)
            test_cols = (test_dep, test_ind)
            if test_note:
                _warn_once(f"样本外数据的变量名与列不完全一致：{test_note}")
        except KeyError as e:
            print(f"[WARN] 样本外数据无法用于绘图，仅画训练点：{e}")

    syms = list(sp.symbols(sym_names))
    try:
        f_orig = sp.lambdify(syms, expr, modules="numpy")
        f_pruned = (sp.lambdify(syms, pruned, modules="numpy")
                    if pruned is not None else None)
    except Exception as e:
        print(f"[WARN] 表达式数值化失败，跳过曲线绘制: {e}")
        return []

    # 曲线沿训练数据路径求值：按当前自变量排序，在完整数据坐标处代入模型。
    # 不要用"其它变量固定在中位数"的切片——数据自变量强耦合时切片不在数据
    # 流形上，曲线会与散点严重脱节（实测 MRFCompress-Cuboid_20260917-134427）。
    written: list[str] = []
    for var in sym_names:
        x_all = np.asarray(data[var], dtype=float)
        order = np.argsort(x_all, kind="stable")
        xs = x_all[order]
        args = [np.asarray(data[col], dtype=float)[order]
                for col in ind_cols]
        y_data = np.asarray(data[dep_col], dtype=float)[order]
        try:
            y_orig = np.asarray(f_orig(*args), dtype=float)
            y_pruned = (np.asarray(f_pruned(*args), dtype=float)
                        if f_pruned is not None else None)
        except Exception as e:
            print(f"[WARN] {var} 曲线求值失败，跳过该自变量: {e}")
            continue

        n_train = int(np.size(x_all))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x_all, np.asarray(data[dep_col], dtype=float),
                   s=28, facecolors="none", edgecolors="tab:gray",
                   label=f"training points ({n_train})")
        ax.plot(xs, y_orig, color="tab:blue", lw=2,
                label="before pruning" if y_pruned is not None
                else "model (no pruning applied)")
        if y_pruned is not None:
            ax.plot(xs, y_pruned, color="tab:red", lw=2, ls="--",
                    label="after pruning")
        # held-out 点：不同标记 + 不连线（它们不在训练路径上，连线会假装有插值关系）
        n_test = 0
        if test_cols is not None:
            test_dep, test_ind = test_cols
            idx = sym_names.index(var)
            ax.scatter(np.asarray(test_data[test_ind[idx]], dtype=float),
                       np.asarray(test_data[test_dep], dtype=float),
                       s=70, marker="^", facecolors="none", edgecolors="tab:green",
                       linewidths=1.8,
                       label=f"held-out / test.csv ({np.size(test_data[test_dep])})"
                             " — not used for fitting")
            n_test = int(np.size(test_data[test_dep]))
        ax.set_xlabel(var)
        ax.set_ylabel(dependent)
        # 未剪枝时（pruned=None）在图注里写明，避免读者以为"少了一条曲线"是画漏了
        note_axes = ("" if y_pruned is not None
                     else "\n(no pruning applied: the pruning step removed nothing)")
        test_note = (f"\ntriangles = held-out points ({n_test}), never used for fitting "
                     f"or model selection" if n_test else "")
        ax.set_title(f"{dependent} vs {var}"
                     f"\n(model evaluated along the training data path, sorted by {var})"
                     + note_axes + test_note)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(results_root, f"expr_curve_{var}.png")
        try:
            fig.savefig(out, dpi=150)
            written.append(out)
            print(f"[INFO] Saved {out}")
        except Exception as e:
            print(f"[WARN] 保存 {out} 失败: {e}")
        finally:
            plt.close(fig)
    return written


def plot_expr_curves(results_root: str, threshold: float = 0.1,
                     sample_range: tuple = (1, 14),
                     test_csv: str | None = None) -> list[str]:
    """对既有实验目录独立补跑：最优样本 → 剪枝 → 核心绘图。

    供 ``python -m drsr_420.analysis.expr_curves`` CLI 使用；实验管线内的自动
    绘图走 ``prune_and_visualize → plot_data_curves``，不经过这里（避免重复剪枝）。
    ``test_csv`` 语义同 ``plot_data_curves``（``None`` = 自动探测）。
    """
    best = find_best_sample(results_root)
    if best is None:
        print("[WARN] 未找到有效样本，跳过曲线绘制。")
        return []
    _score, _path, func, params = best
    parsed = _parse_symbols(func)
    if parsed is None:
        print("[WARN] 无法解析 Dependent/Independents，跳过曲线绘制。")
        return []
    dependent, sym_names = parsed

    expr = expr_substitution(func, params)
    if expr is None:
        print("[WARN] 表达式解析失败，跳过曲线绘制。")
        return []

    # 剪枝：与 prune_and_visualize 同一套参数，保证曲线与实验产物同源
    pruner = SensitivityPruner(symbols=sp.symbols(sym_names),
                               threshold=threshold, sample_range=sample_range)
    try:
        pruned = pruner.prune(expr, verbose=False)
    except Exception as e:
        print(f"[WARN] 剪枝失败，只画剪枝前曲线: {e}")
        pruned = None
    if pruned is not None and not pruner.stats.actually_pruned:
        # 没真剪掉项（prune() 已把公式回退成原式）：只画一条曲线，
        # 不画一条与它完全重合的"剪枝后"曲线。
        print("[INFO] 本次剪枝未移除任何项：曲线只画一条（未剪枝）。")
        pruned = None

    return plot_data_curves(results_root, dependent, sym_names, expr, pruned,
                            test_csv=test_csv)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="剪枝前后表达式曲线 + 数据点")
    parser.add_argument("results_root", help="实验目录")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="敏感度剪枝阈值（与 find_best_eq 默认一致）")
    parser.add_argument("--test_csv", default=None,
                        help="held-out 数据路径；缺省自动探测（同目录 test.csv）；none 关闭")
    args = parser.parse_args()
    plot_expr_curves(args.results_root, threshold=args.threshold,
                     test_csv=args.test_csv)
