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

依赖：matplotlib（缺失时只告警跳过，不影响实验）；公式解析复用
``expr_substitution`` / ``SensitivityPruner``，与剪枝流程的参数完全一致，
因此"剪枝后"曲线与 run.out 里打印的剪枝表达式同源。
"""
from __future__ import annotations

import json
import os

import numpy as np
import sympy as sp

from drsr_420.analysis.expr_parse import expr_substitution
from drsr_420.analysis.find_best_eq import _parse_symbols, find_best_sample
from drsr_420.analysis.sensitivity_prune import SensitivityPruner

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_csv(data_csv: str) -> str | None:
    """config_snapshot 里的 data_csv 相对项目根；兼容绝对路径与 cwd 相对路径。"""
    for base in (_REPO_ROOT, os.getcwd()):
        p = data_csv if os.path.isabs(data_csv) else os.path.join(base, data_csv)
        if os.path.isfile(p):
            return p
    return None


def plot_expr_curves(results_root: str, threshold: float = 0.1,
                     sample_range: tuple = (1, 14)) -> list[str]:
    """对最优样本画剪枝前/后表达式曲线 + 数据散点（每个自变量一幅）。

    返回成功写出的图片路径列表；任何一步失败只告警并返回已完成的路径。
    """
    import matplotlib
    matplotlib.use("Agg")            # 无头环境；必须在 pyplot 之前
    import matplotlib.pyplot as plt

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

    # 训练数据（CSV 列名 = 自变量名 + 因变量名）
    snap_path = os.path.join(results_root, "config_snapshot.json")
    try:
        with open(snap_path, "r", encoding="utf-8") as f:
            data_csv = json.load(f)["data_csv"]
    except Exception as e:
        print(f"[WARN] 读取 config_snapshot.json 失败，跳过曲线绘制: {e}")
        return []
    csv_path = _resolve_csv(data_csv)
    if csv_path is None:
        print(f"[WARN] 数据文件不存在: {data_csv}，跳过曲线绘制。")
        return []
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if dependent not in data.dtype.names:
        print(f"[WARN] 数据里没有因变量列 {dependent}，跳过曲线绘制。")
        return []
    missing = [v for v in sym_names if v not in data.dtype.names]
    if missing:
        print(f"[WARN] 数据里缺自变量列 {missing}，跳过曲线绘制。")
        return []

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
        args = [np.asarray(data[name], dtype=float)[order] for name in sym_names]
        y_data = np.asarray(data[dependent], dtype=float)[order]
        try:
            y_orig = np.asarray(f_orig(*args), dtype=float)
            y_pruned = (np.asarray(f_pruned(*args), dtype=float)
                        if f_pruned is not None else None)
        except Exception as e:
            print(f"[WARN] {var} 曲线求值失败，跳过该自变量: {e}")
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x_all, np.asarray(data[dependent], dtype=float),
                   s=28, facecolors="none", edgecolors="tab:gray",
                   label="data points")
        ax.plot(xs, y_orig, color="tab:blue", lw=2, label="before pruning")
        if y_pruned is not None:
            ax.plot(xs, y_pruned, color="tab:red", lw=2, ls="--",
                    label="after pruning")
        ax.set_xlabel(var)
        ax.set_ylabel(dependent)
        ax.set_title(f"{dependent} vs {var}"
                     f"\n(model evaluated along the training data path, sorted by {var})")
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="剪枝前后表达式曲线 + 数据点")
    parser.add_argument("results_root", help="实验目录")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="敏感度剪枝阈值（与 find_best_eq 默认一致）")
    args = parser.parse_args()
    plot_expr_curves(args.results_root, threshold=args.threshold)
