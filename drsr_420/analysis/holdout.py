"""样本外（held-out）验证：``test.csv`` 上的泛化性指标，**只报告，不参与任何选择**。

角色归属
--------
收尾分析（analysis）阶段的泛化性检查，与 :mod:`drsr_420.analysis.prune_report` 并列：
后者回答"剪枝有没有削弱模型"，本模块回答"模型在没参与拟合的点上还准不准"。

为什么必须与训练点分开
----------------------
评估器（``evaluation.problems.evaluate``）在**同一批点**上拟合参数并打分：
``score = -MSE``、``NMSE = MSE / var(outputs)``，都是样本内指标，选最佳样本用的也是这个
分。MRF 这类小样本问题里这尤其危险——实测某次运行 8 个训练点、10 个自由参数，样本内
NMSE 1.45e-7，而两个同分布 held-out 点上最大相对误差 **7.95%**（NMSE 放大 1.3e6 倍）。
因此这里的指标只写进 run.out / report.md，**绝不**回灌进评分、早停或样本选择：一旦
参与选择，它就不再是 held-out，实验之间也不再可比。

数据来源：ID 与 OOD 两条通道
----------------------------
**同分布（ID）** 由 :func:`resolve_test_csv` 解析：``--test_csv`` 显式指定 →
``config_snapshot.json`` 记录的路径 → 训练数据同目录自动探测 ``test.csv`` →
``test_id.csv`` → 按目录名推断的 ``data/<问题名>/test.csv``（历史目录）。

**分布外（OOD）** 由 :func:`resolve_ood_csv` 解析，语义相同，自动探测 ``ood_test.csv``
→ ``test_ood.csv``（LSR-Synth 四域与 LLM-SR 真实任务各自的命名）。旧实现只探测同分布的
``test.csv``，于是 benchmark 数据里那批 OOD 文件**从未被评估**——论文要求 ID 与 OOD
分开报，缺了 OOD 这一列就看不出一维外推是否失效。

两者都找不到就跳过，老实验目录行为不变。
"""
from __future__ import annotations

import json
import os

import numpy as np
import sympy as sp

from drsr_420.analysis.prune_report import (_warn_once, infer_data_csv,
                                            resolve_columns, resolve_csv)

__all__ = [
    "resolve_test_csv", "resolve_ood_csv", "load_test_data", "load_ood_data",
    "evaluate_holdout", "in_sample_metrics", "format_holdout_summary",
    "HOLDOUT_HEADING", "render_holdout_section", "strip_holdout_section",
    "ID_HOLDOUT_NAMES", "OOD_HOLDOUT_NAMES",
]

#: report.md 里样本外验证小节的标题（机器生成，正文若自带同名小节会被替换）。
HOLDOUT_HEADING = "## 样本外验证"

#: 关闭自动探测的取值：``--test_csv none``。
_DISABLED = ("", "none", "null", "off", "no", "false")

#: **同分布** held-out 的候选文件名（按优先级自动探测）。``test_id.csv`` 是
#: LLM-SR 真实任务（oscillator/stressstrain/bactgrow）的命名。
ID_HOLDOUT_NAMES = ("test.csv", "test_id.csv")

#: **分布外（OOD）** held-out 的候选文件名。``ood_test.csv`` 是 LSR-Synth 四域
#: （BPG0/CRK0/PO0/MatSci0）的命名，``test_ood.csv`` 是 LLM-SR 真实任务的命名。
#: 二者此前都探测不到（旧实现只找 ``test.csv``）→ benchmark 的 OOD 一列永远为空。
OOD_HOLDOUT_NAMES = ("ood_test.csv", "test_ood.csv")

#: 解析结果缓存：曲线与报告都会调用，避免同一路径被反复打印/反复读盘。
_resolved: dict[tuple, str | None] = {}

#: 已打印过的数据来源：同一个文件在一条实验流程里只提示一次。
_logged: set[str] = set()


def _snapshot_value(results_root: str, key: str) -> str:
    """读 config_snapshot.json 里的某个字段（没有则空串）。

    ``test_csv`` 兼容两种写法：``test_csv``（生效值）与早期的 ``test_csv_arg``
    （命令行原值）。
    """
    try:
        with open(os.path.join(results_root, "config_snapshot.json"), "r",
                  encoding="utf-8") as f:
            snap = json.load(f)
    except Exception:
        return ""
    value = snap.get(key)
    if not value and key == "test_csv":
        value = snap.get("test_csv_arg")
    return str(value or "")


def _train_dir_csv(results_root: str, train_csv: str | None) -> str | None:
    """本次运行的训练 CSV 绝对路径（用于"同目录找兄弟文件"）。

    ``train_csv`` 由 CLI 直接把 ``--data_csv`` 传进来；快照要等启动流程后段才落盘，
    只靠快照推断会拿不到这条最可靠的线索，故两者都用，最后才按目录名兜底。
    """
    train_path = resolve_csv(train_csv, results_root) if train_csv else None
    if train_path is None:
        train_data_csv = _snapshot_value(results_root, "data_csv")
        train_path = resolve_csv(train_data_csv, results_root) if train_data_csv else None
    if train_path is None:
        train_path = infer_data_csv(results_root)
    return train_path


def _resolve_holdout(results_root: str, explicit: str | None, snapshot_key: str,
                     names: tuple[str, ...], train_csv: str | None,
                     label: str, flag: str) -> str | None:
    """通用 held-out 路径解析：显式 → 快照 → 训练数据同目录按 ``names`` 顺序探测。

    ``explicit`` 取 ``"none"`` 等关闭值时直接返回 ``None``（不退回自动探测——那会
    违背用户显式关闭的意图）。解析不到返回 ``None``，调用方静默跳过。
    """
    key = (str(results_root), explicit, snapshot_key)
    if key in _resolved:
        return _resolved[key]

    def _remember(path: str | None, how: str) -> str | None:
        if path and path not in _logged:
            _logged.add(path)
            print(f"[INFO] {label}数据（{how}）: {path}")
        _resolved[key] = path
        return path

    if explicit is not None:
        if explicit.strip().lower() in _DISABLED:
            return _remember(None, "已关闭")
        path = resolve_csv(explicit, results_root)
        if path is None:
            print(f"[WARN] {flag} 指定的文件不存在，跳过{label}: {explicit}")
            return _remember(None, "显式指定")
        return _remember(path, "显式指定")

    snapped = _snapshot_value(results_root, snapshot_key)
    if snapped:
        if snapped.strip().lower() in _DISABLED:
            return _remember(None, f"config_snapshot.{snapshot_key}=关闭")
        path = resolve_csv(snapped, results_root)
        if path:
            return _remember(path, f"config_snapshot.{snapshot_key}")

    train_path = _train_dir_csv(results_root, train_csv)
    if train_path:
        directory = os.path.dirname(train_path)
        for name in names:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return _remember(candidate, "训练数据同目录自动探测")
    return _remember(None, "未找到")


def resolve_test_csv(results_root: str, test_csv: str | None = None,
                     train_csv: str | None = None) -> str | None:
    """解析**同分布** held-out 数据路径；解析不到返回 ``None``（调用方静默跳过）。

    优先级：显式 ``test_csv`` → 快照记录 → 训练数据同目录自动探测（``test.csv`` →
    ``test_id.csv``）→ ``data/<问题名>/test.csv``（历史目录兜底）。``test_csv`` 取
    ``"none"`` 等关闭值时直接返回 ``None``。
    """
    return _resolve_holdout(results_root, test_csv, "test_csv", ID_HOLDOUT_NAMES,
                            train_csv, "样本外验证", "--test_csv")


def resolve_ood_csv(results_root: str, test_ood_csv: str | None = None,
                    train_csv: str | None = None) -> str | None:
    """解析**分布外（OOD）** held-out 数据路径；语义同 :func:`resolve_test_csv`。

    自动探测 ``ood_test.csv`` → ``test_ood.csv``（LSR-Synth 与 LLM-SR 真实任务各自的
    命名）。旧实现只探测同分布的 ``test.csv``，benchmark 的 OOD 因此从未被评估。
    """
    return _resolve_holdout(results_root, test_ood_csv, "test_ood_csv",
                            OOD_HOLDOUT_NAMES, train_csv, "分布外验证", "--test_ood_csv")


def _load_struct(path: str | None, label: str) -> np.ndarray | None:
    """把 CSV 读成结构化数组；路径为空 / 读取失败 / 无表头时返回 ``None``。"""
    if not path:
        return None
    try:
        data = np.genfromtxt(path, delimiter=",", names=True)
    except Exception as e:
        print(f"[WARN] 读取{label}失败: {e}")
        return None
    if data.dtype.names is None or data.size == 0:
        print(f"[WARN] {label}为空或缺少表头: {path}")
        return None
    return data


def load_test_data(results_root: str, test_csv: str | None = None,
                   train_csv: str | None = None) -> np.ndarray | None:
    """读取**同分布** held-out 数据（结构化数组）；路径解析不到或读取失败返回 ``None``。"""
    return _load_struct(resolve_test_csv(results_root, test_csv, train_csv=train_csv),
                        "样本外数据")


def load_ood_data(results_root: str, test_ood_csv: str | None = None,
                  train_csv: str | None = None) -> np.ndarray | None:
    """读取**分布外（OOD）** held-out 数据（结构化数组）；语义同 :func:`load_test_data`。"""
    return _load_struct(resolve_ood_csv(results_root, test_ood_csv, train_csv=train_csv),
                        "分布外数据")


def _relative_error(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """逐点相对误差（分母加 1e-12 下限，避免真值 0 处除零）。"""
    return np.abs(pred - y) / np.maximum(np.abs(y), 1e-12)


def evaluate_holdout(dependent: str, sym_names: list[str], expr, test_data: np.ndarray,
                     train_var: float | None = None, path: str = "",
                     train_data: np.ndarray | None = None) -> dict | None:
    """在 held-out 数据上评估表达式，返回指标字典（列名对不上/求值失败返回 ``None``）。

    NMSE 用**训练集方差**作分母（与评估器同口径，便于和样本内 NMSE 直接比大小）。

    ``train_data`` 给定时会检查"held-out 点是否其实就在训练集里"：实测
    ``data/MRFShear-3`` 与 ``data/MRFCompress-3`` 的 ``test.csv`` 与 ``train.csv``
    **曾**逐行相同（现已互异），把它们当独立 held-out 报出去等于谎报验证结果，因此这种
    情况会在指标里标出 ``overlaps_train``，渲染时也会明说。注意另一类同样致命的情形：
    MRF 系的 ``test.csv`` **只有 2 行**且落在训练点之间（插值），即使不重合，指标也
    只是存在性提示——渲染时会照数报 ``n_points``，引用方必须自己看这个分母。
    """
    if test_data is None:
        return None
    try:
        dep_col, ind_cols, note = resolve_columns(test_data, dependent, sym_names)
    except KeyError as e:
        print(f"[WARN] 样本外验证跳过：{e}")
        return None
    if note:
        _warn_once(f"变量名与数据列不完全一致：{note}")

    y = np.asarray(test_data[dep_col], dtype=float)
    args = [np.asarray(test_data[name], dtype=float) for name in ind_cols]
    try:
        func = sp.lambdify(list(sp.symbols(sym_names)), expr, modules="numpy")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pred = np.asarray(func(*args), dtype=float)
    except Exception as e:
        print(f"[WARN] 样本外表达式求值失败: {e}")
        return None
    if pred.shape != y.shape:
        pred = np.broadcast_to(pred, y.shape)
    if not np.isfinite(pred).all():
        print("[WARN] 样本外预测含非有限值（该点按 inf 误差计入）。")

    err = pred - y
    abs_err = np.abs(err)
    rel_err = _relative_error(pred, y)
    mse = float(np.nanmean(np.square(err)))

    # "held-out 其实就在训练集里"检测：逐行比较（列名对齐后），全中即不是独立验证集
    n_overlap = 0
    if train_data is not None:
        try:
            tr_dep, tr_ind, _ = resolve_columns(train_data, dependent, sym_names)
            train_rows = {
                (float(train_data[tr_dep][i]),
                 *(float(train_data[c][i]) for c in tr_ind))
                for i in range(len(train_data[tr_dep]))
            }
            test_rows = [
                (float(y[i]), *(float(test_data[c][i]) for c in ind_cols))
                for i in range(y.size)
            ]
            n_overlap = sum(1 for r in test_rows if r in train_rows)
        except KeyError:
            n_overlap = 0

    out = {
        "path": path,
        "n_points": int(y.size),
        "mse": mse,
        "nmse": (mse / train_var) if train_var else None,
        "max_abs_err": float(np.nanmax(abs_err)),
        "max_rel_err": float(np.nanmax(rel_err)),
        "train_var": train_var,
        "n_overlap_train": int(n_overlap),
        "overlaps_train": bool(n_overlap and n_overlap == y.size),
        "in_sample_nmse": None,
        "rows": [
            {
                "variables": {name: float(test_data[col][i])
                              for name, col in zip(sym_names, ind_cols)},
                "observed": float(y[i]),
                "predicted": float(pred[i]),
                "abs_err": float(abs_err[i]),
                "rel_err": float(rel_err[i]),
            }
            for i in range(y.size)
        ],
    }
    return out


def in_sample_metrics(fit: dict | None) -> dict:
    """样本内指标，**以最终发布的表达式（剪枝后）为准**，缺剪枝后值时才回退剪枝前。

    held-out 侧评的是发布版表达式（``find_best_eq`` 传 ``published``，即未剪枝时等于
    原式），因此样本内对照必须取自同一表达式。实测 report.md 曾把两种口径并列：
    样本内 MSE=0.0523（剪枝前）对样本外 MSE=869（剪枝后）—— 剪枝明明把模型从
    MSE 0.05 削弱到 6305，表格却显示样本内仍"很准"，属于口径不一致造成的误导。

    ``pruned`` 标记本次样本内值是否来自剪枝后表达式，供渲染时写清口径。
    """
    fit = fit or {}

    def pick(after_key: str, before_key: str):
        value = fit.get(after_key)
        return fit.get(before_key) if value is None else value

    after_mse = fit.get("mse_after")
    return {
        "n_points": fit.get("n_points"),
        "mse": pick("mse_after", "mse_before"),
        "nmse": pick("nmse_after", "nmse_before"),
        "max_abs_err": pick("max_abs_err_after", "max_abs_err_before"),
        "max_rel_err": pick("max_rel_err_after", "max_rel_err_before"),
        "pruned": after_mse is not None,
    }


def format_holdout_summary(holdout: dict | None, fit: dict | None = None) -> str:
    """把样本外指标渲染成一行控制台文本（无数据时给出原因）。"""
    if not holdout:
        return "样本外验证：本次没有可用的 held-out 数据（未指定 --test_csv 且未自动探测到 test.csv）。"
    if holdout.get("overlaps_train"):
        return (f"样本外验证：{holdout['path']} 的 {holdout['n_points']} 行"
                f"**全部出现在训练集中**，它不是独立的 held-out 集——"
                f"下面的指标与样本内指标同义，不能当作泛化能力。")
    parts = [f"样本外验证：{holdout['n_points']} 个 held-out 点上 "
             f"MSE={holdout['mse']:.6g}"]
    if holdout.get("nmse") is not None:
        parts.append(f"NMSE={holdout['nmse']:.6g}")
    parts.append(f"最大绝对误差={holdout['max_abs_err']:.6g}")
    parts.append(f"最大相对误差={holdout['max_rel_err']:.2%}")
    line = "，".join(parts) + "。"
    in_nmse = in_sample_metrics(fit)["nmse"]
    if in_nmse and holdout.get("nmse") is not None:
        line += (f"（样本内 NMSE={in_nmse:.6g}，样本外/样本内="
                 f"{holdout['nmse'] / in_nmse:.3g} 倍）")
    line += " 样本外指标不参与采样、打分与样本选择。"
    return line


def _render_holdout_block(h: dict, fit: dict | None) -> list[str]:
    """单组 held-out 的正文：数据来源 + 样本内/外对照表 + 逐点明细。

    ID 与 OOD 各自调用一次；跨组的口径说明由 :func:`render_holdout_section` 统一附，
    避免同一段说明在两组之间重复。
    """
    lines: list[str] = []
    source = h.get("path") or "test.csv"
    lines.append(f"数据来源：`{source}`（{h['n_points']} 个点，未参与参数拟合、"
                 f"打分与样本选择）")
    if h.get("overlaps_train"):
        lines.append("")
        lines.append(f"> **注意：该文件的 {h['n_points']} 行全部出现在训练集里**，"
                     f"它并不是独立的 held-out 集——下表与样本内指标同义，"
                     f"不能用来论证泛化能力。需要真正的样本外验证时，请另取未参与"
                     f"拟合与选择的数据点。")
    lines.append("")
    in_sample = in_sample_metrics(fit)
    in_mse = in_sample["mse"]
    in_n = in_sample["n_points"]
    in_max_abs = in_sample["max_abs_err"]
    in_max_rel = in_sample["max_rel_err"]

    def _fmt(value, spec="{:.6g}"):
        return "本次不可用" if value is None else spec.format(value)

    lines.append("| 指标 | 样本内（训练点） | 样本外（held-out 点） |")
    lines.append("|---|---|---|")
    lines.append(f"| 点数 | {in_n if in_n else '未知'} | {h['n_points']} |")
    lines.append(f"| MSE | {_fmt(in_mse)} | {h['mse']:.6g} |")
    lines.append(f"| NMSE（分母为训练集方差） | {_fmt(in_sample['nmse'])} "
                 f"| {_fmt(h.get('nmse'))} |")
    lines.append(f"| 最大绝对误差 | {_fmt(in_max_abs)} | {h['max_abs_err']:.6g} |")
    lines.append(f"| 最大相对误差 | {_fmt(in_max_rel, '{:.2%}')} "
                 f"| {h['max_rel_err']:.2%} |")
    in_nmse = in_sample["nmse"]
    if in_nmse and h.get("nmse") is not None:
        lines.append("")
        lines.append(f"样本外 NMSE 是样本内的 **{h['nmse'] / in_nmse:.3g} 倍**"
                     f"（样本内 {in_nmse:.6g} → 样本外 {h['nmse']:.6g}）。")

    lines.append("")
    lines.append("逐点明细：")
    lines.append("")
    var_names = list(h["rows"][0]["variables"]) if h["rows"] else []
    header = "| # | " + " | ".join(var_names) + " | 观测 | 预测 | 绝对误差 | 相对误差 |"
    lines.append(header)
    lines.append("|" + "---|" * (len(var_names) + 5))
    for i, row in enumerate(h["rows"], 1):
        vals = " | ".join(f"{row['variables'][v]:.6g}" for v in var_names)
        lines.append(f"| {i} | {vals} | {row['observed']:.6g} | {row['predicted']:.6g} "
                     f"| {row['abs_err']:.6g} | {row['rel_err']:.2%} |")
    return lines


def render_holdout_section(holdout: dict | None, fit: dict | None = None,
                           ood: dict | None = None) -> str:
    """渲染 report.md 的「样本外验证」小节（机器生成，数字不由 LLM 转述）。

    ``holdout`` 是**同分布（ID）** held-out 的指标，``ood`` 是**分布外（OOD）** 的
    指标（:func:`load_ood_data` + :func:`evaluate_holdout` 得到）。两者都给时同列在
    本节内、各一张对照表并标明 ID / OOD——论文要求 ID 与 OOD 分开报，混成一个数字
    看不出外推是否失效。
    """
    lines = [HOLDOUT_HEADING, ""]
    if not holdout and not ood:
        lines.append("本次没有可用的 held-out 数据（未指定 `--test_csv`/`--test_ood_csv`，"
                     "也未在数据目录自动探测到 `test.csv`/`test_id.csv` / "
                     "`ood_test.csv`/`test_ood.csv`），因此没有样本外指标。")
        lines.append("")
        lines.append("> 注意：正文里的 MSE/NMSE 都是**样本内**指标（评估器在同一批点上"
                     "拟合参数并打分），不能当作泛化误差。")
        return "\n".join(lines)

    if holdout and ood:
        lines.append("本小节含两组样本外数据：**同分布 held-out（ID）** 与 "
                     "**分布外（OOD）**；两者点数与口径不同，须分开读。")
        lines.append("")
    if holdout:
        lines += _render_holdout_block(holdout, fit)
    else:
        lines.append("本次**没有同分布（ID）**的 held-out 数据（未指定 `--test_csv`，"
                     "也未自动探测到 `test.csv`/`test_id.csv`）。")
    if ood:
        lines += ["", "### 分布外（OOD）", ""]
        lines += _render_holdout_block(ood, fit)

    in_sample = in_sample_metrics(fit)
    lines.append("")
    lines.append("> 口径说明：样本内指标对应**最终发布的表达式**（发生剪枝时即剪枝后表达式，"
                 "与样本外所用表达式相同），由评估器在同一批训练点上拟合参数并打分得到；"
                 "样本外 NMSE 用训练集方差作分母（与样本内同口径）。")
    if not in_sample["pruned"]:
        lines.append("> 本次没有剪枝后指标（未发生实质剪枝），样本内列即原表达式的指标。")
    lines.append("> 样本外指标只用于报告，不参与采样、打分、早停与样本选择——"
                 "参与选择后它就不再是 held-out。")
    return "\n".join(lines)


def strip_holdout_section(text: str) -> str:
    """去掉正文里自带的「样本外验证」小节（清单/数字一律由系统生成，避免两套数字）。"""
    if not text or HOLDOUT_HEADING not in text:
        return text
    kept: list[str] = []
    skipping = False
    for line in text.splitlines():
        if line.strip().startswith(HOLDOUT_HEADING):
            skipping = True
            continue
        if skipping and line.startswith("#"):
            skipping = False
        if not skipping:
            kept.append(line)
    return "\n".join(kept).rstrip()
