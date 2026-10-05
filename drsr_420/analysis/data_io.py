"""收尾分析的**数据定位与读取**：实验目录 → CSV / JSON → 对齐后的列。

角色归属
--------
 ``analysis`` 层的共享 I/O 内核。此前这些函数住在 :mod:`drsr_420.analysis.prune_eval`
里，于是名字叫"剪枝评估"的模块成了事实上的公共工具库——``holdout``、``expr_curves``
都要 import 它的私有名（``_warn_once``）。本模块把"取数据"从"算剪枝"里分出来，
两件事各自有名字。

为什么统一去重
--------------
一次收尾里 ``load_training_data`` / ``compare_fits`` / 曲线 / 样本外验证会各调一遍，
同一句话不合并会刷 4 遍（``run.out`` 已经很长）。去重集合 :data:`_warned` 是模块级
全局——**这是有意的**：同一目录的同一句告警无论谁来问都只提示一次。

失败策略
--------
一律"返回 ``None`` + 由调用方决定是否告警"：本模块只做 I/O，不吞异常也不代替调用方
说话（:func:`read_snapshot` 把异常对象连同结果一并返回，调用方用自己的措辞打印）。
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

#: 仓库根（``prune_eval``/``holdout`` 里的相对路径解析都以它为兜底）。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: 实验目录名 ``<问题名>_<YYYYMMDD-HHMMSS>``：没有 config_snapshot.json 的历史目录
#: 靠它反推数据集（见 :func:`infer_data_csv`）。
_RUN_DIR_RE = re.compile(r"^(?P<problem>.+)_\d{8}-\d{6}$")

#: 已提示过的信息：同一条在整个进程里只打印一次（见模块 docstring）。
_warned: set[str] = set()


def warn_once(message: str) -> None:
    """打印 ``[WARN] message``，同一条只打印一次（跨模块共享）。"""
    if message not in _warned:
        _warned.add(message)
        print(f"[WARN] {message}")


def reset_warned() -> None:
    """清空去重集合（测试用：让同一条告警可以在新用例里重新出现）。"""
    _warned.clear()


# ── 路径定位 ────────────────────────────────────────────────────

def resolve_csv(data_csv: str, results_root: str = "") -> str | None:
    """data_csv 依次按 results_root、项目根、cwd 解析；兼容绝对路径。

    config_snapshot 里通常存项目根相对路径（``./data/...``），自包含实验目录
    （如测试夹具）则是 results_root 相对路径——两处都要试。

    返回值统一 ``normpath``：快照里的 ``./data/X/train.csv`` 与目录拼接后会得到
    ``…\\./data/X\\train.csv`` 这种混合分隔符，直接进日志/产物说明很难看。
    """
    if os.path.isabs(data_csv):
        return os.path.normpath(data_csv) if os.path.isfile(data_csv) else None
    for base in (results_root, _REPO_ROOT, os.getcwd()):
        path = os.path.join(base, data_csv)
        if os.path.isfile(path):
            return os.path.normpath(path)
    return None


def infer_data_csv(results_root: str) -> str | None:
    """没有 config_snapshot.json 时按目录名推断训练数据：
    ``<问题名>_<时间戳>`` → ``data/<问题名>/train.csv``。

    历史实验目录（用户从别处整理进来的那批）没有 config_snapshot.json，旧实现只能
    放弃——剪枝前后拟合对比与曲线图全部静默缺失。目录名里的问题名唯一，且 ``data/``
    下的数据集是**约定命名**，因此按它兜底能把这批目录重新变成可分析的。

    推断结果只作兜底并会在日志里明说：它终究是"猜"的，不如快照里的记录可靠。
    """
    match = _RUN_DIR_RE.match(os.path.basename(os.path.normpath(results_root)))
    if not match:
        return None
    candidate = os.path.join(_REPO_ROOT, "data", match.group("problem"), "train.csv")
    if os.path.isfile(candidate):
        warn_once(f"目录里没有 config_snapshot.json，按目录名推断训练数据: {candidate}")
        return candidate
    return None


# ── JSON 读取 ──────────────────────────────────────────────────

def read_json_file(path: str) -> tuple[dict | None, Exception | None]:
    """读一个 JSON 对象文件，返回 ``(字典 or None, 异常 or None)``。

    文件不存在返回 ``(None, None)``——"没有"不是错误。解析失败/不是对象时返回
    ``(None, e)``，由调用方用自己的措辞告警（本函数不打印）。
    """
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return None, e
    if not isinstance(data, dict):
        return None, ValueError(f"{path} 不是 JSON 对象")
    return data, None


def read_snapshot(results_root: str) -> tuple[dict | None, Exception | None]:
    """读 ``<results_root>/config_snapshot.json``；语义同 :func:`read_json_file`。"""
    return read_json_file(os.path.join(results_root, "config_snapshot.json"))


def snapshot_value(results_root: str, key: str) -> str:
    """读 config_snapshot.json 里的某个字段（读不到一律返回空串）。

    ``test_csv`` 兼容两种写法：``test_csv``（生效值）与早期的 ``test_csv_arg``
    （命令行原值）。
    """
    snapshot, _error = read_snapshot(results_root)
    value = (snapshot or {}).get(key)
    if not value and key == "test_csv":
        value = (snapshot or {}).get("test_csv_arg")
    return str(value or "")


# ── CSV 读取 ──────────────────────────────────────────────────

def load_struct_csv(path: str | None, label: str) -> np.ndarray | None:
    """把 CSV 读成结构化数组；路径为空 / 读取失败 / 无表头时返回 ``None``。

    ``label`` 只用于告警文案（"样本外数据" / "分布外数据"…）。
    """
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


def load_training_data(results_root: str) -> np.ndarray | None:
    """按 config_snapshot.json 的 data_csv 读取训练数据（结构化数组）；失败返回 None。

    快照缺失或没写 data_csv 时退回 :func:`infer_data_csv` 按目录名推断（历史目录）。
    """
    snapshot, error = read_snapshot(results_root)
    if error is not None:
        warn_once(f"读取 config_snapshot.json 失败: {error}")
    data_csv = (snapshot or {}).get("data_csv")

    if not data_csv:
        data_csv = infer_data_csv(results_root)
    if not data_csv:
        print("[WARN] config_snapshot.json 里没有 data_csv，无法定位训练数据。")
        return None
    csv_path = resolve_csv(data_csv, results_root)
    if csv_path is None:
        print(f"[WARN] 数据文件不存在: {data_csv}")
        return None
    data = load_struct_csv(csv_path, "训练数据")
    return data


# ── 列名对齐 ──────────────────────────────────────────────────

def resolve_columns(data: np.ndarray, dependent: str,
                    sym_names: list[str]) -> tuple[str, list[str], str]:
    """把函数头里的变量名对齐到 CSV 的实际列名。

    函数头里的名字是 LLM 写的，未必与 CSV 表头逐字相同（实测历史运行的因变量写成
    ``um``，而 CSV 列是 ``miu``）——旧实现直接按名字取列，取不到就静默跳过拟合对比与
    曲线（``compare_fits`` 返回空字典、``plot_data_curves`` 直接 return）。

    匹配顺序：

    1. 精确匹配；
    2. 忽略大小写（``Sigma`` / ``sigma``）；
    3. 因变量兜底：自变量都能对上时，取"自变量之外的唯一一列"（CSV 约定最后一列是因变量）。

    三档都不成立时抛 ``KeyError``（调用方只告警跳过，不猜）。

    Returns:
        ``(因变量列名, 自变量列名列表, 说明)``；说明非空表示发生了兜底匹配，供日志明说
        用了哪一列（不能悄悄换列）。
    """
    names = list(getattr(getattr(data, "dtype", None), "names", None) or ())
    lower = {name.lower(): name for name in names}
    notes: list[str] = []

    def _match(name: str, kind: str) -> str | None:
        if name in names:
            return name
        hit = lower.get(str(name).lower())
        if hit is not None:
            notes.append(f"{kind} {name!r} → CSV 列 {hit!r}（忽略大小写）")
        return hit

    ind_cols: list[str] = []
    for sym in sym_names:
        col = _match(sym, "自变量")
        if col is None:
            raise KeyError(f"数据里没有自变量列 {sym!r}（现有列：{names}）")
        ind_cols.append(col)

    dep_col = _match(dependent, "因变量")
    if dep_col is None:
        rest = [name for name in names if name not in ind_cols]
        if len(rest) == 1:
            dep_col = rest[0]
            notes.append(f"因变量 {dependent!r} → CSV 列 {dep_col!r}"
                         f"（按位置兜底：自变量之外的唯一一列）")
        else:
            raise KeyError(f"数据里没有因变量列 {dependent!r}（现有列：{names}）")
    return dep_col, ind_cols, "；".join(notes)


__all__ = [
    "warn_once", "reset_warned", "resolve_csv", "infer_data_csv",
    "read_json_file", "read_snapshot", "snapshot_value",
    "load_struct_csv", "load_training_data", "resolve_columns",
]