"""本实验样本记录的统一读取与"分数是怎么来的"分解。

角色
----
消费**本次实验已经落盘的样本 JSON**（``results_root/samples/``）：收尾分析
（``analysis.find_best_eq`` 选发布解）与采样提示注入（``agents.prompt_injection``
要暴露分数分解）都要读同一份记录。内核放在 core 层是因为分层规则不允许
``analysis`` 依赖 ``evaluation``、而 ``agents`` 只能依赖 core/llm/evaluation/knowledge
——两边唯一能共享的落点就是 core。

两个函数各管一件事
------------------
* :func:`load_sample_records` —— **读盘与去重**：两种命名并存、必须都读，同一
  ``sample_order`` 优先取全量文件（细节见其 docstring）；
* :func:`score_breakdown` —— **口径分解**：评分是 ``score = −(拟合 MSE + 体检罚分)``，
  把"拟合得好"与"没被体检罚分"拆开。混为一谈正是要治的病：实测
  ``MRFCompress-Cuboid_20260926-151008`` 的 order 34 拟合 MSE 只有 0.197，却带
  36.06 罚分——只看 MSE 会把它当成胜利，而它的分数比"拟合 MSE 16.9、罚分 0"的
  干净解差一倍多（观测到的"低 MSE 高罚分 ↔ 高 MSE 零罚分"两模态振荡即源于此）。
"""
from __future__ import annotations

import glob
import json
import math
import os
from typing import Sequence


def load_sample_records(results_root: str) -> list[dict]:
    """读 ``samples/`` 下**全部**样本记录（按 sample_order 去重），按分数降序返回。

    两种命名并存、必须都读：``topNN_samples_<order>.json``（Top-K 排行）与
    ``samples_<order>.json``（全量单样本，``persist_all_samples=True`` 时才有）。
    旧实现只 glob ``*_samples_*.json``，而 ``samples_3.json`` **不匹配**该模式——也就是
    说"全量落盘"模式下收尾分析会一个样本都读不到。默认改成全量落盘之前必须先修这条
    （否则跑完 490 次运行的收尾全部静默失效）。

    每条含 ``score`` / ``penalty`` / ``mse`` / ``nmse`` / ``sample_order`` / ``path`` /
    ``function`` / ``params``；同一 sample_order 有两种文件时优先取全量文件。
    """
    records: dict = {}
    for path in sorted(glob.glob(os.path.join(results_root, "samples", "*.json"))):
        name = os.path.basename(path)
        is_top = name.startswith("top")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        score = data.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        try:
            order = int(data.get("sample_order"))
        except (TypeError, ValueError):
            order = None
        key = order if order is not None else name
        prev = records.get(key)
        # 已有同一 sample_order 的记录时，只有"用全量文件替换 Top-K 副本"才覆盖
        if prev is not None and not (prev["is_top"] and not is_top):
            continue
        records[key] = {
            "score": float(score),
            "penalty": data.get("penalty"),
            "mse": data.get("mse"),
            "nmse": data.get("nmse"),
            "sample_order": order,
            "path": path,
            "function": data.get("function", ""),
            "params": data.get("params"),
            "is_top": is_top,
        }
    return sorted(records.values(), key=lambda r: r["score"], reverse=True)


def _finite(value):
    """有限实数（不是 bool）；其余（None / 字符串 / NaN / inf）返回 ``None``。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _row(record: dict) -> dict:
    """单条记录的口径字段（只取 4 个数字，便于直接渲染）。"""
    return {
        "sample_order": record.get("sample_order"),
        "mse": _finite(record.get("mse")),
        "nmse": _finite(record.get("nmse")),
        "penalty": _finite(record.get("penalty")),
        "score": _finite(record.get("score")),
    }


def score_breakdown(records: Sequence[dict]) -> dict:
    """把样本记录汇总成"分数是怎么来的"分解（纯函数，不做 IO）。

    ``penalty`` 字段是 20260926-094330 之后才落盘的（此前 ``mse`` 里含着罚分）。
    旧目录缺该字段时**不能**把罚分当 0：那会把"未知"读成"干净"，于是
    ``penalty_known=False``，渲染方据此显式降级披露而不是给出假的分解。

    Returns:
        dict：``n_scored`` / ``n_penalty_known`` / ``n_clean`` / ``n_penalized`` /
        ``best``（最高分，不论罚分）/ ``best_clean``（罚分为 0 的最高分）/
        ``best_penalized``（带罚分的最高分——干净解最强的对手）/
        ``best_fit``（拟合 MSE 最低，可能带罚分）。
    """
    scored = [r for r in (records or []) if _finite(r.get("score")) is not None]
    breakdown = {
        "n_scored": len(scored),
        "n_penalty_known": 0,
        "n_clean": 0,
        "n_penalized": 0,
        "penalty_known": False,
        "best": None,
        "best_clean": None,
        "best_penalized": None,
        "best_fit": None,
    }
    if not scored:
        return breakdown

    breakdown["best"] = _row(max(scored, key=lambda r: _finite(r["score"])))
    known = [r for r in scored if _finite(r.get("penalty")) is not None]
    breakdown["n_penalty_known"] = len(known)
    breakdown["penalty_known"] = bool(known)
    if not known:
        return breakdown

    breakdown["n_penalized"] = sum(1 for r in known if _finite(r["penalty"]) > 0)
    clean = [r for r in known if _finite(r["penalty"]) == 0]
    penalized = [r for r in known if _finite(r["penalty"]) > 0]
    breakdown["n_clean"] = len(clean)
    if clean:
        breakdown["best_clean"] = _row(max(clean, key=lambda r: _finite(r["score"])))
    if penalized:
        breakdown["best_penalized"] = _row(max(penalized, key=lambda r: _finite(r["score"])))
    with_mse = [r for r in known if _finite(r.get("mse")) is not None]
    if with_mse:
        breakdown["best_fit"] = _row(min(with_mse, key=lambda r: _finite(r["mse"])))
    return breakdown