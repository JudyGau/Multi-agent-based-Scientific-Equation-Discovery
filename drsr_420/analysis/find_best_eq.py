"""收尾分析编排：最佳样本 → 敏感度剪枝与可视化 → 物理解释。

位置
----
``runtime/pipeline.main`` 在实验结束前调用一次 :func:`find_best_eq`（非 Agent，
是普通工具函数——架构图中的"收尾"环节）。

编排（各步骤都拆成了单一职责的模块）
------------------------------------
::

    find_best_eq(results_root)
      ├── select_published_sample()  **病理门禁**：优先取体检罚分==0 的最高分样本
      │     └── load_sample_records()  core.sample_records：两种命名都读，按 order 去重
      │           （采样提示注入读同一份，故内核放 core 层）
      ├── prune_and_visualize()   **先剪枝**（解释要覆盖剪枝结果与剪枝过程）
      │     ├── expr_parse.expr_substitution()   骨架字符串 → SymPy 表达式
      │     ├── sensitivity_prune.SensitivityPruner.prune()  敏感度剪枝
      │     │     └── 没真剪掉项时返回原式（simplify 只做通分/重排，不算剪枝结果）
      │     ├── prune_report.classify_pruning()  判定真剪枝 / 仅形式变化 + 拟合对比
      │     ├── expr_viz.safe_preview() / render_expr_trees()  预览图与树图
      │     │     └── 未实际剪枝时不产出重复的"剪枝后"图件
      │     └── expr_curves.plot_data_curves()  剪枝前后曲线 + 数据点（可失败，仅告警）
      │     └── 返回剪枝摘要 dict（剪掉了哪些项 + 敏感度 + 拟合变化 + 判定结论）
      └── explain.explain_best_sample(pruning=摘要)
            把剪枝前/后表达式、被移除项与拟合数值一起交给解释 LLM → report.md
            （含参考文献清单：知识库检索结果 + 解释过程中的工具检索结果）

**顺序不能反过来**：report.md 必须解释"剪枝后的表达式"并讲清"剪掉了哪些项、为什么
合理"，这两件事都要求剪枝结果先算出来。

本模块只做"取样本 + 步骤编排 + 兜底告警"，具体逻辑见上表各自的模块。
"""
import glob
import os
import re

import numpy as np
import sympy as sp

from drsr_420.analysis.expr_parse import expr_substitution
from drsr_420.analysis.expr_viz import render_expr_trees, safe_preview
from drsr_420.analysis.explain import explain_best_sample
from drsr_420.analysis.holdout import (evaluate_holdout, format_holdout_summary,
                                       load_ood_data, load_test_data,
                                       resolve_ood_csv, resolve_test_csv)
from drsr_420.analysis.prune_report import (classify_pruning, compare_fits,
                                            format_fit_summary, load_training_data,
                                            resolve_columns)
from drsr_420.analysis.sensitivity_prune import SensitivityPruner
from drsr_420.core.sample_records import load_sample_records


def find_best_sample(results_root: str):
    """扫描 samples 目录，返回分数最高的样本 (score, path, func, params)；无则 None。"""
    records = load_sample_records(results_root)
    if not records:
        return None
    best = records[0]
    return best["score"], best["path"], best["function"], best["params"]


def _selection_entry(record: dict) -> dict:
    """选择小节的单条记录（数字截到 6 位，便于直接渲染）。"""
    def _num(value):
        return (round(float(value), 6)
                if isinstance(value, (int, float)) and not isinstance(value, bool) else value)

    return {
        "sample_order": record.get("sample_order"),
        "score": _num(record.get("score")),
        "mse": _num(record.get("mse")),
        "penalty": _num(record.get("penalty")),
        "path": record.get("path"),
    }


def _is_pathological(record: dict) -> bool:
    """该候选是否**确定**携带数值病理（体检罚分 > 0）。罚分未知（None）不算。"""
    penalty = record.get("penalty")
    return (isinstance(penalty, (int, float)) and not isinstance(penalty, bool)
            and penalty > 0)


def select_published_sample(results_root: str) -> tuple[dict | None, dict]:
    """挑"要发布的解"：优先**无病理**（体检罚分 == 0）的最高分样本，并给出选择依据。

    为什么需要这道门禁：评分是 ``-(拟合 MSE + 体检罚分)``，罚分只把病理解**压低**，
    压不到底时它仍可能是分数最高的那个——实测 ``20260925-134149`` 的发布式自认
    "病理性器件"却照样被发布，``20260926-110809`` 的交付解罚分 0.6877 更**大于**它的
    拟合 MSE 0.4253。发布一个门控器件等于用产物打脸"本方法能治病理"。故把"发布什么解"
    与"谁是最高分"解耦：**能给出干净解就给干净解**；被降级的最高分样本连同它的罚分
    写进 report.md 的选择小节，绝不静默丢弃。

    ``penalty`` 为 ``None``（旧产物 / 拿不到 fit_mse）时**既不算干净也不算病理**：
    只在没有任何 ``penalty==0`` 候选时才可能被选中，并在选择小节里注明"罚分未知"。

    Returns:
        ``(record, info)``；无有效样本时 ``(None, {...})``。``info`` 含
        ``n_candidates`` / ``n_clean`` / ``degraded`` / ``best`` / ``chosen`` /
        ``rejected``（分数更高但带病理、被跳过的候选，最多 5 条）。
    """
    records = load_sample_records(results_root)
    if not records:
        return None, {"n_candidates": 0, "n_clean": 0, "degraded": False,
                      "best": None, "chosen": None, "rejected": [], "n_rejected": 0,
                      "n_unknown_skipped": 0}
    best = records[0]
    clean = [r for r in records if r["penalty"] == 0]
    chosen = clean[0] if clean else best
    skipped = [r for r in records if r["score"] > chosen["score"]]
    # "被跳过"分两类，报告里必须分开写：真带病理（罚分 > 0）与**罚分未知**（旧产物）
    rejected = [r for r in skipped if _is_pathological(r)]
    unknown = [r for r in skipped if r["penalty"] != 0 and not _is_pathological(r)]
    info = {
        "n_candidates": len(records),
        "n_clean": len(clean),
        "degraded": chosen is not best,
        "best": _selection_entry(best),
        "chosen": _selection_entry(chosen),
        "rejected": [_selection_entry(r) for r in rejected[:5]],
        "n_rejected": len(rejected),
        "n_unknown_skipped": len(unknown),
    }
    return chosen, info


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


def _training_points(data, dependent: str, sym_names: list[str]) -> list | None:
    """取训练数据的自变量列作为敏感度采样的补充点；取不到返回 None。

    返回与 ``sym_names`` 顺序一致的列数组列表（与 ``ExpressionEvaluator.points``
    同构）；数据缺失或列名对不上（``KeyError``）时返回 None，由调用方告警降级。
    """
    if data is None:
        return None
    try:
        _dep, ind_cols, _note = resolve_columns(data, dependent, sym_names)
        return [np.asarray(data[c], dtype=float) for c in ind_cols]
    except KeyError as e:
        print(f"[WARN] 训练数据列对不上，敏感度采样不含数据点: {e}")
        return None


def prune_and_visualize(results_root: str, func: str, params,
                        threshold: float, sample_range: tuple,
                        test_csv: str | None = None,
                        test_ood_csv: str | None = None) -> dict | None:
    """基于敏感度分析剪枝最优公式，保存表达式预览图与表达式树图，返回剪枝摘要。

    返回值是给 ``explain`` 用的剪枝摘要（含剪枝前/后表达式、被移除项及其敏感度、
    剪枝统计、剪枝前后在训练数据上的拟合对比、样本外验证指标）；解析/剪枝失败时
    返回 ``None``，调用方据此让解释 LLM 知道"本次没有剪枝结果"。

    ``test_csv`` 给定时（见 ``holdout.resolve_test_csv``）额外做样本外验证：在没参与
    拟合/打分/选择的 held-out 点上算 MSE/NMSE/最大误差，**只报告**。公式的对外发布
    形式在这里确定，样本外验证因而也放在这里（保证验证的就是最终报告的那个公式）。
    """
    parsed = _parse_symbols(func)
    if parsed is None:
        print("[WARN] 无法从样本中解析 Dependent/Independents，跳过剪枝。")
        return None
    dependent, sym_names = parsed
    symbols = sp.symbols(sym_names)

    # 训练数据点并入敏感度采样（必须在剪枝判定之前加载）：均匀随机撒点对"只在
    # 个别数据点承重"的项是盲的——实测 20260921-161549 最优样本的 (1,1) 角点锚
    # 2989.9/(λ12λ23)^126.082 仅在该训练点非零，随机点敏感度≈0 被剪，训练 MSE
    # 0.25 → 1.1e6；数据点参与采样后该项敏感度≈15，正确保留。
    data = load_training_data(results_root)
    data_points = _training_points(data, dependent, sym_names)
    if data_points is None:
        print("[WARN] 敏感度采样不含训练数据点（数据缺失或列对不上），"
              "仅用随机采样点做敏感度判据。")

    pruner = SensitivityPruner(symbols=symbols, threshold=threshold,
                               sample_range=sample_range, extra_points=data_points)
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

    # 判定这次"剪枝"的实质：真剪掉了项 / 只是 simplify 换了写法（通分）/ 什么都没做。
    # 「没真正剪掉项就沿用原式」的策略在 SensitivityPruner.prune 里执行，这里只把证据
    # 算出来——同一份判据同时给控制台、图件选择与 explain 提示词用，避免三处各判一次。
    verdict = None
    if pruned_expr is not None:
        verdict = classify_pruning(expr, pruned_expr, pruner.stats,
                                   sym_names=sym_names, sample_range=sample_range,
                                   extra_points=data_points)
        print(f"[PRUNE] {verdict['summary']}")
    actually_pruned = bool(verdict and verdict["actually_pruned"])
    # 对外发布的表达式：真剪枝才用剪枝结果，否则一律是剪枝前的原式
    published = pruned_expr if actually_pruned else expr

    if actually_pruned:
        # n(2) 仅用于控制台打印/预览图的观感；曲线绘制与解释必须用全精度表达式
        # （2 位有效数字会在 ~2000 量级的项上引入 ±30 的偏差，见 expr_parse 的教训）
        display_expr = pruned_expr.n(2)
        print(f"剪枝后的表达式为 {dependent} =")
        sp.pprint(display_expr)
        safe_preview(display_expr, f'{results_root}/prunedExpr.png')
    else:
        # 未实际剪枝：不再产出与 expr.png 重复（甚至更啰嗦）的"剪枝后"图件，
        # 曲线图也只画一条并在图注里注明本次未剪枝。
        # （剪枝失败的情形上面已有 [WARN]，不再重复同一句话。）
        if pruned_expr is not None:
            print("[PRUNE] 本次未实际剪枝：不生成 prunedExpr.png / pruned_expr_tree，"
                  "曲线图只画一条并注明本次未剪枝。")

    render_expr_trees(results_root, expr, pruned_expr if actually_pruned else None)

    # 剪枝前后在训练数据上的拟合对比：解释 LLM 要靠它论证"剪掉这些项是否合理"。
    # 未实际剪枝时 published == expr，对比结果自然是"逐点完全相同（剪枝未改变模型）"。
    # （data 已在剪枝前加载并用于敏感度采样，此处直接复用。）
    fit = compare_fits(dependent, sym_names, data, expr, published)
    print(f"[PRUNE] {format_fit_summary(fit)}")

    # 动态范围体检（对**最终发布**的表达式）：检测角点钉扎/下溢尖峰类病理解。
    # 训练点 MSE 看不见点与点之间的行为——体检在包围盒网格（含角点壳层）上评估，
    # 与评分器（evaluation/problems.evaluate）同一判据（内核在 core.range_check，
    # 两层共用），结果进 explain 提示词与 report.md 权威小节。
    range_info = None
    try:
        from drsr_420.core.range_check import dynamic_range_check
        _dep_col, ind_cols, _note = resolve_columns(data, dependent, sym_names)
        _X = np.column_stack([np.asarray(data[c], dtype=float) for c in ind_cols])
        _y = np.asarray(data[_dep_col], dtype=float)
        f_pub = sp.lambdify(sym_names, published, modules="numpy")
        # 判据三（大系数抵消）要"换一组参数再算一次"。**不要**用 SymPy 把参数符号化：
        # 实测最优样本常用 ``c0, c1, ... = params[:8]`` 这类元组解包，expr_substitution
        # 解析不了 → 探针建不出来 → 判据三静默弃权（20260925-134149 的 order 83 就是
        # 这种写法）。直接 exec 样本自带的 def（与评估器调用样本的方式一致）最稳。
        probe_fn = None
        try:
            match_def = re.search(r"^def\s+\w+\s*\(", func, re.M)
            if match_def:
                namespace = {"np": np}
                exec(func[match_def.start():], namespace)   # noqa: S102 - 执行的是实验自己选出的样本
                _fn = next(v for k, v in namespace.items()
                           if callable(v) and not k.startswith("__"))
                probe_fn = (lambda *args: np.asarray(
                    _fn(*args[:-1], np.asarray(args[-1], dtype=float))))
        except Exception as e:
            print(f"[WARN] 判据三的参数探针构造失败（本次跳过该判据）: {e}")
        range_info = dynamic_range_check(_X, _y, f_pub, params=params, probe_fn=probe_fn)
        if range_info["penalty"] > 0:
            hits = []
            if range_info.get("span_penalty"):
                hits.append(f"输出跨度={range_info['span_ratio']:.4g}"
                            f"(上限{range_info['limit']})")
            if range_info.get("slope_penalty"):
                hits.append(f"局部斜率={range_info['slope_max']:.4g}"
                            f"(上限{range_info['slope_limit']})")
            if range_info.get("coef_penalty"):
                hits.append(f"系数抵消={range_info['coef_ratio']:.4g}"
                            f"(上限{range_info['coef_limit']})")
            print(f"[RANGE] 动态范围体检：**病理性** 命中 {'；'.join(hits)}"
                  f"——发布公式携带角点钉扎/尖峰/大系数抵消类器件，详见 report.md")
        else:
            coef = range_info.get("coef_ratio")
            coef_txt = ("未评估" if coef is None
                        else f"{coef:.3g} ≤ {range_info['coef_limit']}")
            print(f"[RANGE] 动态范围体检：未检出（输出跨度="
                  f"{range_info['span_ratio']:.3g} ≤ {range_info['limit']}；"
                  f"局部斜率={range_info['slope_max']:.3g} ≤ "
                  f"{range_info['slope_limit']}；系数抵消={coef_txt}）")
    except Exception as e:
        print(f"[WARN] 动态范围体检失败（跳过，不阻塞收尾）: {e}")
        range_info = None

    # 剪枝完成 → 剪枝前后表达式曲线 + 数据点（每个自变量一幅，人工检查贴合度）。
    # 与 expr.png 同级别的"给人看"产物：失败只告警，不拖垮收尾流程。
    try:
        from drsr_420.analysis.expr_curves import plot_data_curves
        plot_data_curves(results_root, dependent, sym_names, expr,
                         published if actually_pruned else None,
                         test_csv=test_csv)
    except Exception as e:
        print(f"[WARN] 剪枝前后曲线图生成失败（跳过）: {e}")

    # 训练进度：MSE 随 sample_order 的变化（历史最优刷新点）。同样是"给人看"的产物，
    # 数据只取自 best_history/*.json（不解析 run.out——.bat/.sh 并不重定向 stdout，
    # run.out 不是每条启动路径都有的产物）；没有记录时返回 None，报告侧跳过该小节。
    try:
        from drsr_420.analysis.progress_curve import plot_progress_curve
        progress = plot_progress_curve(results_root)
    except Exception as e:
        print(f"[WARN] 训练进度图生成失败（跳过）: {e}")
        progress = None

    # 样本外验证：在没参与拟合/打分/选择的 held-out 点上评估**最终发布的**公式。
    # 只报告，不回灌评分——一旦参与选择，它就不再是 held-out（见 holdout 模块说明）。
    # **ID（同分布）与 OOD（分布外）分开评估、分开报告**：论文要求两者分开报，
    # 合成一个数字看不出外推是否失效。自动探测：test.csv/test_id.csv ↔
    # ood_test.csv/test_ood.csv（见 holdout.resolve_ood_csv）。
    train_var = None
    if data is not None:
        try:
            train_dep, _train_ind, _ = resolve_columns(data, dependent, sym_names)
            train_var = float(np.var(np.asarray(data[train_dep], dtype=float)))
        except KeyError:
            train_var = None

    def _eval_holdout(test_data, explicit, resolver):
        if test_data is None:
            return None
        return evaluate_holdout(dependent, sym_names, published, test_data,
                                train_var=train_var,
                                path=(resolver(results_root, explicit) or ""),
                                train_data=data)

    holdout = _eval_holdout(load_test_data(results_root, test_csv), test_csv,
                            resolve_test_csv)
    holdout_ood = _eval_holdout(load_ood_data(results_root, test_ood_csv), test_ood_csv,
                                resolve_ood_csv)
    print(f"[HOLDOUT] {format_holdout_summary(holdout, fit)}")
    if holdout_ood is not None:
        print(f"[HOLDOUT-OOD] {format_holdout_summary(holdout_ood, fit)}")

    return {
        "dependent": dependent,
        "sym_names": list(sym_names),
        "threshold": threshold,
        "sample_range": tuple(sample_range),
        "substituted_expr": sp.sstr(expr),
        # pruned_expr 是**对外发布**的最终表达式：未实际剪枝时它就是剪枝前的原式
        # （下游 explain 的 `after == before` 分支据此说明"与剪枝前完全相同"）。
        "pruned_expr": sp.sstr(published) if pruned_expr is not None else None,
        "used_original": not actually_pruned,
        "verdict": verdict,
        # 诊断用：0 项剪枝时 simplify 会给出的形式（未被采用，仅说明"只是换写法"）
        "simplify_expr": (sp.sstr(pruner.stats.simplified_expr)
                          if pruner.stats.simplified_expr is not None else None),
        "nodes_visited": pruner.stats.nodes_visited,
        "nodes_pruned": pruner.stats.nodes_pruned,
        "prune_rate": pruner.stats.prune_rate,
        "ops_before": pruner.stats.ops_before,
        "ops_after": pruner.stats.ops_after,
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
        # 动态范围体检（对发布式）：检测角点钉扎/下溢尖峰类病理解；None=体检失败
        "range_check": range_info,
        # 样本外验证（held-out）：只报告，不参与任何选择。ID 与 OOD 分开。
        "holdout": holdout,
        "holdout_ood": holdout_ood,
        # 训练进度（历史最优刷新点）：只报告；None=没有 best_history 记录
        "progress": progress,
    }


def find_best_eq(results_root: str, threshold: float = 0.1,
                 sample_range: tuple = (1, 14), role_clients=None,
                 test_csv: str | None = None,
                 test_ood_csv: str | None = None):
    """收尾：寻找最优样本 → 敏感度剪枝与可视化 → 生成物理解释（含剪枝分析）。

    主函数仅做扁平编排，具体逻辑拆分到 select_published_sample / prune_and_visualize /
    explain_best_sample，避免原先 try-with-for-if-try 的深嵌套。

    **发布解的选择带病理门禁**：候选里优先取体检罚分 == 0 的最高分样本；最高分样本带病理
    时把它降级为"不建议采用"并写进 report.md 的「发布解选择」小节（见
    :func:`select_published_sample`）。

    Args:
        role_clients: ``llm.roles.RoleClients``；物理解释按其中的 ``explain`` 角色
            取客户端。省略时由 ``explain`` 模块自行按注册表解析（因此直接调用
            本函数也能拿到正确档案，不再依赖硬编码文件名）。
        test_csv: **同分布（ID）** held-out 数据路径；``None`` 表示自动探测（见
            ``holdout.resolve_test_csv``），``"none"`` 表示关闭。样本外指标只写进
            run.out 与 report.md，不参与采样/打分/选择。
        test_ood_csv: **分布外（OOD）** held-out 数据路径；``None`` 表示自动探测
            （``ood_test.csv`` / ``test_ood.csv``，见 ``holdout.resolve_ood_csv``）。
            ID 与 OOD 分开报告。
    """
    chosen, selection = select_published_sample(results_root)
    if chosen is None:
        print("没有找到有效样本。")
        return

    score, path, func, params = (chosen["score"], chosen["path"],
                                 chosen["function"], chosen["params"])
    print(f"[BEST] score={score} file={path}")
    best_entry, chosen_entry = selection.get("best"), selection.get("chosen")
    if selection.get("degraded"):
        # 病理门禁命中：最高分样本带体检罚分，改发布无病理的最高分样本。
        print(f"[GATE] 最高分样本 order={best_entry['sample_order']}"
              f"（score={best_entry['score']}，拟合 MSE={best_entry['mse']}，"
              f"体检罚分={best_entry['penalty']}）携带数值病理，"
              f"按「优先发布无病理解」口径改为发布 order={chosen_entry['sample_order']}"
              f"（score={chosen_entry['score']}，罚分={chosen_entry['penalty']}）")
    elif not selection.get("n_clean"):
        print(f"[GATE] 本次 {selection.get('n_candidates')} 个候选全部带病理或罚分未知，"
              f"没有可推荐的干净解；发布式仍取最高分样本，report.md 已显式标注")

    # 先剪枝：report.md 要解释"剪枝后的表达式"与"剪掉了哪些项、为什么合理"，
    # 剪枝摘要（含剪枝前后在训练数据上的拟合对比）必须先算出来。
    pruning = prune_and_visualize(results_root, func, params, threshold, sample_range,
                                  test_csv=test_csv, test_ood_csv=test_ood_csv)
    if pruning is not None:
        # 选择依据随剪枝摘要一起进 explain（渲染成 report.md 的「发布解选择」小节）：
        # 被跳过的病理解候选必须留在报告里，否则"为什么发布的不是最高分"无法追溯。
        pruning["selection"] = selection

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
    # 手工排查用：
    #   python -m drsr_420.analysis.find_best_eq [实验目录]
    #       [--test_csv <路径>|none] [--test_ood_csv <路径>|none]
    # 不给路径时取 experiments/ 下最近修改的一次 run（见 _latest_run_dir）。
    import sys

    argv = sys.argv[1:]

    def _take(flag: str):
        if flag in argv:
            i = argv.index(flag)
            value = argv[i + 1] if i + 1 < len(argv) else "none"
            del argv[i:i + 2]
            return value
        return None

    explicit_test = _take("--test_csv")
    explicit_ood = _take("--test_ood_csv")
    target = argv[0] if argv else _latest_run_dir()
    if target is None:
        print("[WARN] 未找到任何实验目录，请显式给出路径。")
    else:
        print(f"[INFO] 收尾分析目标: {target}")
        find_best_eq(target, test_csv=explicit_test, test_ood_csv=explicit_ood)
