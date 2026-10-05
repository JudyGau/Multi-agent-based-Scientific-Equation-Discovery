"""SymPy 表达式的**数值化**：编译成 NumPy 可调用对象，或在给定点上一次求值。

角色归属
--------
``analysis`` 层的共享数值内核。此前 ``lambdify`` + ``errstate`` + 标量广播这套
样板在 4 个模块里各写了一遍（``prune_report._evaluate``、``holdout.evaluate_holdout``、
``expr_curves.plot_data_curves``、``find_best_eq.prune_and_visualize`` 内联探针）。
样板重复本身没多大事，代价在于**口径会漂**：某处补了标量广播、另一处忘了，于是
"剪枝后表达式退化成常数"这类情形只有一半的调用方能算出来。

不打印
------
本模块**不打印**：失败连同异常对象一起返回（与 :func:`drsr_420.analysis.data_io.read_json_file`
同一约定），由调用方用自己的措辞告警——"表达式数值化失败"与"样本外表达式求值失败"
是两句不同的话，不该由内核统一，但异常正文必须保留（它是唯一的诊断线索）。内核管算，
调用方管说。

与 :mod:`drsr_420.analysis.expr_evaluation` 的分工
-------------------------------------------------
那个模块是**敏感度剪枝专用**的求值器：带 ``repr(expr)`` 缓存与逐点 ``subs`` 兜底
（贪心剪枝要反复求值同类父节点），契约不同，故不复用本模块。本模块只服务
"把一条表达式在若干点上算一次"这类收尾调用。
"""
from __future__ import annotations

import numpy as np
import sympy as sp

#: 求值时的数值告警一律静音：剪枝前后的表达式都可能在被修剪的项上溢出，
#: 这里是"评估"而不是"优化"，没有清洗残差的必要，但也没必要刷警告。
_NUMERIC_ERRORS = dict(over="ignore", invalid="ignore", divide="ignore")


def compile_expr(expr, sym_names) -> "callable":
    """把 SymPy 表达式编译成 ``f(*列数组)`` 的可调用对象（NumPy 后端）。

    需要跨多次调用复用同一个可调用对象时用它（例如按自变量逐幅画曲线）；
    只算一次用 :func:`lambdify_eval`。
    """
    return sp.lambdify(list(sp.symbols(sym_names)), expr, modules="numpy")


def lambdify_eval(expr, sym_names, args) -> tuple[np.ndarray | None, Exception | None]:
    """在 ``args``（按 ``sym_names`` 顺序的列数组）上求值表达式。

    Returns:
        ``(结果 or None, 异常 or None)``；成功时异常为 ``None``。

    常量表达式（``lambdify`` 出来是 0 维标量）按点数广播成 1 维：否则下游的
    ``vals[mask]`` 会抛 ``IndexError: too many indices for array``——剪枝结果退化
    成常数时走的正是这条路（``prune_report`` 要判定的那种情形）。
    """
    try:
        func = compile_expr(expr, sym_names)
        with np.errstate(**_NUMERIC_ERRORS):
            values = np.asarray(func(*args), dtype=float)
    except Exception as e:
        return None, e
    if values.ndim == 0:
        length = len(args[0]) if args else 1
        values = np.full(length, float(values))
    return values, None


__all__ = ["compile_expr", "lambdify_eval"]