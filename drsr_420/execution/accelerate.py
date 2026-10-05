# This file aims to accelerate the original evaluate logic using 'numba' package.
# You should install numba package in your Python environment or the later evaluation will fail.

import ast


def add_numba_decorator(
        program: str,
        function_to_evolve: str,
) -> str:
    """
    Accelerates code evaluation by adding @numba.jit() decorator to the target function.

    Note: Not all NumPy functions are compatible with Numba acceleration.

    Example:
    Input:  def func(a: np.ndarray): return a * 2
    Output: @numba.jit()
            def func(a: np.ndarray): return a * 2
    """
    # parse to syntax tree
    tree = ast.parse(program)

    # check if 'import numba' already exists
    numba_imported = False
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == 'numba' for alias in node.names):
            numba_imported = True
            break

    # add 'import numba' to the top of the program
    if not numba_imported:
        import_node = ast.Import(names=[ast.alias(name='numba', asname=None)])
        tree.body.insert(0, import_node)

    # traverse the tree, and find the function_to_run
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_to_evolve:
            # the @numba.jit() decorator instance
            decorator = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='numba', ctx=ast.Load()),
                    attr='jit',
                    ctx=ast.Load()
                ),
                args=[],  
                keywords=[ast.keyword(arg='nopython', value=ast.Constant(value=True))]  
            )
            # add the decorator to the decorator_list of the node
            node.decorator_list.append(decorator)

    # turn the tree to string and return
    modified_program = ast.unparse(tree)
    return modified_program


def try_add_numba_decorator(
        program: str,
        function_to_evolve: str,
        sample_args: tuple,
) -> str:
    """尝试为 `function_to_evolve` 添加 @numba.jit(nopython=True) 装饰。

    numba 对 LLM 生成的任意方程并不总是兼容（不支持的 numpy 操作、复杂控制流等）。
    这里用一组与真实调用 `equation(*X.T, params)` 同构的样例参数触发一次 JIT 编译，
    编译失败或首次调用抛异常时自动降级返回原始程序，避免整条样本因加速而失败。

    Args:
        program: 待执行的完整程序。
        function_to_evolve: 需要加速的方程函数名。
        sample_args: 触发编译的样例参数（含全量输入行与一组占位参数）。
    """
    try:
        accelerated = add_numba_decorator(program, function_to_evolve)
        namespace = {}
        exec(accelerated, namespace)
        fn = namespace[function_to_evolve]
        fn(*sample_args)  # 触发 JIT 编译，验证 numba 兼容性
        return accelerated
    except Exception as e:
        print(f"[WARN] numba 加速不可用，降级为非加速评估（{e}）")
        return program


if __name__ == '__main__':
    code = '''
import numpy as np
import numba

def func1():
    return 3

def func():
    return 5
    '''
    res = add_numba_decorator(code, 'func')
    print(res)
