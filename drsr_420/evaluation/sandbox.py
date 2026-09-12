"""评估执行机制：程序编译、常驻子进程沙箱、残差采样。

从 ``agents/evaluator_agent.py`` 拆出（原文件 516 行里混杂 6 类职责）：
``EvaluatorAgent`` 只保留"评估者"这个**角色**的编排逻辑，与"怎么执行"解耦。

内容：

* ``Sandbox`` / ``LocalSandbox``：常驻 worker 进程池 + 超时/崩溃重建 + numba 可选加速
* ``_eval_worker`` / ``_run_evaluation_task``：worker 侧任务函数
* ``_sample_to_program`` / ``_trim_function_body`` / ``_FunctionLineVisitor`` /
  ``_calls_ancestor``：骨架 → 可运行程序的编译与校验
* ``_sample_residuals``：残差矩阵采样

``agents/evaluator_agent`` 仍 re-export 上述全部名字，历史导入路径不变。
"""
from __future__ import annotations

import ast
import copy
import multiprocessing
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from drsr_420.core import code_manipulation
from drsr_420.evaluation import accelerate as evaluator_accelerate
from drsr_420.evaluation import problems as evaluate_on_problems

class _FunctionLineVisitor(ast.NodeVisitor):
    """ Visitor that finds the last line number of a function with a given name."""

    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None:
        """ Collect the end line number of the target function."""
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        """ Line number of the final line of function `target_function_name`."""
        assert self._function_end_line is not None
        return self._function_end_line

def _trim_function_body(generated_code: str) -> str:
    """ Extract the body of the generated function, trimming anything after it.
    Please note that the indentation is REQUIRED !!!
    """
    if not generated_code:
        return ''

    generated_code = code_manipulation.sanitize_code_text(generated_code)
    code = f'def fake_function_header():\n{generated_code}'

    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)

        except SyntaxError as e:
            if e.lineno is None: # Nothing could be saved when syntaxError
                return ''
            code = '\n'.join(code.splitlines()[:e.lineno - 1])

    if not code:
        return ''

    visitor = _FunctionLineVisitor('fake_function_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    return '\n'.join(body_lines) + '\n\n'

def _sample_to_program(
        generated_code: str,
        version_generated: int | None,
        template: code_manipulation.Program,
        function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
    """
    Return the compiled generated function and the full runnable program.
    This function removes the content after the generated function body.
    """
    body = _trim_function_body(generated_code)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            code=body,
            source_name=f'{function_to_evolve}_v{version_generated}',
            target_name=function_to_evolve
        )

    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body

    return evolved_function, str(program)

class Sandbox(ABC):
    """ Sandbox for executing generated code. """

    @abstractmethod
    def run(
            self,
            program: str,
            function_to_run: str,
            function_to_evolve: str,
            inputs: Any,
            test_input: str,
            timeout_seconds: int,
    ) -> tuple[tuple[Any, bool, str], Any]:

        """ Return `function_to_run(test_input)` and whether execution succeeded. """
        raise NotImplementedError(
            'Must provide a sandbox for executing untrusted code.')

def _eval_worker(task_queue: multiprocessing.Queue, worker_id: int) -> None:
    """常驻评估 worker：循环从任务队列取任务，结果经任务自带的管道送回主进程。"""
    while True:
        task = task_queue.get()
        if task is None:
            return
        (program, function_to_run, function_to_evolve, dataset,
         numba_accelerate, eval_config, warm_start, conn) = task
        try:
            out = _run_evaluation_task(
                program, function_to_run, function_to_evolve, dataset,
                numba_accelerate, eval_config, warm_start)
        except Exception as e:  # 兜底：_run_evaluation_task 内部已捕获，这里防御 worker 意外崩溃
            out = (None, None, False, f'Execution Error: {e}', None)
        try:
            conn.send(out)
        except (BrokenPipeError, EOFError):
            # 主进程可能已因超时关闭管道并重建 worker，丢弃该结果即可
            pass
        finally:
            conn.close()

def _sample_residuals(full_res, sample_size: int):
    """从完整残差矩阵中随机采样至多 sample_size 行；full_res 为空时返回 None。"""
    if full_res is None or not hasattr(full_res, 'shape') or len(full_res) == 0:
        return None
    n = min(sample_size, len(full_res))
    indices = np.random.choice(len(full_res), n, replace=False)
    return full_res[indices]

def _run_evaluation_task(program, function_to_run, function_to_evolve, dataset,
                         numba_accelerate, eval_config, warm_start):
    """在 worker 进程中执行一条样本，返回统一 5 元组：
    (grade, res, runs_ok, remark, optimized_params)。"""
    res = None
    opt_params = None
    try:
        program = code_manipulation.sanitize_code_text(program)
        # numba 加速（可选）：编译失败或方程不受支持时自动降级为原始程序。
        # 探测只需验证"可编译可运行"，用小切片（numba njit 按 dtype 而非 shape
        # 特化）：旧实现拿整个数据集执行探测，每个样本都完整跑两遍方程，
        # 白白吃掉 evaluate_timeout_seconds 预算，把本可及格的样本拖成超时。
        if numba_accelerate:
            X = np.atleast_2d(dataset['inputs'])
            n_params = eval_config.get('n_params', evaluate_on_problems.MAX_NPARAMS)
            probe_rows = min(64, X.shape[0])
            sample_args = tuple(X[:probe_rows].T) + (np.ones(n_params),)
            program = evaluator_accelerate.try_add_numba_decorator(
                program, function_to_evolve, sample_args)

        # 执行程序，把方程函数放入全局命名空间
        all_globals_namespace = {}
        exec(program, all_globals_namespace)
        evolved_function = all_globals_namespace[function_to_evolve]

        results, full_res, opt_params = evaluate_on_problems.evaluate(
            dataset,
            evolved_function,
            n_params=eval_config.get('n_params', evaluate_on_problems.MAX_NPARAMS),
            decimal_places=eval_config.get('decimal_places', evaluate_on_problems.DECIMAL_PLACES),
            n_starts=eval_config.get('n_starts', evaluate_on_problems.N_STARTS),
            max_iter=eval_config.get('max_iter', evaluate_on_problems.MAX_ITER),
            bounds=eval_config.get('bounds', evaluate_on_problems.PARAMS_BOUNDS),
            x0=warm_start,
            seed=eval_config.get('seed', None),
            verbose=eval_config.get('verbose', False),
        )
        if not isinstance(results, (int, float)):
            return None, None, False, 'no output', None
        res = _sample_residuals(
            full_res, eval_config.get('sample_size', evaluate_on_problems.SAMPLE_SIZE))
        return results, res, True, 'yes', opt_params
    except Exception as e:
        return None, None, False, f'Execution Error: {e}', None

class LocalSandbox(Sandbox):
    """在常驻子进程中执行并评估 LLM 生成的程序（支持超时与 numba 可选加速）。

    相比每条样本都 spawn 新进程，常驻 worker 避免重复导入 numpy/scipy（Windows
    下 spawn 启动代价极高）；某条样本超时卡死时销毁并重建 worker，不影响后续评估。
    """

    def __init__(self, verbose=False, numba_accelerate=True, eval_config=None, pool_size=1):
        """
        Args:
            verbose (bool): Enable detailed output.
            numba_accelerate (bool): Use Numba for acceleration of evaluation (limited compatibility).
            eval_config (dict | None): 覆盖 evaluate_on_problems 的默认评估配置
                （n_params/decimal_places/n_starts/max_iter/bounds/seed/sample_size）。
            pool_size (int): 常驻 worker 进程数（当前评估串行调度，默认 1）。
        """
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate
        self._eval_config = dict(eval_config or {})
        self._last_params = None

        # numba 是可选加速依赖：未安装时自动降级，避免所有样本评估失败
        if self._numba_accelerate:
            try:
                import numba  # noqa: F401
            except Exception as e:
                print(f"[WARN] numba 未安装，已关闭 numba 加速评估（{e}）")
                self._numba_accelerate = False

        self._pool_size = max(1, pool_size)
        self._task_queue = multiprocessing.Queue()
        self._task_lock = threading.Lock()
        self._workers = self._spawn_workers()

    def _spawn_workers(self):
        workers = []
        for i in range(self._pool_size):
            p = multiprocessing.Process(
                target=_eval_worker, args=(self._task_queue, i), daemon=True)
            p.start()
            workers.append(p)
        return workers

    def close(self) -> None:
        """关停常驻 worker：向队列投递与 worker 数等量的 None 哨兵（_eval_worker
        收到即正常退出），超时未退再 terminate。旧实现没有任何关停路径，
        pipeline 的初始 evaluator 集合在初次分析后整组闲置，其 worker 进程
        一直挂到解释器退出（daemon 兜底），Windows 下白占内存。"""
        for _ in self._workers:
            try:
                self._task_queue.put(None)
            except Exception:
                break
        for p in self._workers:
            try:
                p.join(timeout=2)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1)
            except Exception:
                pass
        self._workers = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _respawn_workers(self):
        """销毁并重建 worker 池：某条样本超时卡死后恢复调度能力。

        同时丢弃任务队列中陈旧的任务：这些任务的结果管道已被关闭，新 worker 若继续
        消费只会白白执行（重任务可再占用数十秒），造成后续样本评估连锁超时。

        实现要点（**换队列**而不是 get_nowait 清空）：``multiprocessing.Queue`` 的
        投递由后台 feeder 线程异步完成，刚 put 进去的任务常常还没进管道，``get_nowait``
        看不到它们——旧实现因此只是"尽力清空"，残留任务仍会被新 worker 消费。
        换一条全新队列后，陈旧任务在结构上不可能再被任何 worker 取到。
        """
        for p in self._workers:
            if p.is_alive():
                p.terminate()
                p.join()
        stale_queue = self._task_queue
        # 先于 _spawn_workers() 换队：worker 在 spawn 时绑定 self._task_queue
        self._task_queue = multiprocessing.Queue()
        try:
            stale_queue.cancel_join_thread()   # 不等 feeder 线程排空（其数据已无意义）
            stale_queue.close()
        except Exception:
            pass
        self._workers = self._spawn_workers()


    def run(self, program: str, function_to_run: str, function_to_evolve: str,
            inputs: Any, test_input: str, timeout_seconds: int
            ) -> tuple[tuple[Any, bool, str], Any]:
        """
        执行给定样本，返回 (结果三元组, 残差采样)。
        结果三元组为 (grade, runs_ok, remark)；超时/失败时 grade 为 None。

        Note: This sandbox is specific to the equation program skeleton discovery problem.
        """
        dataset = inputs[test_input]
        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
        task = (program, function_to_run, function_to_evolve, dataset,
                self._numba_accelerate, self._eval_config, self._last_params, child_conn)

        # 串行调度：一次只投递一个任务（每个 sampler 独享自己的 Evaluator/Sandbox）
        with self._task_lock:
            self._task_queue.put(task)
            if parent_conn.poll(timeout_seconds):
                try:
                    grade, res, runs_ok, remark, params = parent_conn.recv()
                except (EOFError, OSError):
                    # worker 进程意外崩溃（如被 LLM 生成的代码拖垮），重建后返回失败结果
                    self._respawn_workers()
                    grade, res, runs_ok, remark, params = None, None, False, 'worker crashed', None
            else:
                # 超时：worker 可能被卡死样本占用，销毁并重建后返回超时结果
                self._respawn_workers()
                grade, res, runs_ok, remark, params = None, None, False, 'timeout01', None
            parent_conn.close()

        # 保留最优参数，供下一轮评估热启动（params 为 None 时自动忽略）
        self._last_params = params
        results = (grade, runs_ok, remark)
        if self._verbose:
            self._print_evaluation_details(program, results, function_to_evolve)
        return results, res




    def _print_evaluation_details(self, program, results, func_to_evolve: str = 'equation'):
        """打印被评估的程序与分数（verbose 模式）。

        旧实现从 ``**kwargs`` 里取 ``func_to_evolve``，而调用方从不传 kwargs，
        因此永远回退成默认值 'equation'——verbose 输出对非 equation 任务名是错的。
        现在由 ``run()`` 直接传入真实函数名。
        """
        print('================= Evaluated Program =================')
        program = code_manipulation.sanitize_code_text(program)
        function = code_manipulation.text_to_program(program).get_function(func_to_evolve)
        print(f'{str(function).strip()}\n-----------------------------------------------------')
        print(f'Score: {results}\n=====================================================\n\n')

def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """ Return whether the generated function is calling an earlier version. """
    for name in code_manipulation.get_functions_called(program):
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False
