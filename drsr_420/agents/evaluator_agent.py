# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""评估 Agent：在沙箱中执行并评估 LLM 生成的方程程序。

角色：EvaluatorAgent 是"评估者"，把 SamplerAgent 产出的骨架编译成可运行程序，
交给常驻 worker（LocalSandbox）执行 + 多起点 least_squares 拟合参数，返回
(score, error, residual) 三元组，供 CoordinatorAgent 分类 Good/Bad/None。

协作：
- 上游：CoordinatorAgent（通过 analyse() 提交单个样本）；
- 下游：LocalSandbox（常驻 worker 沙箱）→ evaluate_on_problems（参数优化）。
"""
from __future__ import annotations

import ast
import copy
import multiprocessing
import profile
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Type

import numpy as np

from drsr_420 import buffer
from drsr_420 import code_manipulation
from drsr_420 import evaluate_on_problems
from drsr_420 import evaluator_accelerate

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
            **kwargs

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
        # numba 加速（可选）：编译失败或方程不受支持时自动降级为原始程序
        if numba_accelerate:
            X = dataset['inputs']
            n_params = eval_config.get('n_params', evaluate_on_problems.MAX_NPARAMS)
            sample_args = tuple(X.T) + (np.ones(n_params),)
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

    def _respawn_workers(self):
        """销毁并重建 worker 池：某条样本超时卡死后恢复调度能力。"""
        for p in self._workers:
            if p.is_alive():
                p.terminate()
                p.join()
        self._workers = self._spawn_workers()


    def run(self, program: str, function_to_run: str, function_to_evolve: str,
            inputs: Any, test_input: str, timeout_seconds: int, **kwargs
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
            self._print_evaluation_details(program, results, **kwargs)
        return results, res




    def _print_evaluation_details(self, program, results, **kwargs):
        print('================= Evaluated Program =================')
        program = code_manipulation.sanitize_code_text(program)
        function = code_manipulation.text_to_program(program).get_function(kwargs.get('func_to_evolve', 'equation'))
        print(f'{str(function).strip()}\n-----------------------------------------------------')
        print(f'Score: {results}\n=====================================================\n\n')




def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """ Return whether the generated function is calling an earlier version. """
    for name in code_manipulation.get_functions_called(program):
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False



class EvaluatorAgent:
    """评估 Agent：编译并执行 LLM 生成的方程样本，产出 (score, error, residual)。

    职责：
    - 把骨架样本编译成可运行程序（_sample_to_program）；
    - 在 LocalSandbox 常驻 worker 中逐测试输入执行，收集分数；
    - 成功时把程序与分数注册进 ExperienceBuffer（register_program），
      失败时经 Profiler 记录 score=None 样本。
    """

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            template: code_manipulation.Program,
            function_to_evolve: str,
            function_to_run: str,
            inputs: Sequence[Any],
            timeout_seconds: int = 30,
            sandbox_class: Type[Sandbox] = Sandbox
    ):
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox_class()

    def analyse(
            self,
            sample: str,
            island_id: int | None,
            version_generated: int | None,
            **kwargs
    ) -> tuple[float | None, str, Any]:
        """ Compile the hypothesis sample into a program and executes it on test inputs. """
        new_function, program = _sample_to_program(
            sample, version_generated, self._template, self._function_to_evolve)
        scores_per_test = {}

        time_reset = time.time()

        # print('len of self._inputs: ',len(self._inputs))    # len of self._inputs:  1
        # print(self._inputs) # x1 x2
        '''
        {'data': {'inputs': array([[-0.25197899, -0.17306601],
       [-0.25232508, -0.17300887],
       [-0.25267104, -0.17295167],
       ...,
       [-0.41992701,  0.11309208],
       [-0.41970063,  0.11328232],
       [-0.41947387,  0.11347256]]), 'outputs': array([0.0285521 , 0.02858525, 0.02861839, ..., 0.09512003, 0.09511695,
       0.09511373])}}
        '''

        # print('len of self._inputs: ',len(self._inputs))    # len of self._inputs:  1
        # print(bbbbb)
        for current_input in self._inputs:

            results, res = self._sandbox.run(
                program, self._function_to_run, self._function_to_evolve, self._inputs, current_input,
                self._timeout_seconds
            )
            test_output, runs_ok, error_msg = results
            if runs_ok and not _calls_ancestor(program, self._function_to_evolve) and test_output is not None:
                if not isinstance(test_output, (int, float)):
                    print(f'Error: test_output is {test_output}')
                    raise ValueError('@function.run did not return an int/float score.')
                scores_per_test[current_input] = test_output

        evaluate_time = time.time() - time_reset
        ###################
        # print("error_msg=========")
        # print(error_msg)
        # print(test_output)      # score: -0.0004185108785400066 为针对初始化方程框架的评分
        # print('我从analyse中拿到了res', res)
        # print(bbb)


        # 果代码运行成功并得到有效评分，分数会被保存到经验缓冲区(ExperienceBuffer)：
        '''
        这里的_database就是从sampler.py传入的buffer.ExperienceBuffer实例。它将：

        将函数与其评分一起保存
        将函数分配到适当的"岛屿"(island)中
        根据功能相似性将函数组织到集群(clusters)中
        '''
        if scores_per_test:
            # 将优化参数保存到函数对象，便于 Profiler 写入 samples JSON
            try:
                params = getattr(self._sandbox, '_last_params', None)
                new_function.optimized_params = params
            except Exception:
                pass

            self._database.register_program(
                new_function,
                island_id,
                scores_per_test,
                **kwargs,
                evaluate_time=evaluate_time
            )

        else:
            profiler: profile.Profiler = kwargs.get('profiler', None)
            if profiler:
                global_sample_nums = kwargs.get('global_sample_nums', None)
                sample_time = kwargs.get('sample_time', None)
                new_function.global_sample_nums = global_sample_nums
                new_function.score = None
                new_function.sample_time = sample_time
                new_function.evaluate_time = evaluate_time
                try:
                    params = getattr(self._sandbox, '_last_params', None)
                    new_function.optimized_params = params
                except Exception:
                    pass
                profiler.register_function(new_function)


        return test_output, error_msg, res


# 兼容别名：旧模块名 drsr_420.evaluator.Evaluator 指向本类
Evaluator = EvaluatorAgent
