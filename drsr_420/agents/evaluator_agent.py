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
- 上游：CoordinatorAgent（通过 analyze() 提交单个样本）；
- 下游：LocalSandbox（常驻 worker 沙箱）→ evaluate_on_problems（参数优化）。
"""
from __future__ import annotations

import ast
import copy
import multiprocessing
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
from drsr_420.agents.base import (
    PIPELINE,
    THREAD_PER_SAMPLER,
    AgentSpec,
    BaseAgent,
)
from drsr_420.agents.messages import EvaluationOutcome, EvaluationRequest

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



class EvaluatorAgent(BaseAgent):
    """评估 Agent：编译并执行 LLM 生成的方程样本，产出 (score, error, residual)。

    职责：
    - 把骨架样本编译成可运行程序（_sample_to_program）；
    - 在 LocalSandbox 常驻 worker 中逐测试输入执行，收集分数；
    - 成功时把程序与分数注册进 ExperienceBuffer（register_program），
      失败时经 Profiler 记录 score=None 样本。
    """

    SPEC = AgentSpec(
        key="evaluator",
        role="评估者",
        mission="编译骨架、在常驻沙箱中执行并多起点拟合打分，产出 (score, error, residual)",
        entrypoints=("analyze", "analyse"),   # analyse 为兼容别名（deprecated）
        upstream=("coordinator", PIPELINE),
        downstream=(),      # 下游是 runtime 层的沙箱与拟合，不是 Agent
        consumes=("sample: str", "island_id: int | None", "version_generated: int | None"),
        produces=("score: float | None", "error_msg: str | None",
                  "residual: np.ndarray | None"),
        artifacts=("samples/samples_N.json",),   # 经 Profiler 落盘
        thread_model=THREAD_PER_SAMPLER,
        llm_task=None,
        notes="每个 sampler 线程独享一份实例（避免 LocalSandbox._last_params 等实例状态竞态）。",
    )

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            template: code_manipulation.Program,
            function_to_evolve: str,
            function_to_run: str,
            inputs: Sequence[Any],
            timeout_seconds: int = 30,
            sandbox_class: Type[Sandbox] | None = None
    ):
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        # 默认 LocalSandbox：此前默认值是抽象基类 Sandbox，__init__ 里立刻
        # sandbox_class() 实例化会直接 TypeError，等于"文档声明的默认值不可用"
        self._sandbox = (sandbox_class or LocalSandbox)()

    def analyze(self, request: EvaluationRequest) -> EvaluationOutcome:
        """编译请求中的骨架样本并在沙箱中执行，返回评估结果。

        Args:
            request: 评估请求（样本、岛屿/版本号，以及记录用的归属信息）。
        """
        new_function, program = _sample_to_program(
            request.sample, request.version_generated, self._template,
            self._function_to_evolve)
        scores_per_test = {}

        # 循环外初始化：self._inputs 为空时循环不执行，避免末尾 return 触发 UnboundLocalError
        test_output, error_msg, res = None, None, None

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
                request.island_id,
                scores_per_test,
                profiler=request.profiler,
                global_sample_nums=request.global_sample_nums,
                sample_time=request.sample_time,
                evaluate_time=evaluate_time,
            )

        else:
            profiler = request.profiler
            if profiler:
                new_function.global_sample_nums = request.global_sample_nums
                new_function.score = None
                new_function.sample_time = request.sample_time
                new_function.evaluate_time = evaluate_time
                try:
                    params = getattr(self._sandbox, '_last_params', None)
                    new_function.optimized_params = params
                except Exception:
                    pass
                profiler.register_function(new_function)

        # 缓冲区没有登记这个样本（如调用了祖先版本被拒）时，必须以"无分数"
        # 返回：否则 coordinator 会把被丢弃的程序当作已评分样本更新 best、
        # 写入 experiences.json（带数值分、无 error），经验回路反过来
        # 推荐一个已被判废的程序。
        if not scores_per_test:
            test_output = None
            res = None

        return EvaluationOutcome(score=test_output, error=error_msg, residual=res)

    def analyse(self, sample: str, island_id: int | None,
                version_generated: int | None,
                profiler: Any = None,
                global_sample_nums: int | None = None,
                sample_time: float | None = None
                ) -> tuple[float | None, str | None, Any | None]:
        """兼容入口（deprecated）：旧签名 ``analyse(...)`` → ``analyze(EvaluationRequest)``。

        仅用于过渡期外部调用方，内部调用点已全部改用 :class:`EvaluationRequest`。
        相比旧实现，这里不再用 ``**kwargs`` 收参数——参数名拼错会立刻 TypeError，
        而不是静默丢掉 profiler 记录。
        """
        request = EvaluationRequest(
            sample=sample,
            island_id=island_id,
            version_generated=version_generated,
            global_sample_nums=global_sample_nums,
            sample_time=sample_time,
            profiler=profiler,
        )
        return self.analyze(request).to_legacy()

    def close(self) -> None:
        """释放沙箱资源（LocalSandbox 常驻 worker）；沙箱不支持则静默跳过。"""
        try:
            close = getattr(self._sandbox, 'close', None)
            if callable(close):
                close()
        except Exception:
            pass


# 兼容别名：旧模块名 drsr_420.evaluator.Evaluator 指向本类
Evaluator = EvaluatorAgent
