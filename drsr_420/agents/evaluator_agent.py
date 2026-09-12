"""评估者 Agent：把骨架编译成程序并委托沙箱执行，产出评分与残差。

角色：``EvaluatorAgent`` 是"评估者"——它只负责**编排**：编译骨架、把任务交给
沙箱、按结果注册进经验缓冲或经 Profiler 记录失败。真正的执行机制（常驻进程池、
超时重建、拟合调用、程序编译）在 :mod:`drsr_420.evaluation.sandbox`。

协作：
- 上游：CoordinatorAgent（``analyze(EvaluationRequest)``）；pipeline 亦调用一次以评估初始模板
- 下游：``evaluation.sandbox.LocalSandbox`` → ``evaluation.problems``（多起点拟合）
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Type

from drsr_420.agents.base import (
    PIPELINE,
    THREAD_PER_SAMPLER,
    AgentSpec,
    BaseAgent,
)
from drsr_420.agents.messages import EvaluationOutcome, EvaluationRequest
from drsr_420.core import buffer
from drsr_420.core import code_manipulation

# 执行机制从 evaluation/sandbox.py 引入；同时构成对该模块全部名字的 re-export
# （历史导入路径 `from drsr_420.agents.evaluator_agent import LocalSandbox, ...`
# 与 `drsr_420.evaluator` 兼容层都依赖这一点，故此处一次性列全）。
from drsr_420.evaluation.sandbox import (  # noqa: F401
    LocalSandbox,
    Sandbox,
    _FunctionLineVisitor,
    _calls_ancestor,
    _eval_worker,
    _run_evaluation_task,
    _sample_residuals,
    _sample_to_program,
    _trim_function_body,
)

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
        downstream=(),      # 下游是 evaluation 层的沙箱与拟合，不是 Agent
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


Evaluator = EvaluatorAgent
