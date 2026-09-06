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

"""评估器兼容层（re-export 至 agents 子包）。

原实现已迁移到 drsr_420/agents/evaluator_agent.py：
- Evaluator / Sandbox / LocalSandbox / _eval_worker / _run_evaluation_task 等均在新模块。

本模块仅保留旧模块名，供 config.py / pipeline.py / tests/test_evaluator.py 无缝引用。
"""
from __future__ import annotations

from drsr_420.agents.evaluator_agent import (  # noqa: E402
    EvaluatorAgent as Evaluator,
    Sandbox,
    LocalSandbox,
    _FunctionLineVisitor,
    _trim_function_body,
    _sample_to_program,
    _eval_worker,
    _run_evaluation_task,
    _sample_residuals,
    _calls_ancestor,
)
