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

"""Configuration of a LLMSR experiments
."""
from __future__ import annotations


import dataclasses
from typing import TYPE_CHECKING, Type
import os

if TYPE_CHECKING:
    # 仅用于类型注解（ClassConfig）。运行时延迟解析，避免
    # config -> sampler(shim) -> agents.sampler_agent -> config 循环导入。
    from drsr_420 import sampler
    from drsr_420 import evaluator


@dataclasses.dataclass(frozen=True)
class ExperienceBufferConfig:
    """Configures Experience Buffer parameters.
    
    Args:
        functions_per_prompt (int): Number of previous hypotheses to include in prompts
        num_islands (int): Number of islands in experience buffer for diversity
        reset_period (int): Seconds between weakest island resets
        cluster_sampling_temperature_init (float): Initial cluster softmax sampling temperature
        cluster_sampling_temperature_period (int): Period for temperature decay
    """
    functions_per_prompt: int = 2
    num_islands: int = 10 
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000


@dataclasses.dataclass(frozen=True)
class ExperienceInjectionConfig:
    """Configures experience injection into sampling prompts.

    Args:
        optional_category_probability (float): Probability that a Good/Bad category participates.
            None (failure lessons) is always injected.
        max_per_category (dict): Max entries injected per category ('None'/'Good'/'Bad').
        freshness_threshold (int): Sample order above which the freshness window applies.
        freshness_window_ratio (float): When sample_order > freshness_threshold, only keep
            experiences with sample_order in [current*ratio, current].
        inject_residual_probability (float): Probability of injecting the latest residual analysis.
        max_analysis_chars (int): Max characters of each injected analysis text.
    """
    optional_category_probability: float = 0.5
    max_per_category: dict = dataclasses.field(
        default_factory=lambda: {"None": 3, "Good": 2, "Bad": 2}
    )
    freshness_threshold: int = 50
    freshness_window_ratio: float = 0.7
    inject_residual_probability: float = 0.5
    max_analysis_chars: int = 500


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for LLMSR experiments.
   
   Args:
       experience_buffer: Evolution multi-population settings
       num_samplers (int): Number of parallel samplers
       num_evaluators (int): Number of parallel evaluators
       samples_per_prompt (int): Number of hypotheses per prompt
       evaluate_timeout_seconds (int): Hypothesis evaluation timeout
   """
    experience_buffer: ExperienceBufferConfig = dataclasses.field(default_factory=ExperienceBufferConfig)
    experience_injection: ExperienceInjectionConfig = dataclasses.field(default_factory=ExperienceInjectionConfig)
    num_samplers: int = 1 
    num_evaluators: int = 1
    samples_per_prompt: int = 4
    ####################################################
    # samples_per_prompt: int = 1
    evaluate_timeout_seconds: int = 30  
    # 新增：统一结果目录（results/{problem}_{ts}）
    results_root: str | None = None
    # 实验总时长上限（秒）；None 表示不限制
    wall_time_limit_seconds: int | None = None


@dataclasses.dataclass()
class ClassConfig:
    llm_class: Type[sampler.LLM]
    sandbox_class: Type[evaluator.Sandbox]
