"""命令行入口：把原先脚本式的 ``main.py`` 拆成可测函数。

重构前的问题：

* 有效逻辑全部塞在 ``if __name__ == '__main__':`` 内（约 360 行），任何一段都无法单测；
* ``config = config.Config(...)`` 把导入的 **config 模块名覆盖成实例**——此后任何
  ``config.ClassConfig`` 都会 AttributeError，只是当前恰好用不到才没炸；
* 重复 import（numpy 两次、json 两次）与成片注释掉的调试代码。

现在：参数解析 / 输出双写 / 日志 / LLM 客户端 / 数据集 / spec 渲染 / 产物快照各成
一个函数，``main(argv)`` 只做编排。**命令行接口与产物文件名保持不变**
（``.idea/runConfigurations/*.xml`` 的 4 个 MRF 运行配置、``example.sh`` /
``example.bat`` 与 4 组 ``MRF*.sh`` / ``MRF*.bat`` 都以 ``python -m drsr_420.cli.main``
启动本模块）。
"""
from __future__ import annotations

import json
import logging as pylogging
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from drsr_420.agents.evaluator_agent import LocalSandbox
from drsr_420.agents.sampler_agent import SamplerAgent
from drsr_420.cli.llm_setup import (
    build_llm_client,
    build_role_clients,
    load_llm_config_file,
)
from drsr_420.core import config as config_lib
from drsr_420.core import prompt_config as pc
from drsr_420.runtime import pipeline
from drsr_420.evaluation.problems import MAX_NPARAMS

DEFAULT_BACKGROUND = (
    "The physical properties of this equation are unknown and need to be analyzed "
    "based on experience."
)

#: 动态 spec 模板（NumPy 版）。占位符由 :func:`render_spec` 填充。
SPEC_TEMPLATE_NUMPY = '''\
"""
Find the mathematical function skeleton that fits the data.

Background:
{BACKGROUND}

Variables:
- Independents: {FEATURE_DOC}
- Dependent: {DEPENDENT}
"""

import numpy as np
from scipy.optimize import minimize

# Initialize parameters
MAX_NPARAMS = {MAX_NPARAMS}
params = [1.0]*MAX_NPARAMS

@evaluate.run
def evaluate(data: dict) -> float:
    """ Evaluate the equation on data observations. """
    inputs, outputs = data['inputs'], data['outputs']
    X = inputs

    def loss(params):
        y_pred = equation(*X.T, params)
        return np.mean((y_pred - outputs) ** 2)

    result = minimize(loss, [1.0]*MAX_NPARAMS, method='BFGS')
    loss_val = result.fun
    if np.isnan(loss_val) or np.isinf(loss_val):
        return None
    else:
        return -loss_val

@equation.evolve
def equation({FEATURE_SIG}, params: np.ndarray) -> np.ndarray:
    """Equation to be evolved.

    Background:
    {BACKGROUND}

    Variables:
    - Independents: {FEATURE_DOC}
    - Dependent: {DEPENDENT}

    Parameters:
    - params (np.ndarray): Trainable coefficients used by the equation skeleton.
    """
    return {LINEAR_SEED}
'''

#: 配置文件命名约定（**唯一权威是文件内的 model 字段**，文件名只是给人看的标签）：
#: ``config/<提供商>_<模型>.config``（不入库，含密钥）与同名的 ``.example`` 模板（入库）。
#: 「哪个角色用哪套」由 ``config/agents.config.json`` 声明，见 drsr_420.llm.roles。


class _Tee:
    """把写往标准输出/错误的内容同时旁路到文件。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def fileno(self):
        # 替换 sys.stdout/stderr 后，任何 Popen(stderr=sys.stderr) 或探测终端能力的
        # 第三方库都会撞上无 fileno 的 AttributeError；透传主流的 fileno
        # （归档到 run.out/run.err 的旁路写入不受影响）。
        return self.streams[0].fileno()

    def isatty(self):
        try:
            return bool(self.streams[0].isatty())
        except Exception:
            return False


def build_parser() -> ArgumentParser:
    """构造命令行解析器（接口与重构前完全一致）。"""
    parser = ArgumentParser()
    # 共有参数
    parser.add_argument('--problem_name', type=str, default="problem")
    parser.add_argument('--data_csv', type=str, required=True,
                        help='含表头的 CSV，前 n-1 列为特征，最后一列为因变量')
    parser.add_argument('--experiment_dir', type=str, default=None,
                        help='实验目录（可为绝对/相对路径）。如提供，将直接使用此目录')
    parser.add_argument('--niterations', type=int, default=10,
                        help='搜索迭代轮数；若提供，将按 niterations * num_samplers * samples_per_iteration 计算最大采样数')
    parser.add_argument('--timeout_in_seconds', type=int, default=None,
                        help='总超时时间（秒）。到达即停止训练（即便未达迭代数）')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子（影响 Python 与 NumPy 的随机性）')
    parser.add_argument('--num_islands', type=int, default=None,
                        help='经验缓冲的岛屿数量（覆盖默认 10）')
    parser.add_argument('--num_samplers', type=int, default=None,
                        help='并行采样器数量（多线程并行采样，默认 1）')
    # 算法私有参数
    parser.add_argument('--llm_config', type=str, default=None,
                        help='默认 LLM 档案（config/<提供商>_<模型>.config 或档案 ID）。'
                             '未在 config/agents.config.json 里绑定档案的角色共用它；'
                             '省略时用注册表的 default')
    parser.add_argument('--role-config', action='append', default=None,
                        metavar='ROLE=FILE',
                        help='按角色覆盖档案（可重复），如 --role-config '
                             'explain=<档案ID>；ROLE 用 * 表示所有角色。'
                             '角色清单与当前绑定见 python -m drsr_420.llm.roles')
    parser.add_argument('--background', type=str, default=None, help='背景知识（可选）')
    parser.add_argument('--background_file', type=str, default=None,
                        help='背景知识来源文件（UTF-8 文本，如 backgrounds/MRFCompress-Cuboid.txt）。'
                             '与 --background 互斥；脚本推荐用本参数只传路径——'
                             'txt 是规范源，改动即刻生效，不存在脚本内联副本失步的问题')
    parser.add_argument('--samples_per_iteration', type=int, default=None,
                        help='每轮生成的候选数量（覆盖 config 默认值）')
    # 收敛型早停（全部默认关闭；预算型条件 --niterations/--timeout_in_seconds 照常兜底）
    parser.add_argument('--target_nmse', type=float, default=None,
                        help='早停目标：全局最优 NMSE 达到该值即停（None=关闭）。'
                             '对 MRF 类问题建议 1e-6 量级——实测一 run 第 7 批即达 1e-6，'
                             '之后 20 批纯属浪费')
    parser.add_argument('--early_stop_patience', type=int, default=None,
                        help='平台期早停：连续 N 个全局批次无全局最优改进即停（None=关闭）。'
                             '必须给足（建议 >= 2×num_islands）：实测有 run 连续 18 批无改进'
                             '后才出现全场最优')
    parser.add_argument('--min_batches', type=int, default=None,
                        help='平台期 warmup：全局完成批次数达到该值前不判平台期'
                             '（默认自动取 num_islands，保证每座岛至少轮到一次）')
    parser.add_argument('--max_failed_batches', type=int, default=None,
                        help='失败熔断：连续 N 个批次所有样本评估失败即停（None=关闭）')
    return parser


def apply_seed(seed: int | None) -> None:
    """设定 Python / NumPy 随机种子（失败只告警，不中断实验）。"""
    try:
        if seed is not None:
            import random as _random

            os.environ["PYTHONHASHSEED"] = str(seed)
            _random.seed(seed)
            np.random.seed(seed)
            print(f"[INFO] Random seed set: {seed}")
    except Exception as e:
        print(f"[WARN] Failed to set seed: {e}")


def resolve_results_root(problem_name: str, experiment_dir: str | None) -> str:
    """统一结果目录：``--experiment_dir`` 优先，否则 ``experiments/{problem}_{ts}``。"""
    if experiment_dir:
        results_root = experiment_dir
    else:
        ts = time.strftime('%Y%m%d-%H%M%S')
        results_root = os.path.join('experiments', f"{problem_name}_{ts}")
    os.makedirs(results_root, exist_ok=True)
    # 所有产物直接放在实验根目录（不创建 logs 子目录）
    return results_root


def setup_output_tee(results_root: str):
    """把 stdout/stderr 同时写入 ``run.out`` / ``run.err``，返回两个文件句柄。"""
    out_fp = open(os.path.join(results_root, "run.out"), "a", encoding="utf-8")
    err_fp = open(os.path.join(results_root, "run.err"), "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, out_fp)
    sys.stderr = _Tee(sys.stderr, err_fp)
    return out_fp, err_fp


def configure_logging() -> None:
    """统一 Python 日志格式（含时间戳），影响 absl 等库的输出。"""
    kwargs = dict(
        level=pylogging.INFO,
        format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    try:
        pylogging.basicConfig(force=True, **kwargs)
    except TypeError:
        # 兼容旧版 Python（无 force 参数）
        pylogging.basicConfig(**kwargs)


def load_csv(path: str):
    """读取训练 CSV：前 n-1 列为特征、最后一列为因变量。"""
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError('CSV 至少需要两列（>=1 特征 + 1 因变量）')
    cols = list(df.columns)
    feature_names = cols[:-1]
    y_name = cols[-1]
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy().reshape(-1)
    return X, y, feature_names, y_name


def ensure_feature_names(n: int, names) -> list:
    """特征名缺省时生成 x1..xN；数量不符则报错（避免静默错配变量）。"""
    if names is None:
        return [f"x{i+1}" for i in range(n)]
    if len(names) != n:
        raise ValueError(f"feature_names 长度应为 {n}，实际为 {len(names)}")
    return names


def render_spec(n_features, feature_names=None, dependent_name=None,
                background=None, max_nparams=10) -> str:
    """渲染 NumPy 版 specification（含可运行的初始线性骨架）。

    Note:
        重构前的同名局部函数还接收一个 ``problem`` 参数但从未使用，已移除。
    """
    feats = ensure_feature_names(n_features, feature_names)
    dep = (dependent_name or 'y').strip()
    bg = (background or DEFAULT_BACKGROUND).strip()
    feature_sig = ', '.join([f"{name}: np.ndarray" for name in feats])
    feature_doc = ', '.join(feats)
    idx_c = min(3, n_features)
    terms = [f"params[{i}]*{feats[i]}" for i in range(min(3, n_features))]
    terms.append(f"params[{idx_c}]")
    linear_seed = ' + '.join(terms)
    return SPEC_TEMPLATE_NUMPY.format(
        BACKGROUND=bg,
        FEATURE_SIG=feature_sig,
        FEATURE_DOC=feature_doc,
        DEPENDENT=dep,
        MAX_NPARAMS=max_nparams,
        LINEAR_SEED=linear_seed,
    )


def save_dynamic_spec(results_root: str, specification: str) -> None:
    """把本次动态渲染的 spec 落盘，便于复现与调试。"""
    try:
        spec_out_path = os.path.join(results_root, "spec_dynamic.txt")
        with open(spec_out_path, "w", encoding="utf-8") as f:
            f.write(specification)
        print(f"[INFO] Saved dynamic spec to: {spec_out_path}")
    except Exception as e:
        print(f"[WARN] Failed to save dynamic spec: {e}")


def ensure_experiences_file(results_root: str) -> str:
    """确保 ``experiences.json`` 存在（不存在则写入初始空结构）。"""
    json_experience_file = os.path.join(results_root, "experiences.json")
    if not os.path.exists(json_experience_file):
        initial_experiences = {"None": [], "Good": [], "Bad": []}
        try:
            with open(json_experience_file, "w", encoding="utf-8") as f:
                json.dump(initial_experiences, f, ensure_ascii=False, indent=2)
            print(f"成功创建初始经验 JSON 文件: {json_experience_file}")
        except Exception as e:
            print(f"创建 JSON 文件时出错: {str(e)}")
    else:
        print(f"经验 JSON 文件已存在: {json_experience_file}")
    return json_experience_file


def save_config_snapshot(results_root: str, payload: dict) -> None:
    """把本次实验超参与 LLM 配置快照到实验目录，便于复现（api_key 已打码）。"""
    try:
        path = os.path.join(results_root, "config_snapshot.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved config snapshot: {path}")
    except Exception as e:
        print(f"[WARN] Failed to save config snapshot: {e}")


def resolve_background(parser: ArgumentParser, args) -> str | None:
    """解析背景词：--background（内联文本）与 --background_file（规范源文件）二选一。

    ``--background_file`` 读 UTF-8 文本并去首尾空白——backgrounds/*.txt 是规范源，
    脚本只传路径，txt 改动即刻生效；文件不存在或为空直接终止，绝不让实验
    在"没有背景词"的状态下静默开跑。
    """
    if args.background and args.background_file:
        parser.error('--background 与 --background_file 只能二选一')
    if not args.background_file:
        return args.background
    if not os.path.isfile(args.background_file):
        raise SystemExit(f'[ERROR] 背景词文件不存在: {args.background_file}')
    with open(args.background_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if not text:
        raise SystemExit(f'[ERROR] 背景词文件为空: {args.background_file}')
    print(f"[INFO] 背景词已从 {args.background_file} 读取（{len(text)} 字符）")
    return text


def main(argv: list[str] | None = None) -> int:
    """完整运行一次方程发现实验。返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    background = resolve_background(parser, args)

    class_config = config_lib.ClassConfig(
        llm_class=SamplerAgent, sandbox_class=LocalSandbox)

    # 实验总时长（秒）
    wall_limit_seconds = (int(args.timeout_in_seconds)
                          if args.timeout_in_seconds and args.timeout_in_seconds > 0 else None)

    apply_seed(args.seed)

    results_root = resolve_results_root(args.problem_name, args.experiment_dir)
    setup_output_tee(results_root)
    print(f"[INFO] Results root: {results_root}")
    configure_logging()

    # 允许从命令行覆盖 samples_per_iteration（映射到 Config.samples_per_prompt）与岛屿数
    eb_cfg = config_lib.ExperienceBufferConfig(
        num_islands=(int(args.num_islands) if args.num_islands and args.num_islands > 0
                     else config_lib.ExperienceBufferConfig().num_islands))
    num_samplers = int(args.num_samplers) if args.num_samplers and args.num_samplers > 0 else 1

    # 收敛型早停参数（None = 关闭）
    early_stop_cfg = dict(
        target_nmse=(float(args.target_nmse)
                     if args.target_nmse and args.target_nmse > 0 else None),
        early_stop_patience=(int(args.early_stop_patience)
                             if args.early_stop_patience and args.early_stop_patience > 0
                             else None),
        min_batches_before_early_stop=(int(args.min_batches)
                                       if args.min_batches and args.min_batches > 0
                                       else None),
        max_failed_batches=(int(args.max_failed_batches)
                            if args.max_failed_batches and args.max_failed_batches > 0
                            else None),
    )

    # 注意：局部变量名用 exp_config，绝不再覆盖 config 模块名（重构前的真实陷阱）
    if args.samples_per_iteration is not None and args.samples_per_iteration > 0:
        exp_config = config_lib.Config(
            results_root=results_root,
            samples_per_prompt=int(args.samples_per_iteration),
            wall_time_limit_seconds=wall_limit_seconds,
            experience_buffer=eb_cfg,
            num_samplers=num_samplers,
            **early_stop_cfg,
        )
    else:
        exp_config = config_lib.Config(
            results_root=results_root,
            wall_time_limit_seconds=wall_limit_seconds,
            experience_buffer=eb_cfg,
            num_samplers=num_samplers,
            **early_stop_cfg,
        )

    llm_config = load_llm_config_file(args.llm_config)
    client = build_llm_client(llm_config)
    # 角色化的客户端池：sampling/analysis/experience/residual/explain/summary 各自
    # 按 config/agents.config.json 解析（可用 --role-config 覆盖）
    role_clients = build_role_clients(args.llm_config, args.role_config)

    # 最大采样数量：优先由 --niterations 推导；否则使用默认 1000
    if args.niterations is not None and args.niterations > 0:
        global_max_sample_num = (int(args.niterations)
                                 * int(getattr(exp_config, 'num_samplers', 1))
                                 * int(getattr(exp_config, 'samples_per_prompt', 4)))
    else:
        global_max_sample_num = 1000

    X, y, feature_names, y_name = load_csv(args.data_csv)

    specification = render_spec(
        n_features=X.shape[1],
        feature_names=feature_names,
        dependent_name=y_name,
        background=background,
        max_nparams=MAX_NPARAMS,
    )
    save_dynamic_spec(results_root, specification)

    dataset = {'data': {'inputs': X, 'outputs': y}}
    ensure_experiences_file(results_root)

    # PromptContext：保证提示一致性（变量名/背景）
    prompt_ctx = pc.PromptContext(
        n_features=X.shape[1],
        feature_names=feature_names,
        dependent_name=y_name,
        problem_name=args.problem_name,
        background=background,
        max_params=MAX_NPARAMS,
    )

    save_config_snapshot(results_root, {
        "problem_name": args.problem_name,
        "data_csv": args.data_csv,
        "seed": args.seed,
        "niterations": args.niterations,
        "num_islands": exp_config.experience_buffer.num_islands,
        "num_samplers": exp_config.num_samplers,
        "samples_per_prompt": exp_config.samples_per_prompt,
        "max_sample_nums": global_max_sample_num,
        "wall_time_limit_seconds": wall_limit_seconds,
        "early_stop": {
            "target_nmse": exp_config.target_nmse,
            "early_stop_patience": exp_config.early_stop_patience,
            "min_batches_before_early_stop": exp_config.min_batches_before_early_stop,
            "max_failed_batches": exp_config.max_failed_batches,
        },
        "background": background,
        "background_file": args.background_file,
        "llm": {
            "provider": client._provider_name() if client else None,
            "model": client.model if client else llm_config.get('model', ''),
            "base_url": (client.base_url if client else '') or llm_config.get('base_url', ''),
            "api_key": ("***" if (client and client.api_key) else ""),
            "kwargs": getattr(client, 'kwargs', None) if client else None,
            # 每个角色最终生效的档案与来源（含 --role-config / 环境变量 / 注册表默认），
            # 让"这次实验的 explain 到底用了哪个模型"可追溯——旧结构下这无从查起，
            # 因为 explain 会自己硬编码另一个档案文件。
            "roles": role_clients.describe(),
        },
        "results_root": results_root,
    })

    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=exp_config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        results_root=results_root,
        prompt_ctx=prompt_ctx,
        llm_client=client,
        llm_config=llm_config,
        role_clients=role_clients,
        seed=args.seed,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
