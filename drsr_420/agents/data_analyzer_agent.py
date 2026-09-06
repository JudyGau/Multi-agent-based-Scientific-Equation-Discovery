"""数据分析 Agent：使用 LLM 分析 CSV / 数据字典，产出初次数据分析结果。

角色：DataAnalyzerAgent 是"数据分析者"，在实验开始时对目标数据集做初次分析，
把分析结果写入 residual_analyze.json（sample_order=0 的初始记录），
后续由 CoordinatorAgent 的残差分析在其基础上追加。

协作：
- 上游：pipeline.main（传入 inputs 数据 + PromptContext 动态渲染的初次分析提示）；
- 下游：LLMClient（analysis 任务），产物写入 residual_analyze.json。
"""
import time
import threading
from drsr_420.console import LineStreamPrinter, print_block
import numpy as np
import pandas as pd
import io
from typing import Dict, Any, Optional, Union
import requests
import json
import os
import http.client

from drsr_420 import prompt_config as pc

Port = '5000'


class DataAnalyzerAgent:
    """数据分析 Agent：使用本地大模型分析 CSV / 数据字典。"""

    # 全局变量：保留小数位数和随机采样数量
    DECIMAL_PLACES = 3  # 默认保留3位小数
    SAMPLE_SIZE = 100   # 默认随机采样100条数据

    def __init__(self, api_url: str = f"http://127.0.0.1:{Port}/completions", timeout: int = 300,
                 decimal_places: int = None, sample_size: int = None, base_dir: str | None = None,
                 llm_client: object | None = None, seed: int | None = None):
        """
        初始化数据分析器

        Args:
            api_url: 语言模型API的URL地址
            timeout: API请求超时时间(秒)
            decimal_places: 保留小数位数，None表示使用默认值
            sample_size: 随机采样数量，None表示使用默认值
        """
        self.api_url = api_url
        self.timeout = timeout
        self.base_dir = base_dir or "."
        self.llm_client = llm_client
        self.seed = seed

        # 如果传入了自定义值，则覆盖默认值
        if decimal_places is not None:
            self.__class__.DECIMAL_PLACES = decimal_places
        if sample_size is not None:
            self.__class__.SAMPLE_SIZE = sample_size

    def _read_csv_data(self, csv_file_path: str, max_rows: Optional[int] = None) -> str:
        """
        读取CSV文件并转换为字符串，应用小数位数保留和随机采样

        Args:
            csv_file_path: CSV文件路径
            max_rows: 最大读取行数，None表示读取全部

        Returns:
            str: 处理后的CSV文件内容的字符串形式
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file_path, nrows=max_rows)

            # 对数值列应用小数位数保留
            numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
            for col in numeric_cols:
                df[col] = df[col].round(self.__class__.DECIMAL_PLACES)

            # 如果数据行数超过采样数量，则进行随机采样
            if len(df) > self.__class__.SAMPLE_SIZE and self.__class__.SAMPLE_SIZE > 0:
                df = df.sample(n=self.__class__.SAMPLE_SIZE, random_state=self.seed)  # 使用固定随机种子以保持结果可复现
                print(f"已从{csv_file_path}随机采样{self.__class__.SAMPLE_SIZE}行数据")

            # 转换为字符串
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            return buffer.getvalue()

        except Exception as e:
            print(f"读取CSV文件出错: {e}")
            return ""

    def _read_dataset_and_to_array(self, dataset: dict, max_rows: Optional[int] = None) -> np.ndarray:
        """
        将数据字典转换为NumPy数组，应用小数位数保留和随机采样
        Args:
            dataset: 数据字典，格式为
                data_dict = {'inputs': X, 'outputs': y}
                dataset = {'data': data_dict}
            max_rows: 最大读取行数，None表示读取全部
        Returns:
            np.ndarray: 处理后的NumPy数组
        """
        try:
            # 提取输入和输出数据
            inputs = dataset['data']['inputs']
            outputs = dataset['data']['outputs']

            # 将输入和输出数据转换为NumPy数组
            inputs_array = np.array(inputs)
            outputs_array = np.array(outputs)

            # 如果指定了最大行数，则进行切片
            if max_rows is not None:
                inputs_array = inputs_array[:max_rows]
                outputs_array = outputs_array[:max_rows]

            # 合并输入和输出数据
            combined_data = np.column_stack((inputs_array, outputs_array))

            # 随机采样
            rows_count = combined_data.shape[0]
            if rows_count > self.__class__.SAMPLE_SIZE and self.__class__.SAMPLE_SIZE > 0:
                # 使用独立 RNG 保持可复现，避免污染全局随机态
                rng = np.random.default_rng(self.seed)
                # 随机选择行索引
                indices = rng.choice(rows_count, self.__class__.SAMPLE_SIZE, replace=False)
                combined_data = combined_data[indices]
                print(f"已从数据集随机采样{self.__class__.SAMPLE_SIZE}行数据")

            # 保留指定位数的小数
            combined_data = np.round(combined_data, self.__class__.DECIMAL_PLACES)

            return combined_data

        except Exception as e:
            print(f"转换数据字典出错: {e}")
            return np.array([])

    def _create_prompt(self, csv_data: str, custom_prompt: Optional[str] = None) -> str:
        """
        创建分析提示

        Args:
            csv_data: CSV数据内容
            custom_prompt: 自定义提示，如果为None则使用默认提示

        Returns:
            str: 完整的提示文本
        """
        if custom_prompt:
            return custom_prompt.replace("{csv_data}", csv_data)

        # 默认提示（通用兜底；主流程已通过 custom_prompt 注入 PromptContext 动态模板）
        return f"""
        csv
        {csv_data}
You are a data analysis expert. I have provided a dataset for scientific analysis.
The first columns are the independent variables, and the last column is the dependent variable.
Each row represents a set of independent variables and the corresponding dependent variable value.

Task Requirements:

1. Please help me analyze and summarize the influence of the changes in the values of different independent variables on the dependent variable,
as well as the possible intrinsic relationships among different independent variables.

Your response only needs to answer your analysis results in the form below, and you don't need to show your analysis process.

"""+"""
2.##Output Format##:
STRICTLY deliver results in the following structured format:

  "output_format": {
    "analysis": {
      "independent_to_dependent_relationships": {
        "independent variable1 ": [
          "Hint: analyze the functional relationship between independent variable1 and the dependent variable in different intervals"
        ],
        "independent variable2 ": [
          "Hint: analyze the functional relationship between independent variable2 and the dependent variable in different intervals"
        ]
      },
      "inter_relationships_between_independents": {
        "independent variable1 vs independent variable2": [
          "Hint: analyze the possible functional relationship between the two independent variables in different intervals. If not, you can leave it blank."
        ]
      }
    }
  }

        """


    def _query_model(self, prompt: str) -> str:
        """
        向远程API发送请求

        Args:
            prompt: 提示文本

        Returns:
            str: 模型响应
        """
        try:
            if self.llm_client is None:
                raise RuntimeError('DataAnalyzer requires llm_client, but got None')
            # 按任务克隆客户端：思考强度等私有参数从配置 tasks.analysis 声明，
            # 由 ClientFactory 注入 task_params，provider 差异在 llm.py 适配层处理
            llm_client = self.llm_client.clone_for_task('analysis')
            # 数据分析仅需简短结论，限制输出长度，避免 max_tokens 过大导致服务端长时间生成
            llm_client.kwargs['max_tokens'] = 32768
            # 流式输出：通过 on_delta 回调实时打印思考内容与正文（[思考]/[正文] 视觉分隔）
            stream = LineStreamPrinter()
            shown = 0
            think_label_printed = False
            content_label_printed = False

            def _on_delta(chunk):
                nonlocal shown, think_label_printed, content_label_printed
                reasoning = chunk.get('reasoning_content') or ''
                content = chunk.get('content') or ''
                text = reasoning + content
                if len(text) > shown:
                    if shown < len(reasoning) and not think_label_printed:
                        stream.write("[思考]\n")
                        think_label_printed = True
                    elif not content_label_printed:
                        stream.write_line("[正文]")
                        content_label_printed = True
                    stream.write(text[shown:])
                    shown = len(text)

            resp = llm_client.chat([
                {"role": "system", "content": pc.system_prompt},
                {"role": "user", "content": prompt},
            ], on_delta=_on_delta)
            stream.flush()
            print()  # 流式结束后换行
            # 兜底：推理模型可能把完整分析输出在 reasoning_content 而 content 为空，
            # 此时回退到思考内容，避免初次/残差分析结果丢失导致经验注入链路断裂。
            return resp.get('content', '') or resp.get('reasoning_content', '')
        except Exception as e:
            error_msg = f"请求出错: {str(e)}"
            print(error_msg)
            return error_msg

    def analyze(self,
           data_source: Union[str, Dict],
           custom_prompt: Optional[str] = None,
           max_rows: Optional[int] = None,
           verbose: bool = True) -> str:
        """
        分析数据（支持CSV文件路径或数据字典）

        Args:
            data_source: 数据源，可以是CSV文件路径(str)或数据字典(dict)
                数据字典格式应为: {'data': {'inputs': X, 'outputs': y}}
            custom_prompt: 自定义提示模板，可以包含{csv_data}占位符
            max_rows: 最大读取行数，None表示读取全部
            verbose: 是否打印详细信息

        Returns:
            str: 大模型分析结果
        """
        if verbose:
            if isinstance(data_source, str):
                print(f"正在分析CSV文件数据: {data_source}")
            else:
                print("正在分析数据字典")

        # 根据数据源类型读取数据
        if isinstance(data_source, str):
            # 读取CSV文件
            data_content = self._read_csv_data(data_source, max_rows)
            if not data_content:
                return "无法读取数据文件"
        else:
            # 处理数据字典
            array_data = self._read_dataset_and_to_array(data_source, max_rows)
            print('====================================是字典=============================')
            if array_data.size == 0:
                return "无法处理数据字典"

            # 创建临时DataFrame并转换为CSV字符串
            # 获取列数
            num_cols = array_data.shape[1]
            # 生成列名: x1, x2, ..., xn-1, y
            columns = [f'independent variable{i+1}' for i in range(num_cols-1)] + ['y']

            # 创建DataFrame
            df = pd.DataFrame(array_data, columns=columns)
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            data_content = buffer.getvalue()


        if verbose:
            if isinstance(data_content, str):
                data_size = len(data_content)
                print(f"数据大小: {data_size} 字符")
            else:
                print(f"数据形状: {array_data.shape}")

        # 创建提示
        prompt = self._create_prompt(data_content, custom_prompt)

        if verbose:
            print_block("输入给DataAnalyzer的提示词：\n"+prompt)
            print("正在查询大模型...")

        # 获取分析结果
        result = self._query_model(prompt)

        if verbose:
            print("分析完成")
            json_residual_file = os.path.join(self.base_dir, "residual_analyze.json")
                                # 加载现有的初次分析数据（如果文件存在）
            data_list = []
            if os.path.exists(json_residual_file):
                try:
                    with open(json_residual_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                        if isinstance(existing_data, list):
                            data_list = existing_data
                except json.JSONDecodeError:
                    print(f"现有的初次分析JSON文件格式有误，将创建新文件")
                except Exception as e:
                    print(f"读取现有初次分析文件时出错: {e}")

            # 创建新的初次分析记录
            current_sample_order = 0  # 获取当前样本的顺序号


            # 创建初次分析数据结构
            residual_record = {
                "sample_order": current_sample_order,
                "island_id": 'this is the initial data',
                "equation": None,
                "analysis": result,
                "stats": {
                    "mean_residual": None,
                    "max_absolute_residual": None,
                    "std_residual": None
                },
            }

            # 添加到初次分析数据列表
            data_list.append(residual_record)

            # 保存更新后的初次分析数据
            try:
                with open(json_residual_file, "w", encoding="utf-8") as f:
                    json.dump(data_list, f, ensure_ascii=False, indent=2)
                print(f"成功更新初次分析JSON文件: {json_residual_file}")
            except Exception as e:
                print(f"保存初次分析JSON文件时出错: {e}")

        return result


# 兼容别名：旧模块名 drsr_420.data_analyse_real.DataAnalyzer 指向本类
DataAnalyzer = DataAnalyzerAgent
