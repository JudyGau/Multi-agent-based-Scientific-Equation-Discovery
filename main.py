"""命令行入口：``python main.py ...`` → ``drsr_420.cli.main``。

为什么根目录还留着一个 ``main.py``
----------------------------------
历史用法（`.idea/runConfigurations/*.xml` 的 4 个运行配置、`example.sh`、
`MRFCompress-3.sh`）都以 ``python main.py --problem_name ...`` 启动实验，这是本项目的
对外入口，不是历史遗留的转发层。因此它保留在这里，但实现只有一行委托：

* 真正的参数解析、日志与编排在 :mod:`drsr_420.cli.main`；
* 等价调用：``python -m drsr_420.cli.main``，或安装后的 ``drsr420`` 命令
  （``pyproject.toml`` 的 ``[project.scripts]``）。
"""
from __future__ import annotations

from drsr_420.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
