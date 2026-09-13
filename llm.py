"""兼容模块：``llm``（根模块）→ ``drsr_420.llm``。

历史包袱与被取代的方式
----------------------
LLM 客户端原本是仓库根目录的单模块 ``llm.py``，包内代码用 ``from llm import LLMClient``
引用它。这有两个问题：

1. 包不再自包含（``drsr_420`` 依赖"仓库根在 sys.path 中"），也无法独立安装；
2. ``llm`` 是 PyPI 上的知名包名，环境里一旦装了它，``import llm`` 可能解析到
   **另一个库**，产生难以定位的导入错误。

现在实现位于 ``drsr_420/llm/``，包内代码一律用规范路径导入。本文件只为仍在
``import llm`` 的外部脚本保留**公开名字**的再导出：

* 公开 API：``__all__`` 里的名字（``LLMClient`` / ``ClientFactory`` / 各 provider 等）；
* 私有名（``_post_with_retry``、``GLOBAL_TIME_SECONDS`` 之类）**不再转发**：
  它们属于 ``drsr_420.llm.client`` / ``drsr_420.core.llm_stats``。打桩请打在定义处
  （``mock.patch('drsr_420.llm.client._post_with_retry')``），打在门面上等于没打
  ——``LLMClient.chat`` 是在 ``client`` 模块的全局命名空间里查找这个名字的。
"""
from __future__ import annotations

from drsr_420.llm import *                      # noqa: F401,F403  仅再导出公开 API
from drsr_420.llm import __all__ as _CANONICAL_API

__all__ = list(_CANONICAL_API)


if __name__ == "__main__":
    # 自检冒烟：真实向 provider 发一次请求（`python llm.py`）
    from drsr_420.llm.__main__ import main as _smoke_main

    raise SystemExit(_smoke_main())
