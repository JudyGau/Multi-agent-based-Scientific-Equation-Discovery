"""兼容层（@deprecated）：旧路径 ``llm``（根模块）→ 新路径 ``drsr_420.llm``。

历史原因：LLM 客户端曾是仓库根目录的顶层模块 ``llm.py``，包内代码用
``from llm import LLMClient`` 引用它。这有两个问题：

1. 包不再自包含（``drsr_420`` 依赖"仓库根在 sys.path 上"），也无法独立安装；
2. ``llm`` 是 PyPI 上的知名包名，一旦环境里装了它，``import llm`` 可能解析到
   **另一个库**，产生难以定位的导入错误。

现在实现位于 ``drsr_420/llm/``；本文件只做转发：

* 读：模块级 ``__getattr__``（含私有名与会被重新绑定的 GLOBAL_* 状态）；
* 写/删：把模块类换成 ``_ForwardingModule``，并且**写入转发到"定义该名字的子模块"**
  （``_impl.owner_module``）而不是门面——``LLMClient.chat`` 里的
  ``_post_with_retry`` 是在 ``client`` 模块的全局命名空间里查的，
  打在门面上等于没打（``mock.patch('llm._post_with_retry')`` 曾因此静默失效）。
"""
if __package__ in (None, ""):     # 支持 `python llm.py` 直接执行
    import sys as _sys2
    from pathlib import Path as _Path

    _sys2.path.insert(0, str(_Path(__file__).resolve().parent))

import sys as _sys
import types as _types

from drsr_420.llm import *          # noqa: F401,F403
from drsr_420 import llm as _impl


def _owner(name: str):
    """名字的归属模块（找不到则退回门面）。"""
    module = _impl.owner_module(name)
    return module if module is not None else _impl


class _ForwardingModule(_types.ModuleType):
    """属性读写与删除都转发到"定义该名字的子模块"。"""

    def __setattr__(self, name, value):
        setattr(_owner(name), name, value)

    def __delattr__(self, name):
        delattr(_owner(name), name)


_sys.modules[__name__].__class__ = _ForwardingModule


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__():
    return dir(_impl)


if __name__ == "__main__":
    from drsr_420.llm.__main__ import main as _smoke_main

    raise SystemExit(_smoke_main())
