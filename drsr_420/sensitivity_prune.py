"""兼容层（@deprecated）：旧路径 ``drsr_420.sensitivity_prune`` → 新路径 ``drsr_420.analysis.sensitivity_prune``。

本文件只做转发、不含实现，且**读写都转发**：

* 读：模块级 ``__getattr__`` 转发所有名字（含私有名）；
* 写：把模块类换成"写入转发给实现模块"的 ``ModuleType`` 子类。只做读转发是不够的
  ——`mock.patch` / 测试里的 ``old_path.NAME = stub`` 只会落在本兼容层，实现模块
  看不到，打桩静默失效（MCP 工具与嵌入器单例的测试正是这样打桩的）。

请在新代码中使用新路径。
"""
if __package__ in (None, ""):     # 支持 `python 旧路径.py` 直接执行
    import sys as _sys2
    from pathlib import Path as _Path

    _sys2.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys as _sys
import types as _types

from drsr_420.analysis.sensitivity_prune import *            # noqa: F401,F403  触发子模块导入
from drsr_420.analysis import sensitivity_prune as _impl


class _ForwardingModule(_types.ModuleType):
    """属性读写与删除都转发到实现模块（属性读取由模块级 __getattr__ 处理）。

    ``__delattr__`` 同样必要：``mock.patch.object`` 靠 ``hasattr`` 判断"原本有没有
    这个属性"，有则在退出时 ``delattr`` 还原；只转发写入会让还原阶段抛
    AttributeError（属性实际删在了实现模块上）。
    """

    def __setattr__(self, name, value):
        setattr(_impl, name, value)

    def __delattr__(self, name):
        delattr(_impl, name)


_sys.modules[__name__].__class__ = _ForwardingModule


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__():
    return dir(_impl)
