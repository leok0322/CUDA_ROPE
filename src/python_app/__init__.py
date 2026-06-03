"""python_app 包：FusedQKVNormRope 的 PyTorch 参考实现与 torch.compile 融合替换驱动。

有了本 __init__.py，src/python_app 才被 Python 视作一个【包(package)】，
从而支持包内相对导入(from .model / from .util)与以包方式运行：
    cd src && python -m python_app.runner
（仍兼容脚本方式：cd src/python_app && python runner.py —— 走 runner.py 里的绝对导入兜底。）
"""

# import 本包时，Python 实际执行的就是本文件，下面三条 import 会在此刻运行。
# 把子模块里的符号【re-export(再导出)到包顶层】：用方无需关心它在哪个文件，
#   有了这几行即可 `from python_app import FusedQKVNormRope`(而非 from python_app.model import ...)。
from .model import FusedQKVNormRope      # . = 当前包；搬 model.py 的合并 qkv 模型类到顶层
from .installer import Installer            # 搬 installer.py 的通用 pass 安装器到顶层
from .rmsnorm_rope_replace_pass import RMSNormRoPEreplacePass     # 搬 rmsnorm_rope_replace_pass.py 的子图替换定义到顶层

# __all__：仅控制 `from app import *` 通配导入暴露哪些名字(并作为公共 API 清单)。
#   - 不影响显式导入(from app import X 仍可导入未列出的名字)；
#   - 没有 __all__ 时，import * 会带出所有非下划线开头的名字(含 import 进来的 torch 等)。
__all__ = ["FusedQKVNormRope", "Installer", "RMSNormRoPEreplacePass"]
