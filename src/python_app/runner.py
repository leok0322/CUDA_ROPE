"""runner.py —— 用一个 Runner 类驱动 model.py 的 FusedQKVNormRope，并用 torch.compile
   把 "合并 qkv 的 QK-Norm + RoPE 分步子图" 替换成一次自定义算子
   torch.ops.ROPE_cuda.fused_qkv_norm_rope_{neox,interleave}。

Runner 封装了：配置 → 造输入 → 建模型 → eager 前向(不反向) → 安装融合 pass →
              torch.compile 执行 的整条流程（对应 docs/fork/fused_silu_mul_compile_hook.py）。

运行：
  source /home/liam/python_linux/python_venv/.venv/bin/activate
  cd src/app && uv run python runner.py
"""
import os
import sys

import torch

# util.py，同样兼容【包方式】与【脚本方式】两种运行（理由见下方 model 的导入注释）：
#   包方式(python -m app.runner)走 from . import util；脚本方式(python runner.py)
#   相对导入抛 ImportError，退回裸名 import util（src/app 在 sys.path[0]，同目录可找到）。
try:
    from . import util
except ImportError:
    import util

# ── import ROPE_cuda 前的两件事 ──────────────────────────────────────────────
# (a) 让解释器找得到项目根目录下的 .so（CMake LIBRARY_OUTPUT_DIRECTORY=项目根）。
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# (b) 必须【先 import torch（上面已 import）再 import ROPE_cuda】：
#     ROPE_cuda.so 依赖 libtorch*，且 TORCH_LIBRARY 注册需要 torch 的 Dispatcher 已就位。
#     （详见 docs/python/TORCH_LIBRARY_vs_PYBIND11.txt 情况 1 补充 2。）
try:
    import ROPE_cuda  # noqa: E402,F401  import 即 dlopen → 触发 TORCH_LIBRARY 注册
    _HAS_ROPE_CUDA = True
except Exception as e:  # 未编译 / 未在 sys.path / 依赖未就位
    print(f"[warn] import ROPE_cuda 失败：{e}\n"
          f"       请先用 CMake 构建出 ROPE_cuda.*.so 并置于 {PROJECT_ROOT}。\n"
          f"       Runner 仍会跑 eager 前向，但跳过 torch.compile 融合替换。")
    _HAS_ROPE_CUDA = False

# 导入 model.py 中的模型类，兼容【两种运行方式】：
#   1) 作为包跑(python -m app.runner)：runner 属于 app 包，__package__="app"，
#      用相对导入 from .model（. = 当前包）能找到同包的 model 模块；
#   2) 当脚本直接跑(cd src/app && python runner.py)：runner 是顶层脚本，
#      不属于任何包，相对导入会抛 ImportError("attempted relative import
#      with no known parent package")，于是落到 except，改用裸名绝对导入
#      from model —— 此时 src/app 已被加进 sys.path[0]，同目录的 model.py 可直接找到。
# 先试相对导入、失败再退绝对导入，两种启动方式都能定位到 model.py。
try:
    from .model import FusedQKVNormRope
except ImportError:
    from model import FusedQKVNormRope

# Installer(通用 pass 安装器) 与 RMSNormRoPEreplacePass(子图替换内容定义)，兼容包/脚本两种运行方式。
# 注：两条兜底都用【裸名】绝对导入(脚本方式下 src/app 在 sys.path[0])，
#     勿混入 CUDA_ROPE.src.app.* 深路径——脚本方式找不到、会再次 ImportError。
try:
    from .installer import Installer
    from .rmsnorm_rope_replace_pass import RMSNormRoPEreplacePass
except ImportError:
    from installer import Installer
    from rmsnorm_rope_replace_pass import RMSNormRoPEreplacePass


class Runner:
    """封装 FusedQKVNormRope 的 eager 运行与 torch.compile 融合替换全流程。"""

    def __init__(self, batch=2, token_size=4, head_dim=128,
                 eps=1e-6, base=10000.0, max_token_size=4096, device=None, dtype=torch.float32,
                 seed=0, interleave=False,
                 num_heads_q=8, num_heads_k=8, num_heads_v=8):
        # ---- 配置 ----
        self.batch = batch
        self.token_size = token_size
        self.head_dim = head_dim          # = 最后一维（RMSNorm/RoPE 作用维），须为偶数
        self.eps = eps
        self.base = base
        self.dtype = dtype
        self.seed = seed
        # RoPE 风格：模型与 replace_pass 的 search_fn 须用同一个值，否则计算图对不上
        self.interleave = interleave
        # 合并 qkv 模型(FusedQKVNormRope)：输入 [num_tokens,(Hq+Hk+Hv)*head_dim]，
        #   仅 QK 处理、V 透传、Q/K 分离权重，贴近真实 vLLM kernel；torch.compile 走方案 A。
        self.num_heads_q = num_heads_q
        self.num_heads_k = num_heads_k
        self.num_heads_v = num_heads_v
        # device 兜底：传了就用传入的，没传(None)则自动选 GPU 优先、CPU 兜底。
        #   - or：Python 的 or 返回【操作数本身】而非 True/False，且短路——
        #     左边 device 为真值就直接返回它，右边那串(含 is_available())不求值；
        #     device 为假值(None/""/0)才取右边。这里 device 只会是 None 或设备串，故够用。
        #   - "cuda" if cond else "cpu"：三元表达式，is_available() 探测本机有无可用 GPU。
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 把 (batch, token) 折叠成一个 token 维 → 与 CUDA kernel 的
        # [num_tokens, num_heads, head_dim] 对齐
        self.num_tokens = batch * token_size
        self.half = head_dim // 2
        # 模型的上下文最大token
        self.max_position = max_token_size

        # ---- 运行期状态（惰性构建）----
        self.model = None


    # ------------------------------------------------------------------ 模型
    def build_model(self):
        # 合并 qkv 版：仅 QK 处理、V 透传、Q/K 分离权重
        self.model = FusedQKVNormRope(
            num_heads_q=self.num_heads_q, num_heads_k=self.num_heads_k,
            num_heads_v=self.num_heads_v, head_dim=self.head_dim,
            rotary_dim=self.head_dim, max_position=self.max_position,
            eps=self.eps, base=self.base, interleave=self.interleave,
            dtype=self.dtype, device=self.device,
        ).to(self.device)
        return self.model


    # ------------------------------------------------------------- 输入校验
    @staticmethod
    def _validate_inputs(x, positions):
        """x / positions 任一为空就报错并抛参数异常（三个 run* 方法共用）。"""
        if x is None or positions is None:
            print("[error] 输入 x 或 positions 为空，无法运行；请传入有效张量。")
            raise ValueError("run/run_eager/run_compiled 需要非空的 x 与 positions")

    # ----------------------------------------------------- 第 1 步：eager 前向
    def run_eager(self, x=None, positions=None):
        """实例化 model，eager 前向得到参考输出（不做反向）。"""
        self._validate_inputs(x, positions)
        if self.model is None:
            self.build_model()
        with torch.no_grad():                 # 只前向，不反向
            y_ref = self.model(x, positions)
        print(f"[eager] x: {tuple(x.shape)} -> y: {tuple(y_ref.shape)}  "
              f"(num_tokens={self.num_tokens}=batch*token_size)")
        return y_ref

    # ------------------------------------------- 第 5 步：torch.compile 执行
    def run_compiled(self, x=None, positions=None):
        """安装融合 pass，torch.compile(model) 并执行，返回编译后输出 y。"""
        self._validate_inputs(x, positions)
        if self.model is None:
            self.build_model()
        # is_neox/interleave 不作算子入参，而是【按其值选不同的算子】：
        #   neox(interleave=False) 与 interleave(True) 各对应一个已注册的融合算子。
        op_name = ("ROPE_cuda::fused_qkv_norm_rope_interleave"
                   if self.interleave
                   else "ROPE_cuda::fused_qkv_norm_rope_neox")
        # 第 2 步：RMSNormRoPEreplacePass 只定义"换什么"(search/replace 的语义 + 替换目标 +
        #   example_inputs 的结构)。构造函数只持有服务 search/replace 的长期状态。
        #   合并 qkv 模型：传 num_heads_q/k/v 与 head_dim，供 search 切 Q/K/V、replace 烤进算子实参。
        rp = RMSNormRoPEreplacePass(
            eps=self.eps, num_heads_q=self.num_heads_q, num_heads_k=self.num_heads_k,
            num_heads_v=self.num_heads_v, head_dim=self.head_dim,
            op_name=op_name,                       # 据 interleave 选定的算子
            interleave=self.interleave,            # 与 build_model 的 RoPE 风格保持一致
        )
        # 第 3~4 步：example_inputs 的结构归 rp，所需【运行期规格】(num_tokens/device/dtype)
        #   由 Runner 传入；连同 search/replace 一起注入通用安装器 Installer，由它完成
        #   register_replacement + PatternMatcherPass + post_grad 钩子。
        example_inputs = rp.make_example_inputs(self.num_tokens, self.device, self.dtype)
        installer = Installer(rp.search_fn, rp.replace_fn, example_inputs, rp.op_name)
        installer.install_fusion_pass()
        compiled = torch.compile(self.model)   # 编译时回调 custom_post_grad_pass
        with torch.no_grad():
            y = compiled(x, positions)
        return y

    # --------------------------------------------------- 编排：跑完整流程
    def run(self, x=None, positions=None):
        """跑完整流程，统一返回 (y, y_ref)；无 compiled 输出时 y 为 None。"""
        self._validate_inputs(x, positions)
        y_ref = self.run_eager(x, positions)               # 第 1 步：eager 前向（不反向）
        if not _HAS_ROPE_CUDA:
            print("[skip] 未加载 ROPE_cuda，跳过 torch.compile 融合替换。")
            return None, y_ref
        try:
            y = self.run_compiled(x, positions)            # 第 2~5 步：融合替换 + 编译执行
            return y, y_ref
        except Exception as e:
            print(f"[warn] torch.compile 融合替换未完成：{e}\n"
                  f"       多半因 fused_qkv_norm_rope_{{neox,interleave}} 尚未注册/实现。请先在 "
                  f"interface.cpp 注册算子，schema 为 (Tensor qkv, Tensor q_weight, "
                  f"Tensor k_weight, Tensor cos, Tensor sin, int num_heads_q, int num_heads_k, "
                  f"int num_heads_v, int head_dim, float eps) -> Tensor 并实现 kernel。")
            return None, y_ref


if __name__ == "__main__":
    # 从 config.yaml 读取配置(util.load_config 已把 dtype 字符串映射成 torch.dtype)
    cfg = util.load_config()
    runner = Runner(**cfg)                     # 配置键与 Runner.__init__ 形参一一对应
    # 用 runner 自身的配置造【合并 qkv】输入，避免与 YAML 不一致
    x, positions = util.make_inputs_qkv(
        runner.seed, runner.num_tokens, runner.num_heads_q, runner.num_heads_k,
        runner.num_heads_v, runner.head_dim, runner.device, runner.dtype,
        runner.token_size, runner.batch,
    )
    y, y_ref = runner.run(x, positions)        # run 内部已跑 eager，统一返回 (y, y_ref)
    # 用 allclose 比对 compiled 输出与 eager 参考输出是否一致
    if y is not None:
        # allclose 逐元素判 |y_ref - y| <= atol + rtol*|y|，全部满足才返回 True：
        #   atol(绝对容差)主导接近 0 的值；rtol(相对容差)按量级放宽、主导大数值。
        #   两者结合覆盖全量级。这里设 1e-3 是较宽松的千分级容差，容忍融合 kernel 与
        #   eager 在浮点累加顺序/实现/精度上的微小偏差(右边的 |y| 用第二参数 y，不对称)。
        same = torch.allclose(y_ref, y, atol=1e-3, rtol=1e-3)
        print(f"[compiled] 与 eager 是否一致(allclose): {same}")
    else:
        print("[compiled] 未产生 compiled 输出，跳过比对。")
