"""
最小复刻 vLLM 思路：
  eager 里写"分步算子" → torch.compile 时用 hook 改计算图
  → 把分步子图替换成一次自定义 CUDA kernel 调用。

用 vLLM 里真实存在的 silu_and_mul（SiLU 后逐元素相乘）做例子，
因为它正好是"两个算子融合成一个 kernel"。

需要：CUDA GPU + nvcc。
运行： python fused_silu_mul_compile_hook.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import torch._inductor.config as inductor_config
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
    register_replacement,
    fwd_only,
)

# ============================================================================
# 第 1 步：写 CUDA kernel，并用 TORCH_LIBRARY 注册成 torch 算子
#          —— 对应 vLLM 的 csrc/*.cu + csrc/torch_bindings.cpp
# ============================================================================
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// __global__ kernel：真正在 GPU 上跑的代码（对应 fusedQKNormRopeKernel）
__global__ void silu_and_mul_kernel(float* out, const float* x,
                                    const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float s  = xi / (1.0f + expf(-xi));   // SiLU(x) = x * sigmoid(x)
        out[i]   = s * y[i];                   // 再逐元素乘 y —— 两步融合进一个 kernel
    }
}

// host 函数：CPU 上跑，负责算 grid/block 并发射 kernel（对应 host 端 fused_qk_norm_rope）
// at::Tensor 与 torch::Tensor 是【同一个类型】：torch 头文件里有 `using at::Tensor;`，
//   torch::Tensor 只是 at::Tensor 的别名(static_assert(is_same_v<...>) 成立)。
//   写 at:: 是贴近底层 ATen 的风格(配 at::empty_like)，换成 torch::Tensor 一字不差也能编译。
//   （注意：torch::tensor 小写是"造张量的工厂函数"，不是类型；类型是大写 Tensor。
//    另：vLLM 稳定 ABI 版用的 torch::stable::Tensor 是另一回事——跨版本稳定接口，非别名。）
at::Tensor silu_and_mul(const at::Tensor& x, const at::Tensor& y) {
    auto out = at::empty_like(x);
    int n = x.numel();
    int threads = 256, blocks = (n + threads - 1) / threads;
    silu_and_mul_kernel<<<blocks, threads>>>(
        out.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(), n);
    return out;
}

// 注册算子 schema + CUDA 实现 → 之后可用 torch.ops.myops.silu_and_mul
// TORCH_LIBRARY 第一个参数 myops = 算子命名空间(→ torch.ops.myops)，
//   与 load_inline 的 name="myops"(.so 文件名)无必然关系，此处同名只是巧合。
TORCH_LIBRARY(myops, m) {
    // def 只声明 schema(名字+参数+返回类型)，不绑定实现
    m.def("silu_and_mul(Tensor x, Tensor y) -> Tensor");
}
TORCH_LIBRARY_IMPL(myops, CUDA, m) {
    // impl 把【字符串名 "silu_and_mul"】绑定到【函数地址 &silu_and_mul】：
    //   - "silu_and_mul"  → 算子全名 myops::silu_and_mul(查表用的 key)
    //   - &silu_and_mul   → 该 host 函数编译进 .so 后机器码的入口地址(value)
    //   - CUDA            → 分发键：在 CUDA 张量上调用时走这个实现
    // 调用 torch.ops.myops.silu_and_mul(x,y) 时，Dispatcher 按 "myops::silu_and_mul"
    //   + 设备(CUDA) 查表拿到这个地址，跳进 .so 执行 silu_and_mul 那段机器码。
    m.impl("silu_and_mul", &silu_and_mul);
}
"""

# load_inline 编译上面的源码成 .so 并 import（import 即 dlopen，触发 TORCH_LIBRARY 注册）
#   —— 对应 vLLM 编译期 + 运行期的 `import vllm._C`
load_inline(
    # name 只决定【.so 文件名 / 缓存目录名】：~/.cache/torch_extensions/.../myops/myops.so
    #   （上次清缓存 rm -rf ~/.cache/torch_extensions/*/myops* 匹配的就是它）
    #   它与算子命名空间 torch.ops.myops 无必然关系（后者由 TORCH_LIBRARY(myops) 决定），
    #   此处同名纯属巧合——把 name 改成 "my_build"，算子仍是 torch.ops.myops.silu_and_mul。
    name="myops",
    cpp_sources="",
    cuda_sources=CUDA_SRC,
    with_cuda=True,
    is_python_module=False,   # 关键：只 dlopen 触发 TORCH_LIBRARY 注册，不当 Python 模块导入
                              #   设 False → torch 用 torch.ops.load_library() 纯 dlopen，
                              #   不要求 .so 里有 PyInit_myops；故 name 不是可 import 的模块名
    verbose=False,            # 否则会找 PyInit_myops（pybind11 入口），而本源码只有 TORCH_LIBRARY
)

# fake（meta）实现：只推断输出 shape/dtype，不真算 —— torch.compile 必需
#   对应 vLLM torch_bindings.cpp 的 CompositeExplicitAutograd 块 / _custom_ops.py 的 @register_fake
#   缺它则编译期在 FakeTensor 上分发失败：UnsupportedOperatorException: myops.silu_and_mul
@torch.library.register_fake("myops::silu_and_mul")
def _silu_and_mul_fake(x, y):
    return torch.empty_like(x)    # 输出形状/类型与 x 相同（逐元素算子）

# ============================================================================
# 第 2 步：定义"未融合"模式 和 "融合后"替换 —— 对应 QkNormRopePattern 的
#          Unfused(search) / Fused replacement(replace)
# ============================================================================
def search_fn(x, y):           # 要被匹配的"分步"写法
    return F.silu(x) * y

def replace_fn(x, y):          # 替换成一次自定义 kernel 调用
    return torch.ops.myops.silu_and_mul(x, y)

# ============================================================================
# 第 3 步：把"匹配-替换"注册进一个 PatternMatcherPass
#          —— 对应 QKNormRoPEFusionPass（一个 VllmPatternMatcherPass）
# ============================================================================
pm_pass = PatternMatcherPass()
example_inputs = [
    torch.randn(4, 8, device="cuda"),
    torch.randn(4, 8, device="cuda"),
]
register_replacement(
    search_fn,
    replace_fn,
    example_inputs,
    fwd_only,        # 只在前向图上匹配（对应 vLLM 的 pm.fwd_only）
    pm_pass,         # 把这条规则塞进 pm_pass（对应 self.passes += [QKNormRoPEFusionPass]）
)

# ============================================================================
# 第 4 步：把 pass 挂到 Inductor 的 post_grad 钩子
#          —— 对应 PostGradPassManager 被设为 post_grad_custom_post_pass
# ============================================================================
def custom_post_grad_pass(graph):
    count = pm_pass.apply(graph)         # 对应 PostGradPassManager.__call__ 里 for pass_: pass_(graph)
    if count:
        print(f"[hook] 融合发生：替换了 {count} 处 silu*y → silu_and_mul kernel")

inductor_config.post_grad_custom_post_pass = custom_post_grad_pass

# ============================================================================
# 第 5 步：eager 模型写"分步算子"，用 torch.compile 包起来
#          —— 对应 @support_torch_compile 装饰的模型
# ============================================================================
class MyModel(nn.Module):
    def forward(self, x, y):
        return F.silu(x) * y          # eager 时这是两步；compile 后被换成 kernel

model = MyModel().cuda()
compiled = torch.compile(model)        # 触发编译时会回调 custom_post_grad_pass

# ============================================================================
# 验证
# ============================================================================
if __name__ == "__main__":
    x = torch.randn(4, 8, device="cuda")
    y = torch.randn(4, 8, device="cuda")

    ref = model(x, y)                  # eager（分步）
    out = compiled(x, y)               # compiled（kernel）
    print("最大误差:", (ref - out).abs().max().item())
    assert torch.allclose(ref, out, atol=1e-5)
    print("✓ 编译后走的是自定义 CUDA kernel，结果与 eager 一致")
