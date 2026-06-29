# Fused QK-Norm + RoPE — CUDA Kernel + torch.compile 自定义算子

把 LLM 注意力之前的 **"Q/K RMSNorm + RoPE 旋转"** 融合成**一个 CUDA kernel**，并通过
`torch.compile` 的子图替换（pattern matcher）自动接入 PyTorch：模型里那段"分步的
RMSNorm→RoPE 子图"在编译期被替换成一次自定义算子
`torch.ops.ROPE_cuda.fused_qkv_norm_rope_neox`。相比未融合的 eager 子图，大规模下达到
**~12× 加速**，峰值约 **284 GFLOPS**（fp16，访存带宽瓶颈）。

**硬件环境**：NVIDIA GPU，sm_86（Ampere）
**数据类型**：float16 / bfloat16 / float32
**规模**：合并 qkv 输入 `[num_tokens, (Hq+Hk+Hv)·head_dim]`；
num_tokens ∈ {128, 512, 2048, 8192}，head_dim ∈ {64, 128, 256}，(Hq,Hk,Hv)=(8,8,8)

---

## 目录

- [算法：QK-Norm + RoPE](#算法qk-norm--rope)
- [CUDA Kernel 实现](#cuda-kernel-实现)
- [接入 PyTorch：torch.compile 子图替换](#接入-pytorchtorchcompile-子图替换)
- [性能：自定义算子 vs eager](#性能自定义算子-vs-eager)
- [为什么快](#为什么快)
- [脚本使用](#脚本使用)
- [BLOCK_SIZE_X 调优：为何不敏感](#block_size_x-调优为何不敏感)
- [构建](#构建)

---

## 算法：QK-Norm + RoPE

输入是**合并的 qkv** `[num_tokens, (Hq+Hk+Hv)·head_dim]`，每个 token 内按 `[Q…|K…|V…]` 连续排布。
算子**只处理 Q/K 头**（各 head 独立），**V 透传**：

1. **QK-RMSNorm**（逐 head，沿 head_dim 维）：

   $$\text{RMSNorm}(x_i) = \gamma_i \cdot \frac{x_i}{\sqrt{\frac{1}{H}\sum_j x_j^2 + \varepsilon}}$$

   与 LayerNorm 不同，RMSNorm **不做均值中心化、无 bias**，只按均方根缩放；Q 和 K 各有自己的 γ。

2. **RoPE 旋转**：对归一化后的 Q/K，按位置 `positions` 取 cos/sin，对维度成对 $(2i, 2i+1)$ 做旋转

   $$x'_{0}=x_0\cos-x_1\sin,\qquad x'_{1}=x_0\sin+x_1\cos$$

   支持两种配对风格：**neox**（前后半配对，已注册算子）与 **interleave**（相邻配对）。

整个算子是典型的 **memory-bound**：每元素算术量（平方、rsqrt、几次乘加）远小于其 HBM 访存量，
优化核心是**减少 HBM 往返、提高访存合并与向量化**。

> 算法细节：[`docs/algorithm/fused_ROPE_RMSNorm/ROPE_principle.txt`](docs/algorithm/fused_ROPE_RMSNorm/ROPE_principle.txt)、
> [`qknorm&rope.txt`](docs/algorithm/fused_ROPE_RMSNorm/qknorm&rope.txt)、
> [`rmsnorm_vs_layernorm.txt`](docs/algorithm/fused_ROPE_RMSNorm/rmsnorm_vs_layernorm.txt)、
> 输入布局 [`qkv_input_layout_ragged_vs_padded.txt`](docs/algorithm/fused_ROPE_RMSNorm/qkv_input_layout_ragged_vs_padded.txt)。

---

## CUDA Kernel 实现

`src/kernels/` 下渐进式的 4 个 kernel（C++ 端 `./validation <id>` 走 K0–K3，互为正确性对照）：

| id | 文件 | 说明 |
|----|------|------|
| K0 | `0_ROPE_kernel_base.cuh` | 单线程/朴素 RoPE 基准，非合并访问，作对照 |
| K1 | `1_ROPE_kernel_naive.cuh` | 多线程并行 RoPE，合并访问 |
| K2 | `2_ROPE_kernel_vectorize.cuh` | 向量化 load/store 的 RoPE |
| **K3** | **`3_fuesd_QKNorm_and_ROPE_kernel.cuh`** | **★融合 QK-RMSNorm + RoPE**，Python 自定义算子用的就是它 |

**K3（核心）的设计**：

- **warp-per-head**：一个 warp（32 lane）处理一个 (token, QK-head)，32 个 lane 均分 head_dim；
- **RMSNorm 规约**：lane 内 `__shfl_xor_sync` 蝶形规约求 $\sum x^2$（5 轮，无共享内存、无 `__syncthreads`）；
- **向量化 + 合并访存**：每 lane 一次宽向量读/写（`packed_as<float, N>` → float/float2/float4），
  32 lane 的并集正好平铺成本 head 的连续内存 → 完全合并；
- **就地改写**：直接在 qkv 上原地写回（算子 schema 标 `Tensor(a!)`），无额外输出缓冲；
- **(dtype, head_dim) 合法性守卫**：模板函数内用 `if constexpr (numPerFourBytes ≤ 4)` 在编译期裁掉
  非法组合（如 float×256 需要不存在的 `packed_as<float,8>`），见
  [`docs/c++/switch_case_fallthrough_and_warnings.txt`](docs/c++/switch_case_fallthrough_and_warnings.txt) 第九/十节。

> 渐进对比：[`kernel0_vs_kernel1.txt`](docs/algorithm/fused_ROPE_RMSNorm/kernel0_vs_kernel1.txt)、
> [`kernel2_vs_kernel0.txt`](docs/algorithm/fused_ROPE_RMSNorm/kernel2_vs_kernel0.txt)。

---

## 接入 PyTorch：torch.compile 子图替换

不直接手调算子，而是让 `torch.compile` **自动**把模型里的子图替换成融合 kernel（`src/app/`）：

```
FusedQKVNormRope.forward (eager 分步子图：RMSNorm→RoPE，作参考/被匹配)
        │  RMSNormRoPEreplacePass        定义 search_fn / replace_fn（"换什么"）
        │  Installer.install_fusion_pass  register_replacement + PatternMatcherPass + post_grad 钩子
        ▼
torch.compile(model)  ──编译期──▶  子图被替换成 torch.ops.ROPE_cuda.fused_qkv_norm_rope_neox(...)
```

- `model.py` 的 `FusedQKVNormRope`：合并 qkv、仅 QK 处理、V 透传、Q/K 分离权重，贴近真实 vLLM kernel；
- `runner.py` 的 `Runner`：配置 → 造输入 → 建模型 → eager 前向（参考）→ 装融合 pass → `torch.compile` 执行；
- 算子在 `src/interface.cpp` 用 `TORCH_LIBRARY(ROPE_cuda, …)` 注册，名为 `fused_qkv_norm_rope_neox`。

---

## 性能：自定义算子 vs eager

- **基准(baseline) = eager**：`FusedQKVNormRope.forward`（未融合分步子图）；
- **自定义算子 = `torch.compile(model)` 热路径**（子图已替换成一次融合 kernel）；
- CUDA Event 计时，WARMUP=10 + REPEATS=100 取中位数；`speedup = eager_median / op_median`。

**head_dim=128，(Hq,Hk,Hv)=(8,8,8)**（GFLOPS 越高越好；数据：sm_86，最近一次 benchmark）：

| num_tokens | fp16 op / eager / 加速 | bf16 op / eager / 加速 | fp32 op / eager / 加速 |
|-----------:|:----------------------:|:----------------------:|:----------------------:|
| 128        | 14.9 / 2.8 / **5.3×**  | 17.7 / 3.4 / **5.3×**  | 16.3 / 3.9 / **4.1×**  |
| 512        | 62.9 / 11.4 / **5.5×** | 60.7 / 12.5 / **4.9×** | 62.9 / 13.6 / **4.6×** |
| 2048       | 210.8 / 22.1 / **9.6×**| 251.5 / 22.0 / **11.4×**| 130.9 / 18.1 / **7.2×**|
| 8192       | 280.4 / 25.1 / **11.2×**| 280.4 / 25.2 / **11.1×**| 135.7 / 19.5 / **7.0×**|

**全扫描要点**（dtype × head_dim ∈ {64,128,256} × num_tokens）：

- **峰值**：fp16/bf16 在大 num_tokens 下约 **280–284 GFLOPS**；fp32 约 **135–142 GFLOPS**（每元素 2× 字节、带宽瓶颈，故约为半精度一半）。
- **加速比随规模增大**：num_tokens=128 时 ~4–5×，num_tokens≥2048 时 **~10–12×**（小规模受固定 launch/开销主导，大规模带宽饱和、eager 的多算子开销被充分拉开）。
- **正确性**：全部组合 `allclose` 通过（fp32 atol/rtol=1e-3，fp16=1e-2，bf16=3e-2）。
- **非法组合**：float×256（需 `packed_as<float,8>`）被算子守卫优雅拒绝（SKIP），不计失败。

折线图（op 实线○ / eager 虚线△，横轴 num_tokens）：

```bash
./src/scripts/fused_ROPE_RMSNorm/run_plot_op.sh            # → plot_output/op_benchmark_gflops.png
```

---

## 为什么快

| 原因 | 说明 |
|------|------|
| **融合消除中间张量** | eager 子图含 RMSNorm + 多个 RoPE elementwise + reshape，多次 kernel 启动、中间结果反复落 HBM；融合后**一个 kernel** 一次读、一次写，中间量只在寄存器里流转 |
| **单次 HBM 读 + 就地写回** | 算子 `Tensor(a!)` 原地改写 qkv，省掉输出缓冲与一遍写 |
| **完全合并 + 向量化访存** | warp 的 32 lane 并集平铺成连续内存，每 lane 一次宽向量(float2/float4)读写 → 打满 HBM 带宽 |
| **warp 内规约，无 SMEM/同步** | RMSNorm 的 $\sum x^2$ 用 `__shfl_xor_sync` 在 warp 内完成，省掉共享内存与 `__syncthreads` |
| **规模越大越划算** | 固定 launch/框架开销被摊薄；大 num_tokens 下 warp 数过剩、带宽饱和 → 逼近 roofline |

瓶颈在**访存带宽**：op 已是 eager 的 ~11×，再提升要从"减少访存/提高合并粒度"入手（每 warp 处理更多
head/token、向量化宽度、grid 映射），而非线程块大小（见下）。

---

## 脚本使用

脚本都在 `src/scripts/fused_ROPE_RMSNorm/`，从项目根调用即可（脚本内部自动定位项目根）。

### Python 自定义算子（torch.compile 路径）

```bash
# 1) 正确性：多维度扫描 dtype × head_dim × interleave × token_size × num_heads，逐组合 allclose
./src/scripts/fused_ROPE_RMSNorm/run_check_op.sh
./src/scripts/fused_ROPE_RMSNorm/run_check_op.sh --dtype float16 --head-dim 128   # 只测子集

# 2) 性能：自定义算子 vs eager 耗时/GFLOPS，结果追加写 benchmark_results/ROPE_python_op_benchmark_result.txt
./src/scripts/fused_ROPE_RMSNorm/run_benchmark_op.sh
./src/scripts/fused_ROPE_RMSNorm/run_benchmark_op.sh --token-sizes 256 1024 4096

# 3) 画图：把 benchmark 结果画成折线图 → plot_output/op_benchmark_{metric}.png
./src/scripts/fused_ROPE_RMSNorm/run_plot_op.sh                 # GFLOPS
./src/scripts/fused_ROPE_RMSNorm/run_plot_op.sh --metric speedup

# 4) 自动调优 BLOCK_SIZE_X：sed 改宏 → 重编译 .so → benchmark → 取最佳（见下节）
./src/scripts/fused_ROPE_RMSNorm/autotune_block_size_x.sh
```

### C++ 原始 kernel（`./validation` 路径，K0–K3）

```bash
./src/scripts/fused_ROPE_RMSNorm/run.sh                 # 跑 kernel（改脚本里 seq 范围选 id），日志写 logs/
./src/scripts/fused_ROPE_RMSNorm/autotune_kernel0.sh    # 扫 BLOCK_SIZE，结果写 autotune/
./src/scripts/fused_ROPE_RMSNorm/plot_performance.sh    # 画 C++ benchmark 结果
./src/scripts/fused_ROPE_RMSNorm/run_visualization.sh   # RoPE 旋转可视化
```

> 多数 `run_*.sh` 都透传 `"$@"` 给底层 python 脚本，可加 `--dtype/--head-dim/--token-sizes/--metric` 等过滤。

---

## BLOCK_SIZE_X 调优：为何不敏感

`autotune_block_size_x.sh` 扫 BLOCK_SIZE_X ∈ {32,64,128,256,512,1024}，结果**几乎完全一样**
（op median 逐位相同）。这**不是 bug**（已确认每轮真重编、无外部 `-D` 覆盖），而是物理上不敏感：

- warp-per-head 设计下，BLOCK_SIZE_X **只改"每 block 装几个 warp"**，不改总 warp 数 / 总访存 / 每 warp 工作量；
- kernel 是**带宽瓶颈**，且 warp 数（如 65536）**远超饱和点** → 任意 block 大小都把带宽打满 → 时间不变；
- 真正让 block 大小生效的机制（共享内存分块 / occupancy 受限 / block 内协作规约）本 kernel **都没有**。

> 完整分析：[`docs/algorithm/fused_ROPE_RMSNorm/block_size_x_autotune_insensitivity.txt`](docs/algorithm/fused_ROPE_RMSNorm/block_size_x_autotune_insensitivity.txt)

---

## 构建

```bash
# 构建 Python 扩展 ROPE_cuda.*.so（CMake LIBRARY_OUTPUT_DIRECTORY = 项目根）
cmake -DCMAKE_BUILD_TYPE=Release -G Ninja -S . -B cmake-build-release
cmake --build cmake-build-release --target ROPE_cuda -- -j"$(nproc)"
# 同时可构建 C++ 验证可执行 validation
cmake --build cmake-build-release --target validation -- -j"$(nproc)"
```

> ⚠️ 改了 `src/kernels/*` 或 `src/dispatch.h` 后，跑 Python 脚本前**务必重新构建 `.so`**，否则测到的是旧 kernel。

调用示例（Python）：

```python
import torch, ROPE_cuda   # import 即 dlopen → 触发 TORCH_LIBRARY 注册
# qkv 就地改写：[num_tokens, (Hq+Hk+Hv)*head_dim]
torch.ops.ROPE_cuda.fused_qkv_norm_rope_neox(
    qkv, q_weight, k_weight, cos, sin,
    num_heads_q, num_heads_k, num_heads_v, head_dim, eps)
```
