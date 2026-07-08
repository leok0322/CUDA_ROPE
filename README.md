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
- [NHeads op 算法 0](#nheads-op-算法-0)
- [NHeads op 算法 1](#nheads-op-算法-1)
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

`src/kernels/` 下包含渐进式 K0–K3 和 NHeads 实验实现（C++ 端 `./validation <id>` 走 K0–K3，互为正确性对照）：

| id | 文件 | 说明 |
|----|------|------|
| K0 | `0_ROPE_kernel_base.cuh` | 单线程/朴素 RoPE 基准，非合并访问，作对照 |
| K1 | `1_ROPE_kernel_naive.cuh` | 多线程并行 RoPE，合并访问 |
| K2 | `2_ROPE_kernel_vectorize.cuh` | 向量化 load/store 的 RoPE |
| **K3** | **`3_fuesd_QKNorm_and_ROPE_kernel.cuh`** | **one-head 融合 QK-RMSNorm + RoPE**：一个 warp 处理一个 head，是当前性能对照基线 |
| **NHeads A0** | **`4_fused_QKNorm_ROPE_NHeads_kernel.cuh`** | **一个 warp 处理同一 token 的多个 Q/K heads**；算法 0 用 `cp.async` 双缓冲预取 QKV，当前默认宏启用 |
| **NHeads A1** | **`4_fused_QKNorm_ROPE_NHeads_kernel.cuh`** | 在 A0 基础上继续把 cos/sin 用 `cp.async` 预取到 dynamic SMEM，尝试让 QKV 加载、cos/sin 加载和 RMSNorm/RoPE 计算形成更深流水 |

当前 `src/kernels/common.cuh` 的默认宏是：

```cpp
#define is_MULTI_HEAD_PER_WARP 1
#define HEADS_PER_WARP 32
#define ALGORITHM 0
```

因此 Python 自定义算子默认走 **NHeads A0**；把 `is_MULTI_HEAD_PER_WARP` 改成 `0` 后，才会回到 K3
one-head kernel。若要测试 NHeads A1，把 `ALGORITHM` 改成 `1` 并重新构建 `.so`。

**K3 one-head 的设计**：

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

## NHeads op 算法 0

NHeads 算法 0 的目标是让一个 warp 连续处理同一 token 的多个 Q/K heads，并尝试把
`next head` 的 QKV 加载和 `current head` 的 RMSNorm/RoPE 计算重叠。

核心映射：

- 一个 warp 负责同一 token 内最多 `HEADS_PER_WARP` 个 Q/K heads；
- 一个 block 含 `BLOCK_SIZE_X / 32` 个 warp；
- grid 按 token 独立切分：每个 token 需要 `ceil((num_heads_q + num_heads_k) / HEADS_PER_WARP)` 个 warp；
- V heads 不参与 QK-Norm/RoPE，仍然透传。

算法 0 的数据路径：

```text
QKV:          HBM -> cp.async -> dynamic SMEM 双缓冲 -> register -> RMSNorm/RoPE -> HBM
cos/sin:      HBM -> register
q/k weight:   HBM -> register
```

dynamic SMEM 只给 QKV 双缓冲使用。每个 warp 只需要 current/next 两个 head slot：

```cpp
dynamicSMEM =
    warpsPerBlock * 32 * 2 * elementBytesPerHeadPerThread;
```

其中 `2` 是双缓冲 channel 数，不是 `HEADS_PER_WARP`。SMEM layout 按
`[warp][buffer][lane][element]` 排布，使同一 warp 内每个 lane 读回自己负责的连续片段，降低 bank
conflict 风险。QKV 的 HBM 指针也是 warp 内连续地址，能形成合并访问。

关键限制是：当前 QKV 数据没有 block/warp 内复用。每个 lane 预取的是自己后续要消费的片段，因此
SMEM 在这里主要承担异步流水的中转，而不是共享缓存。只有当 `cp.async` 隐藏掉的 HBM 延迟大于
`SMEM -> register` 读回、`wait_group`、双缓冲索引和额外控制逻辑的成本时，NHeads A0 才会比
one-head 更快。

更详细的数据加载策略见
[`docs/algorithm/fused_ROPE_RMSNorm/one_warp_n_heads_data_loading_strategy.txt`](docs/algorithm/fused_ROPE_RMSNorm/one_warp_n_heads_data_loading_strategy.txt)。

---

## NHeads op 算法 1

NHeads 算法 1 在算法 0 的基础上进一步把当前 token 的 `cos/sin` 也搬到 dynamic SMEM。它的目标是把
QKV 预取、cos/sin 预取和当前 head 的 RMSNorm/RoPE 计算交错起来，减少直接从 HBM 读
`cos/sin` 时暴露出的延迟。

算法 1 的数据路径：

```text
QKV:          HBM -> cp.async -> dynamic SMEM 双缓冲 -> register -> RMSNorm/RoPE -> HBM
cos/sin:      HBM -> cp.async -> dynamic SMEM(per-warp) -> register
q/k weight:   HBM -> register
```

dynamic SMEM 需要手动切分成两段：

```text
[QKV double buffer][warp0 cos][warp0 sin][warp1 cos][warp1 sin]...
```

QKV 仍然使用 `[warp][buffer][lane][element]` 的双缓冲 layout；cos/sin 不按 head 再开 buffer，
因为同一 token 的所有 Q/K head 共用同一行 cos/sin。cos/sin 是独立 tensor，所以分别搬
`cos[token, :]` 和 `sin[token, :]`。

执行流水的核心直觉：

```text
预取 head0 QKV
预取 token 对应 cos/sin

for current head:
  预取 next head QKV
  等 current head QKV 完成
  计算 current head RMSNorm
  等 cos/sin 完成
  计算 current head RoPE
  写回 current head
```

RoPE 前的等待要区分是否已经发出了 next QKV：

```cpp
if (i + 1 < headNumInWarp) {
  // 最新 group 是 next QKV，允许它继续在途，只等待更老的 cos/sin。
  cp_async_wait_group<1>();
} else {
  // 没有 next QKV 时，cos/sin 可能就是最新 group，必须全部等完。
  cp_async_wait_group<0>();
}
__syncwarp();
```

注意 `cp.async` 单条拷贝宽度只能是 4/8/16B。当前算法 1 按 16B chunk 搬 cos/sin，因此要求
每行 `half * sizeof(cache_scalar_t)` 是 16B 的整数倍；不满足时，launch 端应回退到算法 0，
否则最后一笔 16B 拷贝会越过该行有效范围。算法 1 的收益只来自流水隐藏延迟；QKV 本身仍然没有
跨线程复用，因此如果 SMEM 中转、`commit/wait`、地址计算和同步成本超过隐藏掉的 HBM 延迟，
算法 1 就不一定比 one-head 或 A0 更快。

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

**head_dim=128，(Hq,Hk,Hv)=(8,8,8)**（GFLOPS 越高越好；数据：one-head op 基线，sm_86）：

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
- **模板守卫**：若某个 `(dtype, head_dim)` 组合需要超过已特化向量类型的访问宽度，会在模板守卫处
  拒绝并记为 SKIP，不计失败；one-head/NHeads 对比只统计两边都实际产出的共同配置。

**one-head / NHeads A0 / NHeads A1 对比**（按相同 `(dtype, head_dim, num_tokens, heads)` 对齐比较；
0.5% 以内记为基本持平）：

| 对比 | 共同配置 | 后者更快 | 前者更快 | 基本持平 | 后者 / 前者几何平均速度比 |
|------|---------:|---------:|---------:|---------:|--------------------------:|
| one-head → NHeads A0 | 36 | 6 | 21 | 9 | 0.953× |
| NHeads A0 → NHeads A1 | 36 | 22 | 5 | 9 | 1.047× |
| one-head → NHeads A1 | 36 | 19 | 11 | 6 | 0.997× |

按 `head_dim` 聚合，表中 `A / B` 定义为 `B_ms / A_ms`，大于 1 表示左侧的 `A` 更快：

| 对比 | hd=64 | hd=128 | hd=256 |
|------|------:|-------:|-------:|
| NHeads A1 / NHeads A0 | 1.112× | 1.101× | 0.937× |
| NHeads A1 / one-head | 1.099× | 0.999× | 0.904× |

结论：

- **A1 明显优于 A0 的主要区间是 `head_dim=64/128`**：A1 多预取 cos/sin，能更好地覆盖加载延迟；
- **A1 和 one-head 整体接近**：A1 胜出点更多，但几何平均速度比约 `0.997×`，说明整体没有压倒性优势；
- **`head_dim=256` 下 one-head 更稳**：A1 多了 `HBM -> SMEM -> register` 中转、`cp.async`
  `commit/wait`、`__syncwarp`、地址计算和更高 SMEM 占用。高维度下每个 head 的计算/搬运更重，
  这些额外成本不一定能被流水隐藏掉；
- **A0 不占优的核心原因不是 HBM 合并访问或 SMEM bank conflict 没做好**，而是 QKV 没有跨线程复用。
  QKV 先进 SMEM 的收益只来自异步流水；如果隐藏掉的 HBM 延迟小于 SMEM 读回和控制逻辑成本，就会亏。

> 注意：当前仓库里的 one-head 结果时间戳是 `2026-06-28`，NHeads A0 是 `2026-07-07`，
> NHeads A1 是 `2026-07-08`。严格横向比较时，建议同一构建、同一机器状态下连续重跑三组 benchmark。

折线图命令：

```bash
# one-head op vs eager
./src/scripts/fused_ROPE_RMSNorm/run_plot_op_benchmark_one_head_vs_eager.sh
# → plot_output/one_head_vs_eager_{gflops,time,speedup}.png

# one-head op vs NHeads A0/A1
./src/scripts/fused_ROPE_RMSNorm/run_plot_op_benchmark_one_head_vs_NHeads.sh
./src/scripts/fused_ROPE_RMSNorm/run_plot_op_benchmark_one_head_vs_NHeads.sh --nheads-algorithm 1
# → plot_output/one_head_vs_NHeads_algorithm{0|1}_{gflops,time,speedup}.png

# NHeads A0 vs NHeads A1
./src/scripts/fused_ROPE_RMSNorm/run_plot_op_benchmark_NHeads_algorithm_compare.sh
# → plot_output/NHeads_algorithm_compare_{gflops,time,speedup}.png
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
# 若要给 one-head vs NHeads 对比图使用，需要切换 is_MULTI_HEAD_PER_WARP/ALGORITHM 并重编译 .so，
# 再分别把运行结果写入固定文件名：
./src/scripts/fused_ROPE_RMSNorm/run_benchmark_op.sh --result-file benchmark_results/ROPE_python_op_benchmark_result_one_head.txt
./src/scripts/fused_ROPE_RMSNorm/run_benchmark_op.sh --result-file benchmark_results/ROPE_python_op_benchmark_result_NHeads_algorithm0.txt
./src/scripts/fused_ROPE_RMSNorm/run_benchmark_op.sh --result-file benchmark_results/ROPE_python_op_benchmark_result_NHeads_algorithm1.txt

# 3) 画图：one-head op vs eager → plot_output/one_head_vs_eager_{metric}.png
./src/scripts/fused_ROPE_RMSNorm/run_plot_op_benchmark_one_head_vs_eager.sh
./src/scripts/fused_ROPE_RMSNorm/run_plot_op_benchmark_one_head_vs_eager.sh --metric speedup

# 4) 画图：one-head op vs NHeads A0/A1
#    默认读取 benchmark_results/ROPE_python_op_benchmark_result_one_head.txt
#    和 benchmark_results/ROPE_python_op_benchmark_result_NHeads_algorithm0.txt。
./src/scripts/fused_ROPE_RMSNorm/run_plot_op_benchmark_one_head_vs_NHeads.sh
./src/scripts/fused_ROPE_RMSNorm/run_plot_op_benchmark_one_head_vs_NHeads.sh --nheads-algorithm 1
./src/scripts/fused_ROPE_RMSNorm/run_plot_op_benchmark_one_head_vs_NHeads.sh --metric time
./src/scripts/fused_ROPE_RMSNorm/run_plot_op_benchmark_one_head_vs_NHeads.sh --metric speedup

# 5) 画图：NHeads A0 vs NHeads A1 → plot_output/NHeads_algorithm_compare_{metric}.png
./src/scripts/fused_ROPE_RMSNorm/run_plot_op_benchmark_NHeads_algorithm_compare.sh
./src/scripts/fused_ROPE_RMSNorm/run_plot_op_benchmark_NHeads_algorithm_compare.sh --metric time

# 6) 自动调优 BLOCK_SIZE_X：sed 改宏 → 重编译 .so → benchmark → 取最佳（见下节）
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
