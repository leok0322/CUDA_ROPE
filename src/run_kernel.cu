#include <cuda_runtime.h>
#include "error_check.cuh"
#include "kernels.cuh"
#include <torch/extension.h>
#include "dispatch.h"
#include "validation.h"


template<typename qkv_scalar_t,typename cache_scalar_t, uint head_dim>
void launch_fused_QKNorm_and_ROPE_kernel(
    void* qkv_ptr,             // [num_tokens, (Hq+Hk+Hv)*head_dim]  ★就地改写
    const void* q_weight_ptr,       // [head_dim]  Q 的 RMSNorm γ
    const void* k_weight_ptr,       // [head_dim]  K 的 RMSNorm γ
    const void* cos_ptr,            // [num_tokens, rotary_dim/2]  已 gather
    const void* sin_ptr,            // [num_tokens, rotary_dim/2]
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v,
    int64_t num_tokens, int64_t rotary_dim, double eps) {

  // ───────────────────────────────────────────────────────────────────────────
  // ★(dtype, head_dim) 合法性守卫（方案B：必须放在【模板函数】里用 if constexpr）。
  //   kernel 每线程做向量化 load/store，取 numPerFourBytes 个 4 字节，对应 packed_as<float,N>：
  //       numPerFourBytes = (head_dim/32) * sizeof(qkv_scalar_t) / 4
  //   packed_as 只特化 N∈{1,2,4}(float/float2/float4，最大 16 字节)，没有 float8(32 字节向量)。
  //   合法：half/bf16 的 64/128/256(N=1/2/4)、float 的 64/128(N=2/4)；非法：float×256 → N=8。
  //
  //   守卫为何放这里：dispatch 仍会写出并实例化 launch_...<float,256> 本身；但本函数【是模板】，
  //   实例化它时下面的 if constexpr 条件依赖模板参数(head_dim/qkv_scalar_t)，N=8 时含 kernel 的
  //   真支【整段丢弃、不实例化】→ fused_QKNorm_and_ROPE_kernel<float,...,256> 从不实例化 →
  //   不碰 packed_as<float,8>。运行期真撞上非法组合则走 else 抛 TORCH_CHECK(false)。
  //   （若放进 dispatch 宏(非模板上下文)则丢弃支仍 fully checked、仍实例化，挡不住——见
  //     docs/c++/switch_case_fallthrough_and_warnings.txt 第九节。）
  // ───────────────────────────────────────────────────────────────────────────
  constexpr uint numElemsPerThread {head_dim / 32};
  constexpr uint numPerFourBytes {numElemsPerThread * static_cast<uint>(sizeof(qkv_scalar_t)) / 4};

  if constexpr (numPerFourBytes >= 1 && numPerFourBytes <= 4) {
    // ─────────────────────────────────────────────────────────────────────────
    // 本 kernel【真正需要】的约束(都已在别处显式守住，此处汇总)：
    //   ① head_dim % 64 == 0          —— static_assert(下方)。一个 head 由 32 lane 处理、每 lane
    //                                    一次处理 2 个元素(成对 T_in2)，故 head_dim/32 须为偶 → %64。
    //   ② BLOCK_SIZE_X % 32 == 0      —— static_assert(下方)。并行模型是"一个 warp(32 lane)负责一个
    //                                    (token,head)"，blockDim.x 须为整 warp。
    //   ③ rotary_dim % (head_dim/32)==0 —— TORCH_CHECK(下方)。RoPE 用 lane 级边界判定，rotary 边界
    //                                    须落在 lane 边界(每条 lane 要么全旋转要么全不转)。
    //   ④ (dtype,head_dim) 使 numPerFourBytes∈{1,2,4} —— 外层 if constexpr。packed_as<float,N> 须有特化。
    //
    // 注：曾有 `half(=rotary_dim/2) % BLOCK_SIZE_X == 0` 的检查，已删除——它是旧的"整块线程铺 half 维"
    //     设计的遗留，与当前 warp-per-head 模型无关(BLOCK_SIZE_X 只决定每块几个 warp、用于算 grid，
    //     和每个 head 内部的 half 维无关)。该检查会误杀 head_dim=64(half=32%64≠0)等合法配置。
    // ─────────────────────────────────────────────────────────────────────────
    static_assert(BLOCK_SIZE_X % 32 == 0, "BLOCK_SIZE_X需要是32的倍数，因为一个warp负责一行");
    uint totalQKHeads {static_cast<uint>(num_tokens * (num_heads_q + num_heads_k))};
    uint warpsPerBlock {BLOCK_SIZE_X / 32};
    dim3 gridSize {1,cuda::ceil_div(totalQKHeads,warpsPerBlock),1};
    dim3 blockSize {BLOCK_SIZE_X,1,1};


    static_assert(head_dim % 64 == 0,"一个head被32个线程处理，每个线程一次处理2个元素，需要dim是64的倍数");

    // RoPE 旋转用【lane 级】边界判定(kernel 里 if(cosSinInitDimPerThread<half) 一次性判整条 lane)，
    // 故要求 rotary 边界【正好落在 lane 边界】：rotary_dim 必须是 numElemsPerThread(=head_dim/32) 的
    // 整数倍 ⟹ 每条 lane 要么全旋转、要么全不旋转，无需在循环内逐对判。否则(边界落 lane 中间)会算错。
    //   等价：half(=rotary_dim/2) 须为 (head_dim/32)/2 的整数倍。
    // 实践：全旋转(rotary_dim==head_dim)恒满足；head_dim=64 因 rotary_dim 必偶也恒满足；
    //       整齐的部分旋转(rotary_dim 为 8/16 倍数)通常满足。此处显式守住，违反则报错而非静默算错。
    TORCH_CHECK(rotary_dim % numElemsPerThread == 0,
                "[fused_QKNorm_and_ROPE] rotary_dim(=", rotary_dim, ") 必须是 head_dim/32(=",
                numElemsPerThread, ") 的整数倍(RoPE lane 级边界判定要求 rotary 边界对齐 lane 边界)");

    fused_QKNorm_and_ROPE_kernel<qkv_scalar_t,cache_scalar_t,head_dim><<<gridSize,blockSize>>>(qkv_ptr,cos_ptr,sin_ptr,q_weight_ptr,k_weight_ptr,
         num_heads_q,num_heads_k,num_heads_v,rotary_dim,num_tokens,eps);
    cudaCheck(cudaGetLastError());
  } else {
    // 非法 (dtype, head_dim)：如 float×256(每线程 32 字节，需 packed_as<float,8>，无此向量类型)。
    // 因在模板内，本支整段【不实例化】kernel；仅在运行期真撞上时抛错。
    TORCH_CHECK(false,
                "[fused_QKNorm_and_ROPE] head_dim=", head_dim,
                " 不支持当前 dtype：每线程需 ", numPerFourBytes,
                " 个 4 字节(packed_as<float,", numPerFourBytes,
                ">)，仅支持 1/2/4(float/float2/float4)。例如 float×256 非法，请用 half/bf16。");
  }
}

void fused_QKNorm_and_ROPE_neox(
    at::Tensor& qkv,                  // [num_tokens, (Hq+Hk+Hv)*head_dim]  ★就地改写
    const at::Tensor& q_weight,       // [head_dim]  Q 的 RMSNorm γ
    const at::Tensor& k_weight,       // [head_dim]  K 的 RMSNorm γ
    const at::Tensor& cos,            // [num_tokens, rotary_dim/2]  已 gather
    const at::Tensor& sin,            // [num_tokens, rotary_dim/2]
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v,
    int64_t head_dim, int64_t rotary_dim, double eps) {


  ROPE_CHECK(qkv)
  ROPE_CHECK(q_weight)
  ROPE_CHECK(k_weight)
  ROPE_CHECK(cos)
  ROPE_CHECK(sin)
  ROPE_ST_TORCH_CHECK(qkv.dim()==2,"QKV tensor must be 2D: [num_tokens, "
                  "(num_heads_q+num_heads_k+num_heads_v)*head_dim]");
  ROPE_ST_TORCH_CHECK(q_weight.dim() == 1, "Query weights must be 1D: [head_dim]");
  ROPE_ST_TORCH_CHECK(k_weight.dim() == 1, "Key weights must be 1D: [head_dim]");
  ROPE_ST_TORCH_CHECK(cos.dim() == 2,
                  "Cos/sin cache must be 2D: [num_tokens, rotary_dim/2] ]");
  ROPE_ST_TORCH_CHECK(sin.dim() == 2,
                  "Cos/sin cache must be 2D: [num_tokens, rotary_dim/2] ]");
  ROPE_ST_TORCH_CHECK(q_weight.size(0) == head_dim,
                  "Query weights size must match head dimension");
  ROPE_ST_TORCH_CHECK(k_weight.size(0) == head_dim,
                  "Key weights size must match head dimension");

  ROPE_ST_TORCH_CHECK(cos.size(1) % 2 == 0, "rotary_dim must be even");
  ROPE_ST_TORCH_CHECK(sin.size(1) % 2 == 0, "rotary_dim must be even");
  ROPE_ST_TORCH_CHECK(cos.size(1) <= head_dim,
                  "rotary_dim must be less than or equal to head_dim");
  ROPE_ST_TORCH_CHECK(sin.size(1) <= head_dim,
                  "rotary_dim must be less than or equal to head_dim");
  ROPE_ST_TORCH_CHECK(qkv.scalar_type() == q_weight.scalar_type() &&
                      qkv.scalar_type() == k_weight.scalar_type(),
                  "qkv, q_weight and k_weight must have the same dtype");

  int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
  ROPE_ST_TORCH_CHECK(
      qkv.size(1) == total_heads * head_dim,
      "QKV tensor size must match total number of heads and head dimension");

  int64_t num_tokens { qkv.size(0) };
  auto qkv_ptr { qkv.data_ptr() };
  auto q_weight_ptr { q_weight.data_ptr() };
  auto k_weight_ptr { k_weight.data_ptr() };
  auto cos_ptr { cos.data_ptr() };
  auto sin_ptr { sin.data_ptr() };

  // qkv.scalar_type() 返回的是【枚举类 torch::headeronly::ScalarType 的一个枚举值】
  //   (如 ScalarType::Half / ::BFloat16 / ::Float)——是“值”，不是类型，等价于 at::kHalf 等。
  //   ScalarType 是强类型枚举(enum class)，逐一对应 PyTorch 的各 dtype。
  // 这个【运行时枚举值】交给 DISPATCH_FLOATING_TYPES：内部 switch 据它落到某个 case，
  //   再把该 case 的【编译期常量】枚举值经 ScalarTypeToCPPTypeT 映射成类型 scalar_t，
  //   最后调用第三参的 lambda 体(此处为空)。即“运行时 dtype → 编译期类型”的桥。

  //at::Tensor::scalar_type() 和 torch::stable::Tensor::scalar_type() 返回值的类型确实都是 torch::headeronly::ScalarType 枚举(c10::ScalarType / at::ScalarType / torch::headeronly::ScalarType 是同一枚举的别名)
  ROPE_DISPATCH_FLOATING_TYPES(qkv.scalar_type(), "fused_QKNorm_and_ROPE", [&] () -> void {
    using qkv_scalar_t = scalar_t;
    ROPE_DISPATCH_FLOATING_TYPES(cos.scalar_type(), "fused_QKNorm_and_ROPE", [&] () -> void {
      using cache_scalar_t = scalar_t;
      ROPE_DISPATCH_HEAD_DIM(head_dim, "fused_QKNorm_and_ROPE",[&] () -> void {
        launch_fused_QKNorm_and_ROPE_kernel<qkv_scalar_t,cache_scalar_t,headDIM>(qkv_ptr,q_weight_ptr,k_weight_ptr,cos_ptr,sin_ptr,num_heads_q,num_heads_k,num_heads_v,num_tokens,rotary_dim,eps);
        //return;
      });
      //return;   // 不需要：lambda 是 ()->void，走到 } 自动返回 void。这里的 return 只是
                  // “从本 lambda 返回”，与宏内部的 `return __VA_ARGS__();`(从 IIFE 返回、跳出 switch)
                  // 不是一回事——后者才必需，且已由 DISPATCH_FLOATING_TYPES 宏自身提供。
    });
    //return;     // 同上：外层 lambda 末尾也无需 return；整条 dispatch 全程 void，加不加行为一致。
  });

  // ───────────────────────────────────────────────────────────────────────────
  // 上面两层 DISPATCH_FLOATING_TYPES 的【宏展开形式】(去样板，以 outer=Half 分支为例)。
  // 注意每层 dispatch 整体是一个【立即调用的 lambda(IIFE)】：[&]{ switch... }() ；
  //   宏体里的 `return __VA_ARGS__();` 不是从本函数返回，而是【从这个 IIFE 返回】——
  //   __VA_ARGS__() 是“调用用户 lambda”(函数调用)，return 把其结果传出 IIFE 并跳出 switch。
  //   这里用户 lambda 是 ()->void，故返回 void，效果≈“调用后结束该 case”。
  //
  //   [&]{                                                   // 外层 IIFE 开始
  //     const auto& the_type = qkv.scalar_type();
  //     switch (the_type) {
  //       case ScalarType::Float:    { using scalar_t = float;        return (OUTER_LAMBDA)(); }
  //       case ScalarType::Half: {
  //         using scalar_t = c10::Half;                       // ① 外层 scalar_t = qkv 类型
  //         return ( [&]()->void {                            //   return 从外层 IIFE 返回；()调用 OUTER_LAMBDA
  //           using qkv_scalar_t = scalar_t;                  // ② 此刻 scalar_t 仍是①→存住 qkv 类型 ✓
  //           return ( [&]{                                   //   内层 IIFE；return 从内层 IIFE 返回
  //             const auto& the_type = cos.scalar_type();
  //             switch (the_type) {
  //               case ScalarType::Float: {
  //                 using scalar_t = float;                   // ③ 内层 scalar_t = cos 类型【遮蔽①】
  //                 return ( [&]()->void {                    //   ()调用 INNER_LAMBDA
  //                   using cache_scalar_t = scalar_t;        // ④ 此处 scalar_t 是③→cos 类型 ✓
  //                   fused_QKNorm_and_ROPE_kernel<qkv_scalar_t, cache_scalar_t>
  //                       <<<gridSize, blockSize>>>(qkv_ptr, cos_ptr, /*...*/);
  //                 } )();
  //               }
  //               case ScalarType::Half:     { using scalar_t = c10::Half;     return (INNER_LAMBDA)(); }
  //               case ScalarType::BFloat16: { using scalar_t = c10::BFloat16; return (INNER_LAMBDA)(); }
  //               default: /* ST_TORCH_CHECK: cos dtype not implemented */ ;
  //             }
  //           } )();                                          // 内层 IIFE 立即调用
  //         } )();                                            // OUTER_LAMBDA 立即调用
  //       }
  //       case ScalarType::BFloat16: { using scalar_t = c10::BFloat16; return (OUTER_LAMBDA)(); }
  //       default: /* ST_TORCH_CHECK: qkv dtype not implemented */ ;
  //     }
  //   }()                                                     // 外层 IIFE 立即调用
  //
  // 要点：③ 的 using scalar_t 在内层 case 块【遮蔽】了外层①；故必须在进内层【之前】(②)
  //       先 using qkv_scalar_t = scalar_t 抢存 qkv 类型。qkv_scalar_t/cache_scalar_t 是
  //       【类型】，内层 lambda 靠【词法作用域】查找即可见，无需捕获；gridSize/qkv_ptr 等是
  //       【运行时变量】，才需逐层 [&] 捕获。
  // 注：DISPATCH_FLOATING_TYPES 宏体当前有个 `...,` 笔误(dispatch.h)，需去掉才能真正编过。
  // ───────────────────────────────────────────────────────────────────────────

  // AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,  qkv.scalar_type(),"fused_QKNorm_and_ROPE",([&] () -> void {
  //     fused_QKNorm_and_ROPE_kernel<scalar_t><<<gridSize,blockSize>>>(qkv_ptr,cos_ptr,sin_ptr,q_weight_ptr,k_weight_ptr,num_heads_q,num_heads_k,num_heads_v,head_dim,rotary_dim,num_tokens,eps);
  //     cudaCheck(cudaGetLastError());
  // }));
}



