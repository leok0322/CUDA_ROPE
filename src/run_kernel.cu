#include <cuda_runtime.h>
#include "error_check.cuh"
#include "kernels.cuh"
#include <torch/extension.h>
#include "dispatch.h"


template<typename qkv_scalar_t,typename cache_scalar_t, uint head_dim>
void launch_fused_QKNorm_and_ROPE_kernel(
    void* qkv_ptr,             // [num_tokens, (Hq+Hk+Hv)*head_dim]  ★就地改写
    const void* q_weight_ptr,       // [head_dim]  Q 的 RMSNorm γ
    const void* k_weight_ptr,       // [head_dim]  K 的 RMSNorm γ
    const void* cos_ptr,            // [num_tokens, rotary_dim/2]  已 gather
    const void* sin_ptr,            // [num_tokens, rotary_dim/2]
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v,
    int64_t num_tokens, int64_t rotary_dim, double eps) {

  int64_t half { rotary_dim / 2};
  // half(=rotary_dim/2)须为 BLOCK_SIZE_X 的整数倍：每个 head 的 half 维按 BLOCK_SIZE_X 分块，
  // 不能整除会导致边界线程越界/漏算。不满足则：先 fprintf 打终端(stderr)日志，
  // 再用 TORCH_CHECK(false, ...) 抛 c10::Error → Python 收到 RuntimeError(取代原 return 的静默)。
  if (half % BLOCK_SIZE_X != 0) {
    fprintf(stderr,
            "[fused_QKNorm_and_ROPE] half(=rotary_dim/2=%lld) 必须是 BLOCK_SIZE_X(=%d) 的整数倍，"
            "当前不满足，跳过 kernel 启动。\n",
            static_cast<long long>(half), BLOCK_SIZE_X);
    TORCH_CHECK(false,
                "[fused_QKNorm_and_ROPE] half(=rotary_dim/2=", half,
                ") 必须是 BLOCK_SIZE_X(=", BLOCK_SIZE_X, ") 的整数倍");
  }

  static_assert(BLOCK_SIZE_X % 32 == 0, "BLOCK_SIZE_X需要是32的倍数，因为一个warp负责一行");
  uint totalQKHeads {static_cast<uint>(num_tokens * (num_heads_q + num_heads_k))};
  uint warpsPerBlock {BLOCK_SIZE_X / 32};
  dim3 gridSize {1,cuda::ceil_div(totalQKHeads,warpsPerBlock),1};
  dim3 blockSize {BLOCK_SIZE_X,1,1};


  fused_QKNorm_and_ROPE_kernel<qkv_scalar_t,cache_scalar_t,head_dim><<<gridSize,blockSize>>>(qkv_ptr,cos_ptr,sin_ptr,q_weight_ptr,k_weight_ptr,
       num_heads_q,num_heads_k,num_heads_v,rotary_dim,num_tokens,eps);
}

void fused_QKNorm_and_ROPE_interleave(
    at::Tensor& qkv,                  // [num_tokens, (Hq+Hk+Hv)*head_dim]  ★就地改写
    const at::Tensor& q_weight,       // [head_dim]  Q 的 RMSNorm γ
    const at::Tensor& k_weight,       // [head_dim]  K 的 RMSNorm γ
    const at::Tensor& cos,            // [num_tokens, rotary_dim/2]  已 gather
    const at::Tensor& sin,            // [num_tokens, rotary_dim/2]
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v,
    int64_t head_dim, int64_t rotary_dim, double eps) {

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
      ROPE_DISPATCH_HEAD_DIM(qkv.scalar_type(),head_dim, "fused_QKNorm_and_ROPE",[&] () -> void {
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



