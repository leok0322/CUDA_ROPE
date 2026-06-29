#pragma once

#include <cuda_runtime.h>
#include "common.cuh"
#include "typeConvert.cuh"


template<typename scalar_t_in,typename scalar_t_cache,uint head_dim>
__global__ void fused_QKNorm_and_ROPE_kernel(void* qkv_void, const void* cos_void, const void* sin_void,
  const void* q_weight_void, const void* k_weight_void,const int64_t num_heads_q,
  const int64_t num_heads_k,const int64_t num_heads_v,
  const int64_t rotary_dim, const int64_t num_tokens, const double eps) {
  // 先转换类型
  static_assert(_typeConvert<scalar_t_in>::exists,"不支持该类型");
  using T_in = typename _typeConvert<scalar_t_in>::hinType;
  using T_in2 = typename _typeConvert<scalar_t_in>::hinType2;

  // 输入张量形状（均行优先连续；total = num_heads_q + num_heads_k + num_heads_v）：
  auto* qkv = static_cast<T_in* >(qkv_void);            // [num_tokens, total*head_dim]
  //   每 token 内 [Q 头…|K 头…|V 头…]，
  //   元素地址 = token*total*head_dim + head*head_dim + d
  auto* q_weight = static_cast<const T_in* >(q_weight_void);  // [head_dim]  Q 的 RMSNorm γ
  auto* k_weight = static_cast<const T_in* >(k_weight_void);  // [head_dim]  K 的 RMSNorm γ


  // 在run_kernel.cu文件中，通过static_assert确保了blockDim.x是32的倍数
  uint warpNumPerBlock {blockDim.x / 32};
  uint warpIdx {threadIdx.x / 32};
  uint laneIdx {threadIdx.x % 32};

  uint globalWarpIdx {blockIdx.x * warpNumPerBlock + warpIdx};
  auto totalQKHeadsPerToken {static_cast<uint>(num_heads_q + num_heads_k)};


  uint tokenIdx {globalWarpIdx / totalQKHeadsPerToken};
  uint localHeadIdx {globalWarpIdx % totalQKHeadsPerToken};

  if (tokenIdx >= num_tokens) return;

  const bool isQ = localHeadIdx < num_heads_q;
  uint headIdx {isQ ? localHeadIdx:localHeadIdx - static_cast<uint>(num_heads_q)};
  uint num_heads {static_cast<uint>(num_heads_q + num_heads_k + num_heads_v)};

  // 在run_kernel.cu文件中，通过static_assert确保了head_dim是32的倍数
  constexpr uint elementNumPerThread {head_dim / 32};
  // sizeof(float)的类型是unsigned long
  constexpr uint elementBytesPerThread {elementNumPerThread * static_cast<uint>(sizeof(scalar_t_in))};
  // 搬运时统一按照4字节搬运，并使用向量化一次性搬运
  static_assert(elementBytesPerThread % 4 == 0);
  constexpr uint numPerFourBytesPerThread {elementBytesPerThread / 4};
  // 【合法性由 launch_fused_QKNorm_and_ROPE_kernel 的 if constexpr(numPerFourBytes<=4) 守卫】
  // (方案B；dispatch 宏对所有 dtype 放行 {64,128,256}，非法组合在模板里被丢弃、不实例化本 kernel)：
  // 类型为半精度时，numPerFourBytesPerThread取值是1、2、4，对应的元素个数分别是2、4、8(head_dim 64/128/256)。
  // 类型为float的时候，numPerFourBytesPerThread取值是2、4，对应元素个数是2、4(head_dim 64/128)，元素个数不能是1，因为后续处理，是按照T_in2进行处理的；
  //   float×256 → numPerFourBytesPerThread=8 → packed_as<float,8> 未特化，故被 launch_ 的 if constexpr 挡在外、不实例化到这里。
  // 由此进到本 kernel 的组合，numPerFourBytesPerThread 必 ∈ {1,2,4}，packed_as<float,N> 必有特化。
  // numPerFourBytesPerThread是模板参数，需要是编译期常量，所以head_dim必须是编译期常量，这也是把head_dim作为模板参数的原因
  using T_tran = typename packed_as<float,numPerFourBytesPerThread>::type;

  uint offsetElementPerThread;
  // Token和Head级别的偏移
  if (isQ) offsetElementPerThread  = tokenIdx * num_heads * head_dim + headIdx * head_dim;
  else offsetElementPerThread  = tokenIdx * num_heads * head_dim + static_cast<uint>(num_heads_q) * head_dim + headIdx * head_dim;
  // head_dim级别的偏移
  offsetElementPerThread += laneIdx * elementNumPerThread;

  // HBM load
  // ★唯一的 HBM 加载，且【完全合并访问(fully coalesced)】：
  //   offsetElementPerThread = headBase + laneIdx*elementNumPerThread → 32 个 lane 各读一段连续元素，
  //   首尾平铺成整块连续内存(正好是本 head 的 head_dim 个元素)；每 lane 一次宽向量读(T_tran)，
  //   并集对齐连续 → 硬件合并成最少 cache line 事务、满载零浪费。
  //   合法的 (dtype,head_dim) 组合(由 launch_ 的 if constexpr 守住)下，可一次性将本线程要处理的所有数值加载到寄存器
  auto vecTran { reinterpret_cast<T_tran*>(&qkv[offsetElementPerThread])[0] };
  // 在run_kernel.cu文件中，通过static_assert确保了head_dim是64的倍数
  constexpr uint numPerT_in2PerThread {elementBytesPerThread / sizeof(T_in2)};
  static_assert(numPerT_in2PerThread * 2 == elementNumPerThread);

  float sumOfSquares {0.f};
  // elementNumPerThread是编译期常量，可以用于创建数组
  float elementPerThread[elementNumPerThread] {};

// 以下全程在【寄存器】上处理已加载的 vecTran，不再访问 HBM(合并加载在上面第 77 行已完成)：
//   把宽数据 vecTran 切成 T_in2(2 元素打包)逐块转 float，顺带累加平方和给 RMSNorm。
// 循环次数是编译期常量，所以#pragma unroll可以全部展开

  // 计算RMSNorm
#pragma unroll
  for (uint i = 0; i < numPerT_in2PerThread; i++) {
    auto vecCompute { *(reinterpret_cast<T_in2*>(&vecTran) + i) };
    float2 val {_typeConvert<scalar_t_in>::convert(vecCompute)};
    sumOfSquares += val.x * val.x;
    sumOfSquares += val.y * val.y;
    // 寄存器保存搬运的元素
    elementPerThread[2 * i] = val.x;
    elementPerThread[2 * i + 1] = val.y;
  }

#pragma unroll
  // 所有线程都持有自己warp要计算的head的sumOfSquares
  for (int mask = 16; mask > 0; mask >>= 1) {
    sumOfSquares += __shfl_xor_sync(0xffffffff,sumOfSquares,mask,32);
  }

  float rmsRcp { rsqrtf(sumOfSquares / static_cast<float>(head_dim) + static_cast<float>(eps)) };
  for (uint i = 0; i < elementNumPerThread; i++) {
    // 将scalar_t_in类型对应的cuda类型转换为float
    if (isQ) elementPerThread[i] *=  rmsRcp * _typeConvert<scalar_t_in>::convert(q_weight[laneIdx * elementNumPerThread + i]);
    else elementPerThread[i] *= rmsRcp * _typeConvert<scalar_t_in>::convert(k_weight[laneIdx * elementNumPerThread + i]);
  }

  //计算ROPE
  static_assert(_typeConvert<scalar_t_cache>::exists,"不支持该类型");
  using T_cache = typename _typeConvert<scalar_t_cache>::hinType;
  // using T_cache2 = typename _typeConvert<scalar_t_cache>::hinType2;

  // rotary_dim：每头参与 RoPE 旋转的维度数(<= head_dim)；half = rotary_dim/2 = cos/sin 的列宽。
  //   全旋转时 rotary_dim==head_dim；部分旋转时前 rotary_dim 维旋转、其余 [rotary_dim, head_dim) 透传。
  // 去掉 cast 里的 const：static_cast<const uint>(...) 转到【值类型】产出的是【右值(prvalue)】，
  //   标量右值的顶层 const 无意义(会被忽略)→ nvcc 警告 #191-D "type qualifier is meaningless"。
  //   注意 rotary_dim 是左值，但 rotary_dim/2(算术结果)与 cast 结果都是右值；const 修饰的是这个
  //   临时值，非 rotary_dim。且 auto 本就丢顶层 const，half 无论如何是 uint。
  //   若想让 half 只读，应把 const 放【变量】上(const uint half=...)，而不是放进 cast。
  const uint half = static_cast<uint>(rotary_dim / 2);   // cos_void/sin_void 形状 [num_tokens, half]
  auto* cos = static_cast<const T_cache* >(cos_void);            // [num_tokens, half]  已 gather(方案 A)
  auto* sin = static_cast<const T_cache* >(sin_void);            // [num_tokens, half]

  // 通过约束head_dim，已保证了该条件
  static_assert(elementNumPerThread % 2 == 0,"每个线程处理的元素个数需要是2的倍数");


  // ── cos/sin 不做向量化加载，逐元素处理；原因如下 ──────────────────────────
  //  RoPE 成对旋转：一对 qkv 元素 (x_2i,x_2i+1) 共用一个 (cos_i,sin_i)，故每线程的 cos/sin
  //  数 = elementNumPerThread/2，是 qkv 每线程元素数的【一半】→ 向量化宽度随之减半。
  //  而 elementNumPerThread 最小=2(head_dim=64)，此时 cos/sin 数=1：只有 1 个值，根本【无从
  //  向量化】(单值即标量)；按 4 字节向量化还会踩边界：
  //    · head_dim=64 + 半精度：cosSinBytes = 1*2 = 2，不满足 %4 → 上面 static_assert 直接失败；
  //    · 数=1 时 CosSinTransNum=1，需 packed_as<float,1>(已补特化=float，但仍是“标量”非真向量)。
  //  cos/sin 数据量本就小(qkv 的一半)、且 RoPE 以转换/旋转为主，逐元素加载影响有限。
  //  ⟹ 放弃 cos/sin 的统一向量化，直接按 T_cache 标量逐个读，覆盖全部 head_dim 最稳。

  // constexpr uint cosSinBtyesPerThread {elementNumPerThread / 2 *  static_cast<uint>(sizeof(T_cache))};
  // static_assert(cosSinBtyesPerThread % 4 == 0,"CosSinBtyesPerThread需要是4的倍数");
  // uint CosSinTransNumPerFourBtyesPerThread {cosSinBtyesPerThread / 4};
  // using T_cosSin = packed_as<float,CosSinTransNumPerFourBtyesPerThread>::type;

  // // 合并事务访问
  // // cos 是 const T_cache*（cos/sin 只读）→ &cos[..] 是指向 const 的指针；
  // //   reinterpret_cast 不能去 const(只能加不能减)，故目标类型必须带 const(const T_cosSin*)，
  // //   写成非 const 会报"丢弃 const"错。末尾 [0] 取值拷进 vecCosSin 副本，不影响原 const 数据。
  // auto vecCos = reinterpret_cast<const T_cosSin*>(&cos[offsetCosSinPerThread])[0];
  // auto vecSin = reinterpret_cast<const T_cosSin*>(&sin[offsetCosSinPerThread])[0];

  //cos和sin的偏移
  // ── 只在【lane 级】判一次边界，循环内不再逐对判 ──────────────────────────────
  //  假设：rotary_dim 是 numElemsPerThread(=head_dim/32) 的整数倍 → rotary 边界正好落在 lane
  //        边界，每条 lane 要么【全旋转】要么【全不旋转】，无"半个 lane 跨边界"。
  //        故 if(cosSinInitDimPerThread<half) 一次判整条 lane 即可；循环里 elementNumPerThread/2
  //        对必然全在 rotary 范围内，已删掉逐对 if(...+i<half)，省循环内分支。
  //  该假设由 run_kernel.cu 的 TORCH_CHECK(rotary_dim % numElemsPerThread == 0) 在启动前守住
  //  (rotary_dim 是运行时参数，不能在 device 端 static_assert)；违反则报错而非静默算错。
  //  (对照 vllm：同样是 lane 级 gate if(laneId < rotary_lanes)。)
  uint cosSinInitDimPerThread {elementNumPerThread / 2  * laneIdx};
  if (cosSinInitDimPerThread < half) {
    uint offsetCosSinPerThread {tokenIdx * half + cosSinInitDimPerThread};
#pragma unroll
    for (uint i = 0; i < elementNumPerThread / 2; i++) {
      auto eleCos = _typeConvert<scalar_t_cache>::convert(*(&cos[offsetCosSinPerThread] + i));
      auto eleSin = _typeConvert<scalar_t_cache>::convert(*(&sin[offsetCosSinPerThread] + i));
      float ele1 = elementPerThread[2 * i];
      float ele2 = elementPerThread[2 * i + 1];
      elementPerThread[2 * i] = ele1 * eleCos - ele2 * eleSin;
      elementPerThread[2 * i + 1] = ele1 * eleSin + ele2 * eleCos;
    }
  }


  // ── HBM 写回(就地改写 qkv，与加载第 73 行镜像)──────────────────────────────
  //  把寄存器里的 float 结果逐个转回 T_in，再整体当一个 T_tran 向量化写回原位。
  //  ★必须 alignas(sizeof(T_tran))：局部数组 T_in[N] 默认只 alignof(T_in)=2/4 字节对齐，
  //    而 T_tran(如 float4)需 16 字节对齐；不加 alignas，reinterpret 成 T_tran 读写=未对齐 UB。
  //  字节正好对得上：sizeof(T_tran)==elementBytesPerThread==N*sizeof(T_in)，T_tran 恰好覆盖整段。
  //  (逐元素 convert(float)→T_in 正确；若想省一半转换指令可改成对 convert(float2)→T_in2，等价。)
  //  写回全部 elementNumPerThread 个元素：含部分旋转时未旋转、仅经 RMSNorm 的透传元素。


//   alignas(sizeof(T_tran)) T_in elementPerThreadStore[elementNumPerThread];
// #pragma unroll
//   for (uint i = 0; i < elementNumPerThread; i++) {
//     elementPerThreadStore[i] = _typeConvert<scalar_t_in>::convert(elementPerThread[i]);
//   }
//   auto vec = reinterpret_cast<T_tran*>(&elementPerThreadStore)[0];      // 整段当一个 T_tran
//   reinterpret_cast<T_tran*>(&qkv[offsetElementPerThread])[0] = vec;     // 一次宽向量合并写回

//   alignas(sizeof(T_tran))  T_in2 elementPerThreadStore[numPerT_in2PerThread] {};
// #pragma unroll
//   for (uint i = 0; i < numPerT_in2PerThread; i++) {
//     elementPerThreadStore[i] = _typeConvert<scalar_t_in>::convert(make_float2(elementPerThread[2 * i],elementPerThread[2 * i + 1]));
//   }
//
//   auto vec = reinterpret_cast<T_tran*>(&elementPerThreadStore)[0];      // 整段当一个 T_tran
//   reinterpret_cast<T_tran*>(&qkv[offsetElementPerThread])[0] = vec;     // 一次宽向量合并写回

  T_tran elementT_tranPerThreadStore;
  #pragma unroll
    for (uint i = 0; i < numPerT_in2PerThread; i++) {
      T_in2 elementT_in2PerThreadStore = _typeConvert<scalar_t_in>::convert(make_float2(elementPerThread[2 * i],elementPerThread[2 * i + 1]));
      *(reinterpret_cast<T_in2*>(&elementT_tranPerThreadStore) + i) = elementT_in2PerThreadStore;
    }
  reinterpret_cast<T_tran*>(&qkv[offsetElementPerThread])[0] = elementT_tranPerThreadStore;
}


