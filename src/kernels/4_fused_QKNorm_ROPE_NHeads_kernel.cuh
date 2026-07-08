#pragma once

#include <cuda_runtime.h>
#include "../error_check.cuh"
#include "typeConvert.cuh"   // std::is_same_v
#include "async_utils.cuh"
#include <type_traits>  // std::conditional_t


template<typename scalar_t_in,typename scalar_t_cache,uint head_dim,uint headsPerWarp, uint algorithm>
__global__ void fused_QKNorm_ROPE_NHeads_kernel(void* qkv_void, const void* cos_void, const void* sin_void,
  const void* q_weight_void, const void* k_weight_void,const int64_t num_heads_q,
  const int64_t num_heads_k,const int64_t num_heads_v,
  const int64_t rotary_dim, const int64_t num_tokens, const double eps) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && (!defined(USE_ROCM))
  if constexpr (std::is_same_v<scalar_t_in, c10::BFloat16> || std::is_same_v<scalar_t_cache, c10::BFloat16>)  return;
  else {
#endif
    // 线程编号及负责的heads
    // 每一个warp负同一个token的若干个head，该WARP中的线程处理每个Head的若干个元素，同一个block可能会处理不同的token的head
    // 所以加载的时候，每个线程加载对应token的对应若干head的对应若干qkv元素，
    uint warpIdxPerBlock {threadIdx.x / 32};
    uint threadIdxPerWarp {threadIdx.x % 32};
    uint warpNumPerBlock {blockDim.x / 32};
    uint globalWarpIdx {warpNumPerBlock * blockIdx.x + warpIdxPerBlock};
    uint qHeadsPerToken {static_cast<uint>(num_heads_q)};
    uint qkHeadsPerToken {static_cast<uint>(num_heads_q + num_heads_k)};
    uint totalHeads {static_cast<uint>(qkHeadsPerToken * num_tokens)};
    uint totalHeadsPerToken {static_cast<uint>(num_heads_q + num_heads_k + num_heads_v)};

    uint warpNumPerToken {(qkHeadsPerToken + headsPerWarp - 1) / headsPerWarp};
    uint tokenIdx { globalWarpIdx / warpNumPerToken};
    uint warpIdxInToken {globalWarpIdx % warpNumPerToken};

    if (tokenIdx >= num_tokens) return;

    uint fistHeadIdxPerTokenPerThread {warpIdxInToken * headsPerWarp};
    uint headNumInWarp {};
    if (warpIdxInToken == warpNumPerToken - 1) {
      fistHeadIdxPerTokenPerThread + headsPerWarp <= qkHeadsPerToken? headNumInWarp = headsPerWarp:headNumInWarp = qkHeadsPerToken - fistHeadIdxPerTokenPerThread;
    } else {
      headNumInWarp = headsPerWarp;
    }


    // ROPE_DISPATCH_FLOATING_TYPES已经将除了float、c10::Half、c10::BFloat16以外的类型排除了，这里是防御性编程
    static_assert(_typeConvert<scalar_t_in>::exists,"不支持该类型");
    using T_in =  _typeConvert<scalar_t_in>::hinType;
    using T_in2 = _typeConvert<scalar_t_in>::hinType2;
    static_assert(_typeConvert<scalar_t_cache>::exists,"不支持该类型");
    using T_cache = _typeConvert<scalar_t_cache>::hinType;

    // 类型转换
    auto qkv = reinterpret_cast<T_in*>(qkv_void);
    auto q_weight = reinterpret_cast<const T_in*>(q_weight_void);
    auto k_weight = reinterpret_cast<const T_in*>(k_weight_void);
    auto cos = reinterpret_cast<const T_cache*>(cos_void);
    auto sin = reinterpret_cast<const T_cache*>(sin_void);


    static_assert(head_dim % 64 == 0,"每个warp的线程处理至少一对元素，所以head_dim需要是64的倍数");
    constexpr uint elementNumPerHeadPerThread {head_dim / 32};
    constexpr uint elementBytesPerHeadPerThread {elementNumPerHeadPerThread * static_cast<uint>(sizeof(T_in))};
    constexpr uint packedNumIn { elementBytesPerHeadPerThread / 4 };
    // 外层launch函数保证进入本kernel的组合只会让packedNum落在1、2、4之一,
    // 因此packed_as<float, packedNum>一定有特化。
    // T_in是半精度时,head_dim=64/128/256分别得到packedNumIn=1/2/4; elementBytesPerHeadPerThread=4/8/16,elementNumPerHeadPerThread=2/4/8
    // T_in是float时,head_dim=64/128分别得到packedNumIn=2/4,float+256会得到8,
    // 需要由外层launch守卫挡掉,否则packed_as<float,8>没有特化。elementBytesPerHeadPerThread=8/16,,elementNumPerHeadPerThread=2/4
    // T_tran不是转换成T_in2,而是后续把同一段寄存器字节按T_in2重新解释并分块处理;
    // 合法组合下sizeof(T_tran)是sizeof(T_in2)的整数倍。
    // cp.async异步预取本身不依赖T_in_tran;它只需要global地址、shared地址和4B/8B/16B拷贝字节数。
    // T_in_tran主要用于普通向量化load/store,或后续从SMEM读回寄存器时按float/float2/float4宽度读取。
    using T_in_tran = typename packed_as<float,packedNumIn,false>::type;



    // 动态SMEM
    // 一个block只有一块动态shared memory, launch时第三个参数指定的就是这块总大小。
    // 如果写多个extern __shared__声明,它们通常都会从同一个动态SMEM起始地址开始解释,
    // 不是多块独立分配的空间,读写会互相覆盖。需要多段临时空间时,应该只声明这一块
    // 字节数组,再按偏移和对齐手动切分。
    // 这里显式加__align__(16),是为了让dynamic SMEM的base address满足当前最大16B访问的对齐要求:
    //   · cp.async 可能按4B/8B/16B把global数据写入shared,shared目标地址也应满足对应对齐。
    //   · 后续可能把SMEM解释成float/float2/float4等向量类型,其中float4通常要求16B对齐。
    //   · base 16B对齐只能保证起点对齐;后续手动切分offset时仍要让每段起始偏移是对应访问宽度的整数倍。
    extern __shared__ __align__(16) char dynamicSmem[];



    if constexpr (algorithm == 0) {
      // 算法0：
      // cos/sin、q_weight/K_weight分别加载到寄存器
      // 每个线程预取Head0的qkv到SMEM
      // Head0计算RMSnorm、ROPE重叠和Head1加载qkv到SMEM重叠


      // 提前将改线程要用到的cos/sin、 q_weight/k_weight加载到寄存器，每个head复用可以复用
      uint half {static_cast<uint>(rotary_dim / 2)};
      static_assert(elementNumPerHeadPerThread % 2 == 0,"每个warp的线程处处理的元素是2的倍数");


      // cos/sin的加载
      // 如果cos/sin的精度和qkv相同，数量是elementNumPerHeadPerThread的一半，并且packedNumIn是1，不能用packed_as<float,...>::type
      // 如果cos/sin的精度是qkv的一倍，数量是elementNumPerHeadPerThread的一半，刚好能用packed_as<float,...>::type
      // sizeof(qkv_scalar_t) <= sizeof(cache_scalar_t)保证了cos/sin的精度不会是qkv的一半
      constexpr uint cosSinNumPerHeadPerThread {elementNumPerHeadPerThread / 2};
      constexpr uint cosSinBytesPerHeadPerThread {elementNumPerHeadPerThread / 2 * static_cast<uint>(sizeof(T_cache))};

      // 分支外别名可以在分支外使用
      // T_cache\T_in是半精读，并且elementNumPerHeadPerThread是2,packedNumIn是1，elementBytesPerHeadPerThread是4，此时只能用半精读标量
      // 这种分类方法会把全精度标量也归纳进去，
      // constexpr bool is_scalar = cosSinBytesPerHeadPerThread < 4;只会把半精读标量分进去
      constexpr bool is_scalar {sizeof(T_cache) == sizeof(T_in) && cosSinNumPerHeadPerThread == 1};
      static_assert(is_scalar == true ||
                    cosSinBytesPerHeadPerThread == 4 ||
                    cosSinBytesPerHeadPerThread == 8 ||
                    cosSinBytesPerHeadPerThread == 16,
                    "cos/sin向量化加载只支持4B/8B/16B");
      //  这是编译期类型选择，不是运行时代码。
      using T_cache_tran_base = std::conditional_t<is_scalar, T_cache, float>;
      constexpr uint packedNumCache {is_scalar? 1:cosSinBytesPerHeadPerThread / 4};
      using T_cache_tran = typename packed_as<T_cache_tran_base,packedNumCache,is_scalar>::type;

      // elementCos和elementSin的const不可加， const 变量不能后续赋值，而且没有初始化的 const 局部变量本身也不合法
      // const只限制这个临时变量后续不能被重新赋值,不负责保护cos/sin源内存的只读性。
      // 对比下面向量化分支的reinterpret_cast<const T_cache_tran*>:那个const修饰的是源内存指针指向的数据,
      // 必须保留cos/sin的只读属性;否则会把const T_cache*错误地转换成可写指针,丢掉const-correctness。
      T_cache_tran elementCos;
      T_cache_tran elementSin;

      // rotary_dim % numElemsPerThread == 0保证了一个线程负责的若干个元素不会出现一部分需要旋转一部分不需要旋转的情况。
      if (elementNumPerHeadPerThread / 2 * threadIdxPerWarp < half) {

        if constexpr (is_scalar) {
          // 只有一个元素，标量加载，合并事务访问
          elementCos = cos[tokenIdx * half + elementNumPerHeadPerThread / 2 * threadIdxPerWarp];
          elementSin = sin[tokenIdx * half + elementNumPerHeadPerThread / 2 * threadIdxPerWarp];
        } else {
          // 合并事务访问

          // 不在分支外声明T_cache elementCos[elementNumPerHeadPerThread / 2]后再用
          // reinterpret_cast<T_cache_tran*>(&elementCos[0])[0]接收向量化加载结果。
          // 原因:对本地数组取地址并按向量类型写入,会让该数组成为可寻址对象;nvcc通常能优化成寄存器切片,
          // 但不能保证。寄存器压力或别名分析不理想时可能落到local memory,多出ld.local/st.local并降低性能。

          // 完整对齐推导:
          // 目标是证明&cos[offset]/&sin[offset]满足alignof(T_cache_tran)对齐。
          // 令E=elementNumPerHeadPerThread/2,B=sizeof(T_cache),V=E*B=cosSinBytesPerHeadPerThread。
          // 向量化分支中packedNumCache=V/4,T_cache_tran=float/float2/float4,对应V=4/8/16B,
          // 其对齐要求不超过16B。这里的推导依赖host侧已用TORCH_CHECK保证cos/sin base pointer至少16B对齐,
          // 所以base已满足该要求;kernel内只继续证明token/lane偏移不会破坏这个base对齐。
          // 访问offset=tokenIdx*half+E*threadIdxPerWarp,相对base的字节偏移为:
          //   byte_offset=(tokenIdx*half+E*lane)*B
          //              = tokenIdx*half*B + E*lane*B
          // 已有rotary_dim%elementNumPerHeadPerThread==0。因为half=rotary_dim/2,E=elementNumPerHeadPerThread/2,
          // 可得half=k*E,即half%E==0。代入:
          //   tokenIdx*half*B = tokenIdx*k*E*B = tokenIdx*k*V
          //   E*lane*B        = lane*V
          // 因此byte_offset=(tokenIdx*k+lane)*V,是V的整数倍;base已对齐且偏移不破坏对齐。
          elementCos  = reinterpret_cast<const T_cache_tran*>(&cos[tokenIdx * half + elementNumPerHeadPerThread / 2 * threadIdxPerWarp])[0];
          elementSin  = reinterpret_cast<const T_cache_tran*>(&sin[tokenIdx * half + elementNumPerHeadPerThread / 2 * threadIdxPerWarp])[0];
        }

        // 分支内别名不能在分支外使用
        // // 必须用if constexpr而不是普通if:普通if的两个分支在模板实例化时都要通过编译,
        // // 非标量分支可能形成packed_as<float,0>这类非法类型; if constexpr会在编译期丢弃不可达分支。
        // if constexpr (sizeof(T_cache) == sizeof(T_in) && elementBytesPerHeadPerThread == 4) {
        //   const T_cache elementCos = cos[tokenIdx * half + elementNumPerHeadPerThread / 2 * threadIdxPerWarp];
        //   const T_cache elementSin = sin[tokenIdx * half + elementNumPerHeadPerThread / 2 * threadIdxPerWarp];
        // } else {
        //   constexpr uint packedNumCache {cosSinBytesPerHeadPerThread / 4};
        //   static_assert(cosSinBytesPerHeadPerThread == 4 ||
        //                 cosSinBytesPerHeadPerThread == 8 ||
        //                 cosSinBytesPerHeadPerThread == 16,
        //                 "cos/sin向量化加载只支持4B/8B/16B");
        //   using T_cache_tran = typename packed_as<float,packedNumCache>::type;
        //   auto elementCos  = reinterpret_cast<const T_cache_tran*>(&cos[tokenIdx * half + elementNumPerHeadPerThread / 2 * threadIdxPerWarp])[0];
        //   auto elementSin  = reinterpret_cast<const T_cache_tran*>(&sin[tokenIdx * half + elementNumPerHeadPerThread / 2 * threadIdxPerWarp])[0];

        }

      // 加载q_weight和k_weight
      // 合并访问事务
      auto elementQWeight  = reinterpret_cast<const T_in_tran*>(&q_weight[elementNumPerHeadPerThread * threadIdxPerWarp])[0];
      auto elementKWeight  = reinterpret_cast<const T_in_tran*>(&k_weight[elementNumPerHeadPerThread * threadIdxPerWarp])[0];

      // 当前thread，处理qkv的第tokenIdx个token的[fistHeadIdx,fistHeadIdx+headNumInWarp)个head的[threadIdxPerWarp * elementNumPerHeadPerThread,threadIdxPerWarp * elementNumPerHeadPerThread + elementNumPerHeadPerThread)个元素

      // 传统的HBM数据同步
      // auto element_tran = reinterpret_cast<T_tran*>(&qkv[tokenIdx * totalHeadsPerToken * head_dim + (fistHeadIdxPerTokenPerThread + i) * head_dim+ threadIdxPerWarp * elementNumPerHeadPerThread])[0];

      // 对Head0的qkv进行异步预取
      // SMEM指针
      constexpr uint HeadsNumPerWarpSMEM {2};
      T_in* smem_base = reinterpret_cast<T_in*>(dynamicSmem);
      // auto* smem_ptr {static_cast<void *>(&smem_base[warpIdxPerBlock * 32  * HeadsNumPerWarpSMEM * elementNumPerHeadPerThread +
      //   threadIdxPerWarp * HeadsNumPerWarpSMEM * elementNumPerHeadPerThread +
      //   elementNumPerHeadPerThread * 0])};


      // QKV双缓冲只需要current/next两个head slot,所以每个warp的SMEM按2个buffer切分。
      // 下面有两种常见索引layout,单位都是T_in元素:
      //
      // layout A: [warp][lane][buffer][element]
      //   offset =
      //     warpIdxPerBlock * 32 * HeadsNumPerWarpSMEM * elementNumPerHeadPerThread +
      //     threadIdxPerWarp * HeadsNumPerWarpSMEM * elementNumPerHeadPerThread +
      //     bufferIdx * elementNumPerHeadPerThread + elementIdx
      //
      //   对同一个buffer、同一个elementIdx,相邻lane的地址间隔是:
      //     HeadsNumPerWarpSMEM * elementNumPerHeadPerThread * sizeof(T_in)
      //   按32个bank、bank宽度4B估算:
      //     bank_stride_A = HeadsNumPerWarpSMEM * elementNumPerHeadPerThread * sizeof(T_in) / 4
      //   本实现HeadsNumPerWarpSMEM=2,因此bank stride会被双缓冲维度额外放大2倍。
      //   这里E表示每个lane负责的QKV元素个数(elementNumPerHeadPerThread),B表示每个T_in元素字节数。
      //   例如half/bf16时B=2B,float时B=4B; head_dim=64时E=2,head_dim=128时E=4。
      //   例如half/bf16且head_dim=64时,E=2,B=2B,bank_stride_A=2,约2-way conflict;
      //   half/bf16且head_dim=128时,E=4,B=2B,bank_stride_A=4,约4-way conflict。
      //
      // layout B: [warp][buffer][lane][element]  当前采用
      //   offset =
      //     warpIdxPerBlock * HeadsNumPerWarpSMEM * 32 * elementNumPerHeadPerThread +
      //     bufferIdx * 32 * elementNumPerHeadPerThread +
      //     threadIdxPerWarp * elementNumPerHeadPerThread + elementIdx
      //
      //   对同一个buffer、同一个elementIdx,相邻lane的地址间隔是:
      //     elementNumPerHeadPerThread * sizeof(T_in)
      //   bank_stride_B = elementNumPerHeadPerThread * sizeof(T_in) / 4。
      //   相比layout A少了HeadsNumPerWarpSMEM这个倍数,所以在当前双缓冲下bank conflict风险更低。
      //   例如half/bf16且head_dim=64时,E=2,B=2B,bank_stride_B=1,同一elementIdx跨lane读通常不冲突;
      //   half/bf16且head_dim=128时,E=4,B=2B,bank_stride_B=2,约2-way conflict。
      //
      // 注意:cp.async写入shared不是普通st.shared路径,bank conflict主要看后续从SMEM读回QKV计算时的ld.shared模式。
      // 当前QKV后续是每个lane消费自己搬入的连续片段,layout B更符合这个访问模式。
      auto* smem_ptr {static_cast<void *>(&smem_base[warpIdxPerBlock  * 32  * HeadsNumPerWarpSMEM * elementNumPerHeadPerThread +
        0 * 32 * elementNumPerHeadPerThread +
        threadIdxPerWarp * elementNumPerHeadPerThread])};
      // HBM指针
      // 同一个warp预取同一个token的同一个head时,tokenIdx和headIdx都相同,只有threadIdxPerWarp不同。
      // lane访问模式是base + lane * elementNumPerHeadPerThread,每个lane搬连续的elementBytesPerHeadPerThread字节。
      // 因此整个warp覆盖该head上一段连续的QKV地址空间,global侧访问是coalesced-friendly;
      // 硬件会按地址连续性和对齐情况合并成尽量少的HBM memory transactions,不是语义上固定"一次事务"。
      auto* qkv_ptr {static_cast<void *>(&qkv[tokenIdx * totalHeadsPerToken * head_dim +
        fistHeadIdxPerTokenPerThread * head_dim +
        threadIdxPerWarp * elementNumPerHeadPerThread])};

      // cp.async按elementBytesPerHeadPerThread选择4B/8B/16B拷贝宽度,这里不需要T_in_tran参与。
      cp_async_shared_global_ca(qkv_ptr,smem_ptr,elementBytesPerHeadPerThread);
      cp_async_commit_group();

      // headNumInWarp是运行期变量，无法用#pragma unroll展开
      // qkv的预取与计算重叠
      for (uint i = 0; i < headNumInWarp; i++) {


        // 等待预取完成
        cp_async_wait_group<0>();

        // 因为现在1个warp处理多个heads,所以需要在循环里加入的isQ判断
        // fistHeadIdx + i只是当前token内的QK扁平head编号:前num_heads_q个是Q,后num_heads_k个是K。
        // 每次循环处理的head可能从Q跨到K,所以必须用isQ，区分当前head的语义,后续才能选择q_weight/k_weight
        // 并在需要段内headIdx时，对K head减去num_heads_q。
        uint currentHead = fistHeadIdxPerTokenPerThread + i;
        bool currentIsQ = currentHead < static_cast<uint>(num_heads_q);

        // 预取nextHead的qkv
        if (i + 1 < headNumInWarp ) {
          // SMEM指针
          uint nextChannelSMEM {(i + 1) % 2 };
          smem_ptr = static_cast<void *>(&smem_base[warpIdxPerBlock  * 32  * HeadsNumPerWarpSMEM * elementNumPerHeadPerThread +
            nextChannelSMEM * 32 * elementNumPerHeadPerThread +
            threadIdxPerWarp * elementNumPerHeadPerThread]);
          // HBM指针
          qkv_ptr = static_cast<void *>(&qkv[tokenIdx * totalHeadsPerToken * head_dim +
            (fistHeadIdxPerTokenPerThread + i + 1) * head_dim +
            threadIdxPerWarp * elementNumPerHeadPerThread]);
          // cp.async按elementBytesPerHeadPerThread选择4B/8B/16B拷贝宽度,这里不需要T_in_tran参与。
          cp_async_shared_global_ca(qkv_ptr,smem_ptr,elementBytesPerHeadPerThread);
          cp_async_commit_group();
        }

        // 计算RMSNorm
        float sumSquares {0};
        float elementPerHeadPerThread[elementNumPerHeadPerThread];
        uint currentChannelSMEM {i % 2 };
#pragma unroll elementNumPerHeadPerThread
        for (uint j {}; j < elementNumPerHeadPerThread; j++) {
          // 读取smem，不存在bank_conflict
          elementPerHeadPerThread[j] = _typeConvert<scalar_t_in>::convert(smem_base[warpIdxPerBlock  * 32  * HeadsNumPerWarpSMEM * elementNumPerHeadPerThread +
            currentChannelSMEM * 32 * elementNumPerHeadPerThread +
            threadIdxPerWarp * elementNumPerHeadPerThread + j]);
          sumSquares += elementPerHeadPerThread[j] * elementPerHeadPerThread[j];
        }

#pragma unroll
        for (int j {16}; j > 0 ; j>>=1) {
          sumSquares += __shfl_xor_sync(0xffffffff,sumSquares,j,32);
        }

        float sumSquaresRsqurt  = rsqrtf(sumSquares / static_cast<float>(head_dim) + static_cast<float>(eps));

        const auto* elementQWeight_ptr = reinterpret_cast<const T_in*>(&elementQWeight);
        const auto* elementKWeight_ptr = reinterpret_cast<const T_in*>(&elementKWeight);

#pragma unroll elementNumPerHeadPerThread
        for (uint j {}; j < elementNumPerHeadPerThread; j++) {
          // elementQWeight/elementKWeight是前面从global一次性向量化加载到本线程的packed本地值。
          // 这里对&elementQWeight取地址再按T_in*拆包,目的是从packed值中取第j个标量weight。
          // 因为elementNumPerHeadPerThread是constexpr且循环完全展开,j在展开后是编译期常量;
          // 只要这个地址不逃逸到函数外、不用运行期下标访问、寄存器压力不导致spill,nvcc通常会把它优化成
          // 寄存器内的切片/搬移,而不是先写入local memory再ld.local读回。但这不是C++语义保证,
          // 是否真正没有local memory需要检查PTX/SASS或ptxas -v。
          // 当前sm_86编译证据:algorithm0的已实例化组合均显示
          // "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"；
          // register用量约32~56/thread,说明此处取地址拆包在当前代码形态下没有造成local memory spill。
          currentIsQ ? elementPerHeadPerThread[j] *= sumSquaresRsqurt * _typeConvert<scalar_t_in>::convert(elementQWeight_ptr[j])
              : elementPerHeadPerThread[j] *= sumSquaresRsqurt * _typeConvert<scalar_t_in>::convert(elementKWeight_ptr[j]);
        }

        // 计算ROPE
        if (elementNumPerHeadPerThread / 2 * threadIdxPerWarp < half) {
          const auto* elementCos_ptr = reinterpret_cast<const T_cache*>(&elementCos);
          const auto* elementSin_ptr = reinterpret_cast<const T_cache*>(&elementSin);
#pragma unroll elementNumPerHeadPerThread / 2
          for (uint j {}; j < elementNumPerHeadPerThread / 2; j++) {
            float element1 = elementPerHeadPerThread[2 * j];
            float element2 = elementPerHeadPerThread[2 * j + 1];
            float element_cos = _typeConvert<scalar_t_cache>::convert(elementCos_ptr[j]);
            float element_sin = _typeConvert<scalar_t_cache>::convert(elementSin_ptr[j]);

            elementPerHeadPerThread[2 * j] = element_cos * element1 - element_sin* element2;
            elementPerHeadPerThread[2 * j + 1] = element_sin * element1 + element_cos * element2;
          }
        }

        // 写回到HBM
        T_in_tran elementT_tranPerThreadStore;
        for (uint j {}; j < elementNumPerHeadPerThread / 2; j++) {
          T_in2  elementT_in2PerThreadStore = _typeConvert<scalar_t_in>::convert(make_float2(elementPerHeadPerThread[2 * j],elementPerHeadPerThread[2 * j + 1]));
          // reinterpret_cast不能把普通数值直接"转换"成另一个数值类型;这里是先对packed局部变量取地址,
          // 再把该地址重解释为T_in2*，按T_in2粒度填充elementT_tranPerThreadStore内部字节。
          // 当前ptxas -v证据同上:0 spill stores/loads且0 bytes stack frame,说明当前写回组包也没有落到local memory。
          *(reinterpret_cast<T_in2*>(&elementT_tranPerThreadStore) + j) = elementT_in2PerThreadStore;
        }
        // 合并访问事务
        reinterpret_cast<T_in_tran* >(&qkv[tokenIdx * totalHeadsPerToken * head_dim +
        (fistHeadIdxPerTokenPerThread + i) * head_dim +
        threadIdxPerWarp * elementNumPerHeadPerThread])[0] = elementT_tranPerThreadStore;
      }
    }
    else if constexpr (algorithm == 1) {
    }
    else if constexpr (algorithm == 2) {
    }
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && (!defined(USE_ROCM))
  }
#endif
}
