#pragma once

// ── 本文件用到的符号 → 各自所属头文件(按“自包含/IWYU”显式引齐) ──────────────
#include <cuda_runtime.h>                      // float2/float4、uint(template<...,uint num>)；CUDA 基础类型
#include <cuda_fp16.h>                          // __half/__half2 + fp16 intrinsics
                                                //   (__half2float/__half22float2/__float2half_rn/__float22half2_rn)
                                                //   ★也提供 __ldg(const __half*) 重载 → ROPE_LDG(cos/sin) 对 fp16 可用
#include <cuda_bf16.h>                          // ★必需：__nv_bfloat16/__nv_bfloat162 + bf16 intrinsics
                                                //   (__bfloat162float/__bfloat1622float2/__float2bfloat16_rn/__float22bfloat162_rn)
                                                //   ★也提供 __ldg(const __nv_bfloat16*) 重载 → ROPE_LDG(cos/sin) 对 bf16 可用
                                                //   (float 的 __ldg 由 cuda_runtime.h 内置；故 cos/sin 三种 T_cache 都有 __ldg)
                                                //   —— 之前漏了它，bf16 特化全靠其它头“传递包含”才编过，脆弱。
#include <torch/headeronly/util/Half.h>         // c10::Half        ← 轻量 headeronly 头
#include <torch/headeronly/util/BFloat16.h>     // c10::BFloat16    ← 取代重型 <torch/extension.h>
// 说明：本文件只用到 torch 的 c10::Half / c10::BFloat16 两个类型，无需 <torch/extension.h>
//   那套巨型伞头(会拖进 pybind11 / 整个 ATen / <Python.h>)，故改用上面两个 headeronly 头。

template<typename scalar_t, uint num>
struct packed_as;

template <>
struct packed_as<float,4> {
  using type = float4;
};

template <>
struct packed_as<float,2> {
  using type = float2;
};

template <>
struct packed_as<float,1> {
  using type = float;
};


// _typeConvert：类型转换 traits，实现【静态多态(编译期多态)】。它把 torch/C++ 标量类型
//   (float / c10::Half / c10::BFloat16) 映射到 CUDA 端使用的类型，并提供值转换。
//   ── 它做【两件事】(别只记成“转类型”) ──
//     (a) 类型映射(编译期，靠模板特化)：成员别名 hinType / hinType2[/hinType4]
//         float→{float,float2,float4}；c10::Half→{__half,__half2}；
//         c10::BFloat16→{__nv_bfloat16,__nv_bfloat162}。
//         真正“换类型”的是 Half/BFloat16(torch 封装类型→CUDA 原生类型，device 端才能算)；
//         float 这条是【恒等】(float 本就是 CUDA 类型)。
//     (b) 值转换(运行时，靠 convert 重载)：把数据在原类型↔float 间互转
//         (__half2float / __float2half_rn …)；float 的 convert 是恒等 return x。
//   ── 机制 ──“类模板显式特化(按 T 选实现集) + 成员函数重载(按参数选具体 convert)”，
//     全部编译期解析，无虚表、无运行时分支、可内联，零开销；对比 virtual 的动态多态。
//     注意：是【类模板特化 + 重载】，不是“模板函数”。dispatch 宏先把运行时 dtype 枚举
//     桥接成编译期类型 scalar_t，再由本 traits 静态选中对应特化。
//   主模板默认 exists=false 当“白名单”守门：未特化的类型一律拦在 static_assert。
template <typename torch_type>
struct _typeConvert {
  // ── 静态成员变量的几种写法（以本例的编译期布尔标志为参照）─────────────────
  // exists 被用在 static_assert(Converter::exists, ...) 里，必须是【编译期常量】，
  // 故只能选下面 ①/②，不能用 ③/④。
  //
  //  ① static constexpr bool exists = false;   ← 本行采用，推荐
  //     · 编译期常量：可用于 static_assert / if constexpr / 模板实参；
  //     · C++17 起【隐式 inline】：类内定义即完整定义，多 TU 包含不冲突，无需类外定义；
  //     · 通常不占运行时存储（值被内联），仅 ODR-use(取地址等)时才在静态区分配。
  //
  //  ② inline static const bool exists = false; (C++17)
  //     · const + 常量初始化器 → 也能用于常量表达式，static_assert 可过；
  //     · inline 负责免类外定义。但表意弱于 constexpr，且需手写 inline，不如①。
  //
  //  ③ inline static bool exists{false};        // ✗ 不可用于本场景
  //     · 非 const/constexpr → 是【运行时可变变量】，不是常量表达式；
  //     · static_assert(Converter::exists) 会编译失败；还白占一份可变存储。
  //     · inline 只解决 ODR(免类外定义)，解决不了“常量性”——两者职责不同。
  //
  //  ④ static bool exists;                       // 普通静态成员：类内声明、类外定义
  //     · 须在某一个 .cpp 里类外定义一次：bool _typeConvert<T>::exists = false;
  //       否则 ODR-use 时链接报错；同样非常量，不能用于 static_assert。
  //
  // 存储/常量性小结：static 成员有静态存储期；但 constexpr 常量常被内联、不占内存。
  //   “类内能初始化”是 constexpr 的功劳；“免类外定义”是 inline(C++17)的功劳，别混。
  static constexpr bool exists {false};
};


template <>
struct _typeConvert<float> {
  static constexpr bool exists {true};
  using hinType = float;
  using hinType2 = float2;
  using hinType4 = float4;
  // ── 关于下面这些成员函数的几点 ──────────────────────────────────────────
  // 1) 类内带函数体 { return x; } 即【定义】：定义已完成，无需类外再定义。
  //    （对比：类内只写声明 `static float convert(hinType);` 才需类外补
  //      `float _typeConvert<float>::convert(...){...}`。）
  // 2) 类内定义的成员函数【隐式 inline】：获 ODR 豁免，故本头文件被多个 .cu
  //    包含也不会重复定义冲突。叠加它又在模板特化里，双重豁免。
  // 3) static 只表示“静态成员函数”：调用不需要对象、【没有 this 指针】，
  //    与 inline / 定义放哪【无关】。
  // 4) __forceinline__ 只管 device 端【强制内联展开】(代码生成)，不提供 ODR
  //    豁免；ODR 豁免来自上面的“类内定义/模板”。两者职责分开。
  //
  // 哪些函数才【隐式 inline】(默认普通函数不是 inline)：
  //   · 类内定义的成员函数（本例）        · constexpr 函数 / constexpr 成员
  //   · consteval 函数 (C++20)            · C++17 起 inline 变量 / static constexpr 成员变量
  // 而“类外定义的成员函数”“普通自由函数”默认非 inline，想放头文件需手写 inline；
  // 函数模板的实例靠模板自身的 ODR 规则豁免，不算“隐式 inline”。
  __device__ __forceinline__ static float convert(hinType x) {
    return x;
  }
  __device__ __forceinline__ static float2 convert(hinType2 x) {
    return x;
  }
  __device__ __forceinline__ static float4 convert(hinType4 x) {
    return x;
  }
};


template <>
struct _typeConvert<c10::Half> {
  static constexpr bool exists {true};
  using hinType = __half;
  using hinType2 = __half2;
  // 为何 convert 用 __half2float / __float2half_rn 等 intrinsic，而不是 static_cast：
  //   __half/__nv_bfloat16 ↔ float 的转换运算符在 host/device、CUDA/HIP 版本、GPU 架构间
  //   【不一致、不可靠】（device-only、bf16 需 SM80+ 等），裸 static_cast 不能稳定使用；
  //   故统一改用官方数值转换 intrinsic（成对版如 __half22float2 还借硬件一次转 2 个）。
  //   注意：reinterpret_cast 更不行——half/bf16/float 位格式与宽度都不同，按位重解释是垃圾值。
  __device__ __forceinline__ static float convert(hinType x) {
    return __half2float(x);
  }
  __device__ __forceinline__ static float2 convert(hinType2 x) {
    return __half22float2(x);
  }
  __device__ __forceinline__ static hinType convert(float x) {
    return __float2half_rn(x);
  }
  __device__ __forceinline__ static hinType2 convert(float2 x) {
    return __float22half2_rn(x);
  }
};


template <>
struct _typeConvert<c10::BFloat16> {
  static constexpr bool exists {true};
  using hinType = __nv_bfloat16;
  using hinType2 = __nv_bfloat162;
  __device__ __forceinline__ static float convert(hinType x) {
    return __bfloat162float(x);
  }
  __device__ __forceinline__ static float2 convert(hinType2 x) {
    return __bfloat1622float2(x);
  }
  __device__ __forceinline__ static hinType convert(float x) {
    return __float2bfloat16_rn(x);
  }
  __device__ __forceinline__ static hinType2 convert(float2 x) {
    return __float22bfloat162_rn(x);
  }

};


// ═══════════════════════════════════════════════════════════════════════════
// 【对照：不用“多个特化”，改用“单个通用模板 + std::conditional_t / if constexpr”】
//
//   功能与上面的多特化【等价】，但通常更丑、更难扩展，仅作对照参考(用 #if 0 关闭，
//   启用会与上面的特化【重定义冲突】，二者不可并存)。
//   · 类型映射：用 std::conditional_t 嵌套挑类型(类型多了就是“嵌套地狱”)。
//   · 值转换  ：用 if constexpr 在一个函数体内按类型分支(反向/打包都得塞进同一体)。
//   · exists  ：用 is_same_v 的“或”列出白名单。
//   缺点：加一种 dtype 要改多处(每个 conditional_t / 每个 if constexpr)，易漏 → 漂移；
//        故主流(本文件)用“一型一特化”的写法，清爽、加类型只加一段。
// ═══════════════════════════════════════════════════════════════════════════
#if 0
#include <type_traits>   // std::conditional_t / std::is_same_v

template <typename T>
struct _typeConvert_generic {
  // ① 白名单：用 is_same 的“或”枚举支持的类型
  static constexpr bool exists =
      std::is_same_v<T, float> ||
      std::is_same_v<T, c10::Half> ||
      std::is_same_v<T, c10::BFloat16>;

  // ② 类型映射：conditional_t 嵌套(自上而下命中第一个为真的分支)
  using hinType =
      std::conditional_t<std::is_same_v<T, c10::Half>,     __half,
      std::conditional_t<std::is_same_v<T, c10::BFloat16>, __nv_bfloat16,
      /* 其余视作 float */                                  float>>;
  using hinType2 =
      std::conditional_t<std::is_same_v<T, c10::Half>,     __half2,
      std::conditional_t<std::is_same_v<T, c10::BFloat16>, __nv_bfloat162,
      /* 其余视作 float */                                  float2>>;

  // ③ 值转换：if constexpr 在【同一个函数体】里按类型分支
  //   原类型 → float / float2（Load 方向）
  __device__ __forceinline__ static float convert(hinType x) {
    if constexpr      (std::is_same_v<T, c10::Half>)     return __half2float(x);
    else if constexpr (std::is_same_v<T, c10::BFloat16>) return __bfloat162float(x);
    else                                                 return x;            // float 恒等
  }
  __device__ __forceinline__ static float2 convert(hinType2 x) {
    if constexpr      (std::is_same_v<T, c10::Half>)     return __half22float2(x);
    else if constexpr (std::is_same_v<T, c10::BFloat16>) return __bfloat1622float2(x);
    else                                                 return x;            // float2 恒等
  }
  //   float / float2 → 原类型（Store 方向）。注意 float 特化下与上面的 convert(float)
  //   形参类型相同会“重定义”，故通用版要按 T 决定是否提供——这恰恰暴露单模板的别扭：
  //   float 时 hinType==float，convert(hinType) 与 convert(float) 撞签名。多特化则天然无此问题。
};
#endif

