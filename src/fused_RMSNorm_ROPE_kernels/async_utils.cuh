#pragma once


// 提供 uint32_t / uint64_t / uintptr_t 等固定宽度整数类型。
// 两种常见用法：
//   1. #include <stdint.h>
//      使用 uint32_t / uint64_t / uintptr_t，全局命名空间，C/C++ 都常用。
//   2. #include <cstdint>
//      标准 C++ 推荐使用 std::uint32_t / std::uint64_t / std::uintptr_t。
//      一些编译器也会顺带提供全局 uint32_t，但严格可移植代码不应该依赖这一点。
// 当前代码使用不带 std:: 的全局名字，因此使用 <stdint.h> 更直接。
#include <stdint.h>

#include <cuda_runtime.h>


// 头文件里的 CUDA device helper 需要避免多翻译单元重复定义。
//
// 普通 namespace scope 的 __device__ 函数仍然是外链接函数；如果把
// 非 static / 非 inline / 非模板的 __device__ 函数定义放在 .cuh 里，并被多个
// .cu 翻译单元 include，就可能产生 multiple definition。
//
// __forceinline__ 不是 C++ inline 那种 ODR 豁免；它的作用是要求 nvcc 把这个很小的
// device helper 展开到调用点，通常不再生成独立的 out-of-line __device__ 函数符号。
// 因此这里依赖“强制内联、不发独立符号”来避免头文件多 TU 包含时的重复定义问题。
//
// 更保守的头文件写法也可以是：
//   static __device__ __forceinline__ void helper(...)
// 这样即使某些情况下生成了 out-of-line 符号，也会是当前 TU 私有的内部链接符号。
// [[maybe_unused]] 标在参数声明处：无论哪个 #if 编译分支没有用到该参数，都不报警告。
// (void)param 通常写在某个具体空分支里，只消除那个分支的 unused warning；
// 它只是“空使用”，不会解引用，也不会读写内存。
__device__ __forceinline__ void cp_async_shared_global_ca(
   [[maybe_unused]] const void* glob_ptr,
   [[maybe_unused]] void* smem_ptr,
   [[maybe_unused]] uint size_bytes) {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // cp.async 的 shared 目标操作数不是 CUDA C++ generic pointer；
  // 需要先转成 PTX shared address/offset，常用 32-bit register 传入。
  auto smem_addr {static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr))};

  // global 源地址本身就是 cp.async 需要的 64-bit global pointer。
  // 这里转成 uintptr_t 只是显式保留“地址值”；不解引用、不读取 global memory。
  auto glob_addr {reinterpret_cast<uintptr_t>(glob_ptr)};

  if (size_bytes == 4) {
      asm volatile ("cp.async.ca.shared.global [%0], [%1], 4;\n"
        :
        : "r"(smem_addr), "l"(glob_addr));
  }
  else if (size_bytes == 8) {
    asm volatile ("cp.async.ca.shared.global [%0], [%1], 8;\n"
      :
      :"r"(smem_addr), "l"(glob_addr));
  }
  else if (size_bytes == 16) {
    asm volatile ("cp.async.ca.shared.global [%0], [%1], 16;\n"
      :
      : "r"(smem_addr), "l"(glob_addr));
  }
#elif defined(__CUDA_ARCH__)
  // sm_80 以下没有 cp.async，只能退回普通同步 load/store；
  // 这条路径不能做到 global->shared 异步预取和计算重叠。
  if (size_bytes == 4) {
    *reinterpret_cast<uint32_t*>(smem_ptr) = *reinterpret_cast<const uint32_t*>(glob_ptr);
  }
  else if (size_bytes == 8) {
    *reinterpret_cast<uint64_t*>(smem_ptr) = *reinterpret_cast<const uint64_t*>(glob_ptr);
  }
  else if (size_bytes == 16) {
    *reinterpret_cast<float4*>(smem_ptr) = *reinterpret_cast<const float4*>(glob_ptr);
  }
#else
#endif
}


__device__ __forceinline__ void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n"
    :
    :);
#endif
}


template <uint n>
__device__ __forceinline__ void cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n"
    :
    :"n"(n));
#endif
}
