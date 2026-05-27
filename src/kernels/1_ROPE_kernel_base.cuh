#pragma once

#include <cuda_runtime.h>
#include "common.cuh"

template<typename scalar_t, typename index_t>
__global__ void ROPE_kernel_base(index_t totalRow, index_t totalCol,index_t coupleNum, scalar_t* A,scalar_t* out) {
  // 该线程负责处理的维度对
  index_t coupleIdx {blockIdx.x * blockDim.x + threadIdx.x};
  // 该该线程负责处理的维度对所在A的行和列组
  index_t row {coupleIdx / coupleNum};
  // 越界保护：runner.cu 曾错误地将 gridSizeX 设为 totalRow * coupleNum（工作项数），
  // 而非 ceil(totalRow * coupleNum / BLOCK_SIZE)（所需 block 数）。
  // 导致实际启动线程数 = totalRow * coupleNum * BLOCK_SIZE（多 256 倍），
  // 多余线程算出的 row >= totalRow，访问 A[row * totalCol + ...] 越出显存 buffer，
  // 触发报错：an illegal memory access was encountered。
  // 修复：此处提前 return，使越界线程不执行任何显存访问。
  if (row >= totalRow) return;
  index_t colGroup {coupleIdx % coupleNum};
  // 拿到维度对的元素
  scalar_t element1 {A[row * totalCol + colGroup * 2]};
  scalar_t element2 {A[row * totalCol + colGroup * 2 + 1]};



  // 曾有错误写法：1 / powf(theta, (2 * colGroup) / totalCol)
  //   (2 * colGroup) 和 totalCol 均为 index_t（整数），做整数除法。
  //   colGroup 范围 [0, coupleNum)，故 2 * colGroup < totalCol（= 2 * coupleNum），
  //   整除结果恒为 0 → theta_i = 1/theta^0 = 1.0f，所有维度对频率相同，
  //   多频率设计完全失效。
  //   修复方式等价，任选其一：
  //     写法一（当前）：(2 * colGroup) / (2.0f * coupleNum)   ← 除数含 2.0f，触发浮点提升
  //     写法二：        (2 * colGroup) / static_cast<float>(totalCol)  ← 显式 cast，语义更清晰
  //     写法三：        static_cast<float>(2 * colGroup) / totalCol    ← 分子 cast，同样触发浮点除法
  //   三种写法中只要除法任意一侧是 float，C++ 就将另一侧提升为 float，整数截断消失。
  float theta_i {1.f / powf(theta,(2 * colGroup) / (2.0f * coupleNum)) };


  scalar_t newElement1 {static_cast<scalar_t>(cosf(row * theta_i)) * element1 - static_cast<scalar_t>(sinf(row * theta_i)) *  element2};
  scalar_t newElement2 {static_cast<scalar_t>(sinf(row * theta_i)) * element1 + static_cast<scalar_t>(cosf(row * theta_i)) *  element2};

  out[row * totalCol + colGroup * 2] =   newElement1;
  out[row * totalCol + colGroup * 2 + 1] = newElement2;
}