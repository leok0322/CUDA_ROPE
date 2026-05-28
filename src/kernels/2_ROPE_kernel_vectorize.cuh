#pragma once

#include <cuda_runtime.h>
#include "common.cuh"

template<typename scalar_t, typename scalar_t4, typename index_t>
__global__ void ROPE_kernel_vectorize(index_t totalRow, index_t totalCol, index_t coupleNum, scalar_t *A, scalar_t *out) {
  // 总的线程号
  index_t threadIdxTotal { blockIdx.x * blockDim.x + threadIdx.x };
  // 该线程负责的行和列组
  index_t colGroup { threadIdxTotal % coupleNum };
  index_t row { threadIdxTotal / coupleNum };
  if (row >= totalRow) return;
  // warp32个线程，每个线程负责2对，4个元素，一个float元素4个字节，共16个字节
  // 32个线程就是128个元素，512个字节，4个cache line
  //  coupleNum ≥ 32，row不变，colGroup改变，所以可以合并访问事务。
  scalar_t4 vecA { reinterpret_cast<scalar_t4 *>(&A[row * totalCol + colGroup * 4])[0] };

  float theta_i1 {1.0f / powf(theta, 2.0f * colGroup * 2 / static_cast<float>(totalCol))};
  float theta_i2 {1.0f / powf(theta, 2.0f * (colGroup * 2 + 1) / static_cast<float>(totalCol))};

  scalar_t4 vecOut { };
  vecOut.x = static_cast<scalar_t>(cosf(row * theta_i1)) * vecA.x - static_cast<scalar_t>(sinf(row * theta_i1)) * vecA.y;
  vecOut.y = static_cast<scalar_t>(sinf(row * theta_i1)) * vecA.x + static_cast<scalar_t>(cosf(row * theta_i1)) * vecA.y;
  vecOut.z = static_cast<scalar_t>(cosf(row * theta_i2)) * vecA.z - static_cast<scalar_t>(sinf(row * theta_i2)) * vecA.w;
  vecOut.w = static_cast<scalar_t>(sinf(row * theta_i2)) * vecA.z + static_cast<scalar_t>(cosf(row * theta_i2)) * vecA.w;
  reinterpret_cast<scalar_t4 *>(&out[row * totalCol + colGroup * 4])[0] = vecOut;
}