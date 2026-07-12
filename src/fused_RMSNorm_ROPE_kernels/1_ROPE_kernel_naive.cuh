#pragma once

#include <cuda_runtime.h>
#include "common.cuh"

template<typename scalar_t, typename index_t>
__global__ void ROPE_kernel_naive(index_t totalRow, index_t totalCol,scalar_t *A, scalar_t *out) {
  // 该线程负责的维度对
  index_t colGroup { threadIdx.x };
  // 该线程负责的行
  index_t row {blockIdx.x };

  if (row >= totalRow) return;
  // 该线程负责旋转的元素
  scalar_t element1 { A[row * totalCol + colGroup * 2] };
  scalar_t element2 { A[row * totalCol + colGroup * 2 + 1] };


  float theta_i { 1.0f / powf(rope_theta, 2 * colGroup / static_cast<float>(totalCol) )  };
  scalar_t newElement1 { static_cast<scalar_t>(cosf(row * theta_i) * element1 - sinf(row * theta_i) * element2) };
  scalar_t newElement2 { static_cast<scalar_t>(sinf(row * theta_i) * element1 + cosf(row * theta_i) * element2 ) };

  out[row * totalCol + colGroup * 2] = newElement1;
  out[row * totalCol + colGroup * 2 + 1] = newElement2;
}