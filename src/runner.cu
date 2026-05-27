#include <assert.h>

#include "error_check.cuh"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cuda/cmath>


void run_ROPE_kernel_base(uint totalRow, uint totalCol, float* A, float* out) {
    // 每一个block只处理一个维度对，进行ROPE旋转
    dim3 block(BLOCK_SIZE, 1,1);
    // 列可以完整覆盖维度对
    assert(totalCol % 2 == 0 && "列需要完整覆盖维度对");
    uint coupleNum {totalCol / 2};
    uint gridSizeX {cuda::ceil_div(totalRow *  coupleNum,BLOCK_SIZE)};
    dim3 grid(gridSizeX, 1, 1);
    ROPE_kernel_base<float, uint><<<grid,block>>>(totalRow,totalCol,coupleNum,A,out);
    cudaCheck(cudaGetLastError());
}