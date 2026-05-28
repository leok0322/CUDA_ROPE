#include <assert.h>        // assert()
#include <stdexcept>       // std::invalid_argument
#include <cstdlib>         // EXIT_FAILURE

#include "error_check.cuh" // cudaCheck()
#include "kernels.cuh"     // ROPE_kernel_base, ROPE_kernel_naive, ROPE_kernel_vectorize, BLOCK_SIZE
#include <cuda_runtime.h>  // dim3, cudaGetLastError
#include <cuda/cmath>      // cuda::ceil_div


void run_ROPE_kernel_base(uint totalRow, uint totalCol, float* A, float* out) {
    // 每一个block只处理BLOCK_SIZE个维度对，进行ROPE旋转
    dim3 block(BLOCK_SIZE, 1,1);
    // 列可以完整覆盖维度对
    // assert在release模式下不可用
    // assert(totalCol % 2 == 0 && "列需要完整覆盖维度对");
    if (totalCol % 2 != 0) {
        fprintf(stderr, "run_ROPE_kernel_vectorize: totalCol=%u, otalCol need to be the cover all the element couple.\n",totalCol);
        exit(EXIT_FAILURE);
    }
    uint coupleNum {totalCol / 2};
    uint gridSizeX {cuda::ceil_div(totalRow *  coupleNum,BLOCK_SIZE)};
    dim3 grid(gridSizeX, 1, 1);
    ROPE_kernel_base<float, uint><<<grid,block>>>(totalRow,totalCol,coupleNum,A,out);
    cudaCheck(cudaGetLastError());
}

void run_ROPE_kernel_naive(uint totalRow, uint totalCol, float* A, float* out) {
    // 列可以完整覆盖维度对
    // assert在release模式下不可用
    // assert(totalCol % 2 == 0 && "列需要完整覆盖维度对");
    if (totalCol % 2 != 0) {
        fprintf(stderr, "run_ROPE_kernel_vectorize: totalCol=%u, otalCol need to be the cover all the element couple.\n",totalCol);
        exit(EXIT_FAILURE);
    }

    uint coupleNum {totalCol / 2};
    // ── 曾遇报错：invalid configuration argument ────────────────────────────
    // 报错位置：cudaCheck(cudaGetLastError()) 处
    // 报错原因：kernel1 的 block 大小 = coupleNum = totalCol/2，随输入动态变化。
    //   GPU 每个 block 最多 1024 线程，当 totalCol=4096 时 coupleNum=2048 > 1024，
    //   dim3 block(2048) 是非法配置，kernel 启动失败。
    //   该限制对 totalCol <= 2048（coupleNum <= 1024）的输入不触发。
    //
    // 第一次修复尝试：加 assert
    //   assert(coupleNum <= 1024) 在 Debug 模式下有效，
    //   但 CMake Release 构建默认定义 NDEBUG，assert 被预处理器删除，
    //   程序继续执行到非法的 dim3 block(2048)，仍然崩溃。
    //
    // 第二次修复尝试：if + exit(EXIT_FAILURE)
    //   能打印报错信息，但 exit() 直接终止整个进程，
    //   validation.cu 的外层 size 循环无法继续，后续 size 全部跳过。
    //
    // 最终修复：if + throw std::invalid_argument
    //   throw 在 Debug/Release 均生效，异常沿调用栈传播至
    //   validation.cu 的 catch 块，由调用方决定是否继续循环，
    //   不终止进程，外层 nsize 循环可正常跳过当前 size 继续运行。
    assert(coupleNum <= 1024 && "一个block的线程不能超过1024个");
    if (coupleNum > 1024) {
        throw std::invalid_argument(
            "run_ROPE_kernel_naive: coupleNum=" + std::to_string(coupleNum) +
            " exceeds max threads per block (1024), totalCol=" + std::to_string(totalCol));
    }
    // 每一个block只处理一行所有的维度对，进行ROPE旋转
    dim3 block(coupleNum, 1,1);
    dim3 grid(totalRow, 1, 1);
    ROPE_kernel_naive<float, uint><<<grid,block>>>(totalRow,totalCol,A,out);
    cudaCheck(cudaGetLastError());
}

void run_ROPE_kernel_vectorize(uint totalRow, uint totalCol, float* A, float* out) {
    // 每一个block只处理BLOCK_SIZE* 2个维度对，进行ROPE旋转
    dim3 block(BLOCK_SIZE, 1,1);
    // 列可以完整覆盖维度对
    // assert在release模式下不可用
    // assert(totalCol % 2 == 0 && "列需要完整覆盖维度对");
    if (totalCol % 4 != 0) {
        fprintf(stderr, "run_ROPE_kernel_vectorize: totalCol=%u, otalCol need to be the cover all the double element couple.\n",totalCol);
        exit(EXIT_FAILURE);
    }
    // 一个线程负责两个维度对
    uint coupleNum {totalCol / 4};
    uint gridSizeX { cuda::ceil_div(totalRow *  coupleNum,BLOCK_SIZE) };
    dim3 grid(gridSizeX, 1, 1);
    ROPE_kernel_vectorize<float, float4, uint><<<grid,block>>>(totalRow,totalCol,coupleNum,A,out);
    cudaCheck(cudaGetLastError());
}