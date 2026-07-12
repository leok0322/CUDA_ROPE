#pragma once

#include "fused_RMSNorm_ROPE_kernels/0_ROPE_kernel_base.cuh"
#include  "fused_RMSNorm_ROPE_kernels/1_ROPE_kernel_naive.cuh"
#include "fused_RMSNorm_ROPE_kernels/2_ROPE_kernel_vectorize.cuh"
#include "fused_RMSNorm_ROPE_kernels/3_fuesd_QKNorm_and_ROPE_kernel.cuh"
#include "fused_RMSNorm_ROPE_kernels/4_fused_QKNorm_ROPE_NHeads_kernel.cuh"
