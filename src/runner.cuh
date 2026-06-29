#pragma once


void run_ROPE_kernel_base(uint totalRow, uint totalCol, float* A, float* out);

void run_ROPE_kernel_naive(uint totalRow, uint totalCol, float* A, float* out);

void run_ROPE_kernel_vectorize(uint totalRow, uint totalCol, float* A, float* out);

void run_fused_QKNorm_and_ROPE_kernel(uint totalRow, uint totalCol, float* A, float* out);