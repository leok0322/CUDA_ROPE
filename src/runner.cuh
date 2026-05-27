#pragma once


void run_ROPE_kernel_base(uint totalRow, uint totalCol, float* A, float* out);

void run_ROPE_kernel_naive(uint totalRow, uint totalCol, float* A, float* out);