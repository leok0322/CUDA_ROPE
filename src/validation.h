#pragma once


#include "error_check.cuh"


#define ROPE_CHECK_CONTIGUOUS(x)             \
  ROPE_ST_TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous")


#define ROPE_CHECK_IN_CUDA(x)                                          \
  ROPE_ST_TORCH_CHECK(x.is_cuda(),#x," must be a CUDA tensor")


# define ROPE_CHECK(x)   \
  ROPE_CHECK_IN_CUDA(x)   \
  ROPE_CHECK_CONTIGUOUS(x)
