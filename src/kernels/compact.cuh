#pragma once


#ifndef USE_ROCM
#define ROPE_LDG(arg) __ldg(arg)
#else
#define  ROPE_LDG(arg) *(arg)
#endif


