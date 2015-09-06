#ifndef FFT_CONST_GEOM_CU
#define FFT_CONST_GEOM_CU

#include "cuda_runtime.h"
#include "definitions.cuh"

__host__ void fft_const_geom(const float dir, cpx *in, cpx *out, cpx *W, unsigned int *buf, const int n);

#endif