#ifndef FFT_CONST_GEO_CU
#define FFT_CONST_GEO_CU

#include "cuda_runtime.h"
#include "definitions.cuh"

__host__ void fft_const_geo(const float dir, cuComplex *in, cuComplex *out, cuComplex *W, unsigned int *buf, const int n);

#endif