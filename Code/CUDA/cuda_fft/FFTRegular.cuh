#ifndef FFT_REGULAR_CUH
#define FFT_REGULAR_CUH

#include "cuda_runtime.h"
#include "definitions.cuh"

__host__ void FFTRegular(fftDirection dir, cpx **dev_in, cpx **dev_out, cpx *dev_W, const int n);
__host__ int FFTRegular_Validate(const size_t n);
__host__ double FFTRegular_Performance(const size_t n);

#endif