#ifndef FFT_TOBB_CUH
#define FFT_TOBB_CUH

#include "cuda_runtime.h"
#include "definitions.cuh"

__host__ void FFTTobb(fftDirection dir, cpx **dev_in, cpx **dev_out, cpx *dev_W, const int n);

__host__ int FFTTobb_Validate(const size_t n);
__host__ double FFTTobb_Performance(const size_t n);

#endif