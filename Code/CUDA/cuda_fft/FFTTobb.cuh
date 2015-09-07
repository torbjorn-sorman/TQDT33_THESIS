#ifndef FFT_TOBB_CUH
#define FFT_TOBB_CUH

#include "cuda_runtime.h"
#include "definitions.cuh"

__host__ void FFTTobb(fftDirection dir, cuFloatComplex *dev_in, cuFloatComplex *dev_out, cuFloatComplex *dev_W, unsigned int *buf, const int n);

#endif