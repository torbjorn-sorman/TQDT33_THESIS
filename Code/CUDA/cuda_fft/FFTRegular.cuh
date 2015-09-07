#ifndef FFT_REGULAR_CUH
#define FFT_REGULAR_CUH

#include "cuda_runtime.h"
#include "definitions.cuh"

__host__ void FFTRegular(fftDirection dir, cuFloatComplex *dev_in, cuFloatComplex *dev_out, cuFloatComplex *dev_W, unsigned int *buf, const int n);

#endif