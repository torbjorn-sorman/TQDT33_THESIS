#ifndef TSREGULAR_CUH
#define TSREGULAR_CUH

#include "cuda_runtime.h"
#include "tsDefinitions.cuh"

__host__ void tsRegular(fftDirection dir, cpx **dev_in, cpx **dev_out, cpx *dev_W, const int n);

__host__ int tsRegular_Validate(const size_t n);
__host__ double tsRegular_Performance(const size_t n);

#endif