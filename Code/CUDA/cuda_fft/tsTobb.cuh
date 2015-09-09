#ifndef TSTOBB_CUH
#define TSTOBB_CUH

#include "cuda_runtime.h"
#include "tsDefinitions.cuh"

__host__ void tsTobb(fftDirection dir, cpx **dev_in, cpx **dev_out, cpx *dev_W, const int n);

__host__ int tsTobb_Validate(const size_t n);
__host__ double tsTobb_Performance(const size_t n);

#endif