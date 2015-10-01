#ifndef TSTOBB_CUH
#define TSTOBB_CUH

#include "cuda_runtime.h"
#include "tsDefinitions.cuh"

__host__ void tsTobb(fftDir dir, cpx **dev_in, cpx **dev_out, cpx *dev_W, int n);

__host__ int tsTobb_Validate(int n);
__host__ double tsTobb_Performance(int n);

#endif