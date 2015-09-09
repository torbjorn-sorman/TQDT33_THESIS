#ifndef TSTOBB_SB_CUH
#define TSTOBB_SB_CUH

#include "cuda_runtime.h"
#include "tsDefinitions.cuh"

__host__ void tsTobb_SB(fftDirection dir, cpx **dev_in, cpx **dev_out, const int n);

__host__ int tsTobb_SB_Validate(const int n);
__host__ double tsTobb_SB_Performance(const int n);

#endif