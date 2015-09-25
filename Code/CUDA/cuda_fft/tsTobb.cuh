#ifndef TSTOBB_CUH
#define TSTOBB_CUH

#include "cuda_runtime.h"
#include "tsDefinitions.cuh"

__host__ void tsTobb(fftDir dir, cpx **dev_in, cpx **dev_out, cpx *dev_W, cInt n);

__host__ int tsTobb_Validate(cInt n);
__host__ double tsTobb_Performance(cInt n);

#endif