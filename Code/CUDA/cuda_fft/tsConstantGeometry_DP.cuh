#ifndef TSCONSTANTGEOMETRY_DP_CUH
#define TSCONSTANTGEOMETRY_DP_CUH

#include "cuda_runtime.h"
#include "tsDefinitions.cuh"

__host__ void tsConstantGeometry_DP(fftDir dir, cpx **dev_in, cpx **dev_out, int n);

__host__ int tsConstantGeometry_DP_Validate(int n);
__host__ double tsConstantGeometry_DP_Performance(int n);

#endif