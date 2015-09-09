#ifndef TSCONSTANTGEOMETRY_DP_CUH
#define TSCONSTANTGEOMETRY_DP_CUH

#include "cuda_runtime.h"
#include "tsDefinitions.cuh"

__host__ void tsConstantGeometry_DP(fftDirection dir, cpx **dev_in, cpx **dev_out, const int n);

__host__ int tsConstantGeometry_DP_Validate(const size_t n);
__host__ double tsConstantGeometry_DP_Performance(const size_t n);

#endif