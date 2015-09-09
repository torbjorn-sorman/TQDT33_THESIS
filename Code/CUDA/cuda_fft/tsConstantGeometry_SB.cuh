#ifndef TSCONSTANTGEOMETRY_SB_CUH
#define TSCONSTANTGEOMETRY_SB_CUH

#include "cuda_runtime.h"
#include "tsDefinitions.cuh"

__host__ void tsConstantGeometry_SB(fftDirection dir, cpx **dev_in, cpx **dev_out, const int n);

__host__ int tsConstantGeometry_SB_Validate(const int n);
__host__ double tsConstantGeometry_SB_Performance(const int n);

#endif