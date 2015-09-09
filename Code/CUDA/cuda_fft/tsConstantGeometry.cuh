#ifndef TSCONSTANTGEOMETRY_CUH
#define TSCONSTANTGEOMETRY_CUH

#include "cuda_runtime.h"
#include "tsDefinitions.cuh"

__host__ void tsConstantGeometry(fftDirection dir, cpx **dev_in, cpx **dev_out, cpx *dev_W, const int n);

__host__ int tsConstantGeometry_Validate(const size_t n);
__host__ double tsConstantGeometry_Performance(const size_t n);

#endif