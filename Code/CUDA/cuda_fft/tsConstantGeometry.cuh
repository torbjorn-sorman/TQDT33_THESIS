#ifndef TSCONSTANTGEOMETRY_CUH
#define TSCONSTANTGEOMETRY_CUH

#include "cuda_runtime.h"
#include "tsDefinitions.cuh"

__host__ void tsConstantGeometry(fftDir dir, cpx **dev_in, cpx **dev_out, cpx *dev_W, cInt n);

__host__ int tsConstantGeometry_Validate(cInt n);
__host__ double tsConstantGeometry_Performance(cInt n);

#endif