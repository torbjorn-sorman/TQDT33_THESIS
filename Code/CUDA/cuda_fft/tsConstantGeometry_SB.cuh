#ifndef TSCONSTANTGEOMETRY_SB_CUH
#define TSCONSTANTGEOMETRY_SB_CUH

#include "cuda_runtime.h"
#include "tsDefinitions.cuh"

__host__ void tsConstantGeometry_SB(fftDir dir, cpx **dev_in, cpx **dev_out, cInt n);

__host__ int tsConstantGeometry_SB_Validate(cInt n);
__host__ double tsConstantGeometry_SB_Performance(cInt n);

#endif