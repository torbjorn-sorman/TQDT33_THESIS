#ifndef FFT_CONSTANTGEOM_CUH
#define FFT_CONSTANTGEOM_CUH

#include "cuda_runtime.h"
#include "definitions.cuh"

__host__ void FFTConstGeom(fftDirection dir, cpx **dev_in, cpx **dev_out, cpx *dev_W, const int n);

__host__ int FFTConstGeom_Validate(const size_t n);
__host__ double FFTConstGeom_Performance(const size_t n);

#endif