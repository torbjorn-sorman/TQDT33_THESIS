#ifndef FFT_CONSTANTGEOM_CUH
#define FFT_CONSTANTGEOM_CUH

#include "cuda_runtime.h"
#include "definitions.cuh"

__host__ void FFTConstGeom(fftDirection dir, cuFloatComplex *dev_in, cuFloatComplex *dev_out, cuFloatComplex *dev_W, unsigned int *buf, const int n);
__host__ void FFTConstGeom2(fftDirection dir, cuFloatComplex *dev_in, cuFloatComplex *dev_out, unsigned int *buf, const int n);

#endif