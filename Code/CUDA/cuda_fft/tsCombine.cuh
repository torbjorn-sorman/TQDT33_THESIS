#ifndef TSCOMBINE_CUH
#define TSCOMBINE_CUH

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "tsDefinitions.cuh"
#include "tsHelper.cuh"
#include "tsTest.cuh"

__host__ void tsCombine(fftDir dir, cpx **dev_in, cpx **dev_out, int n);
__host__ void tsCombine2D(fftDir dir, cpx **dev_in, cpx **dev_out, int n);
__host__ void tsCombine2DSurf(fftDir dir, cuSurf *surfaceIn, cuSurf *surfaceOut, int n);

__host__ int tsCombine_Validate(int n);
__host__ int tsCombine2D_Validate(int n);
__host__ int tsCombine2DSurf_Validate(int n);
__host__ double tsCombine_Performance(int n);
__host__ double tsCombine2D_Performance(int n);
__host__ double tsCombine2DSurf_Performance(int n);

#endif