#ifndef TSCOMBINEGPUSYNCTEX_CUH
#define TSCOMBINEGPUSYNCTEX_CUH

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cuda_texture_types.h>

#include "tsDefinitions.cuh"
#include "tsHelper.cuh"
#include "tsTest.cuh"

__host__ void   tsCombineGPUSyncTex2D(fftDir dir, cuSurf surfIn, cuSurf surfOut, int n);

__host__ int    tsCombineGPUSyncTex2D_Test(int n);

#endif