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

__host__ void   tsCombineGPUSyncTex(fftDir dir, cpx **dev_in, cpx **dev_out, cInt n);
__host__ void   tsCombineGPUSyncTex2D(fftDir dir, cudaSurfaceObject_t surface, cpx *dev_in, cpx *dev_out, cInt n);

__host__ int    tsCombineGPUSyncTex_Validate(cInt n);
__host__ double tsCombineGPUSyncTex_Performance(cInt n);
__host__ int    tsCombineGPUSyncTex2D_Test(cInt n);

#endif