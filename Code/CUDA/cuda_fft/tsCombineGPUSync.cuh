#ifndef TSCOMBINEGPUSYNC_CUH
#define TSCOMBINEGPUSYNC_CUH

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "tsDefinitions.cuh"
#include "tsHelper.cuh"
#include "tsTest.cuh"

__host__ void   tsCombineGPUSync(fftDir dir, cpx **dev_in, cpx **dev_out, cInt n);
__host__ void   tsCombineGPUSync2D(fftDir dir, cpx *dev_in, cpx *dev_out, cInt n);

__host__ int    tsCombineGPUSync_Validate(cInt n);
__host__ double tsCombineGPUSync_Performance(cInt n);
__host__ int    tsCombineGPUSync2D_Test(cInt n);

#endif