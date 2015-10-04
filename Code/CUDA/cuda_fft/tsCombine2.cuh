#ifndef TSCOMBINE2_CUH
#define TSCOMBINE2_CUH

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "tsDefinitions.cuh"
#include "tsHelper.cuh"
#include "tsTest.cuh"

__host__ void tsCombine2(fftDir dir, cpx **dev_in, cpx **dev_out, int n);

__host__ int tsCombine2_Validate(int n);
__host__ double tsCombine2_Performance(int n);

#endif