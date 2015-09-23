#ifndef TSCOMBINE_CUH
#define TSCOMBINE_CUH

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "tsDefinitions.cuh"
#include "tsHelper.cuh"
#include "tsTest.cuh"

__host__ void tsCombine(fftDirection dir, cpx **dev_in, cpx **dev_out, const int n);

__host__ int tsCombine_Validate(const int n);
__host__ double tsCombine_Performance(const int n);

#endif