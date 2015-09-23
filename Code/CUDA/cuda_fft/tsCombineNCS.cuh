#ifndef TSCOMBINENCS_CUH
#define TSCOMBINENCS_CUH

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "tsDefinitions.cuh"
#include "tsHelper.cuh"
#include "tsTest.cuh"

__host__ void tsCombineNCS(fftDirection dir, cpx **dev_in, cpx **dev_out, const int n);

__host__ int tsCombineNCS_Validate(const int n);
__host__ double tsCombineNCS_Performance(const int n);

#endif