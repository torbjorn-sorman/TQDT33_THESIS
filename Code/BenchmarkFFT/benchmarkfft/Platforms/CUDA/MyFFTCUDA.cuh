#ifndef MYFFTCUDA_CUH
#define MYFFTCUDA_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../../Definitions.h"
#include "MyHelperCUDA.cuh"
#include "../../Common/mymath.h"

__host__ void tsCombine(fftDir dir, cpx **dev_in, cpx **dev_out, int n);
__host__ void tsCombine2D(fftDir dir, cpx **dev_in, cpx **dev_out, int n);

__host__ int tsCombine_Validate(int n);
__host__ int tsCombine2D_Validate(int n);

__host__ double tsCombine_Performance(int n);
__host__ double tsCombine2D_Performance(int n);

#endif