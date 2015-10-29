#pragma once
#ifndef MYFFTCUDA_CUH
#define MYFFTCUDA_CUH

#include "../../Definitions.h"
#include "../../Common/mymath.h"
#include "../../Common/mytimer.h"
#include "../../Common/mycomplex.h"
#include "../C/c_helper.h"
#include "cuda_helper.cuh"

__host__ void cuda_fft(transform_direction dir, cpx *dev_in, cpx *dev_out, int n);
__host__ void cuda_fft_2d(transform_direction dir, cpx *dev_in, cpx *dev_out, int n);

__host__ int cuda_validate(int n);
__host__ int cuda_2d_validate(int n, bool write_img);

__host__ double cuda_performance(int n);
__host__ double cuda_2d_performance(int n);

#endif