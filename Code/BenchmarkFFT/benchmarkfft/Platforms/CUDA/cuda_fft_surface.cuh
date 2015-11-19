#pragma once
#ifndef CUDAFFTSURFACE_CUH
#define CUDAFFTSURFACE_CUH

#ifdef _NVIDIA
#include <cuda_runtime.h>

#include "../../Definitions.h"
#include "../../Common/mathutil.h"
#include "../../Common/mytimer.h"
#include "../../Common/mycomplex.h"
#include "../C/c_helper.h"
#include "cuda_helper.cuh"

__host__ void cuda_surface_fft(transform_direction dir, cudaSurfaceObject_t *surface_in, cudaSurfaceObject_t *surface_out, int n);
__host__ int cuda_surface_validate(int n);
__host__ double cuda_surface_performance(int n);
#endif
#endif