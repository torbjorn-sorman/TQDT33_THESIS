#ifndef CUDAFFTSURFACE_CUH
#define CUDAFFTSURFACE_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../../Definitions.h"
#include "../../Common/mymath.h"
#include "../../Common/mytimer.h"
#include "../../Common/mycomplex.h"
#include "../C/c_helper.h"
#include "cuda_helper.cuh"

__host__ void cuda_surface_fft(transform_direction dir, cudaSurfaceObject_t *surface_in, cudaSurfaceObject_t *surface_out, int n);

#endif