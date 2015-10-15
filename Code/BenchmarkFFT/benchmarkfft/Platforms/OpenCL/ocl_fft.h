#ifndef MYFFTOPENCL_H
#define MYFFTOPENCL_H

#include <stdio.h>
#include "CL\cl.h"

#include "../../Definitions.h"
#include "../../Common/mytimer.h"
#include "../../Common/mymath.h"
#include "../../Common/mycomplex.h"
#include "ocl_helper.h"

bool    opencl_validate(const int n);
bool    opencl_2d_validate(const int n);

double  opencl_performance(const int n);
double  opencl_2d_performance(const int n);

#endif