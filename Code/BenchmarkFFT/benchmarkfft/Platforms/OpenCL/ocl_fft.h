#ifndef MYFFTOPENCL_H
#define MYFFTOPENCL_H

#include <stdio.h>
#include "CL\cl.h"

#include "../../Definitions.h"
#include "../../Common/mytimer.h"
#include "../../Common/mymath.h"
#include "../../Common/mycomplex.h"
#include "ocl_helper.h"

bool    OCL_validate(const int n);
bool    OCL2D_validate(const int n);

double  OCL_performance(const int n);
double  OCL2D_performance(const int n);

#endif