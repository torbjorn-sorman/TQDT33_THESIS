#pragma once
#ifndef MYFFTOPENCL_H
#define MYFFTOPENCL_H

#include "../../Definitions.h"
#include "../../Common/mytimer.h"
#include "../../Common/mathutil.h"
#include "../../Common/fftutil.h"
#include "../../Common/mycomplex.h"
#include "ocl_helper.h"

bool    ocl_validate(const int n);
bool    ocl_2d_validate(const int n, bool write_img);

double  ocl_performance(const int n);
double  ocl_2d_performance(const int n);

#endif