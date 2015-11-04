#pragma once
#ifndef GL_FFT_H
#define GL_FFT_H

//#include "../../Common/mathutil.h"

//#include "../../Common/mytimer.h"
//#include "../../Common/imglib.h"
#include <stdio.h>
#include <string>
#include "../../Common/cpx_debug.h"
#include "../../Common/fftutil.h"

#include "gl_helper.h"

int gl_validate(const int n);
int gl_2d_validate(const int n, bool write_img);

double gl_performance(const int n);
double gl_2d_performance(const int n);

#endif