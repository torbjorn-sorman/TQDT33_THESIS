#pragma once
#ifndef GL_FFT_H
#define GL_FFT_H

//#include "../../Common/mymath.h"

//#include "../../Common/mytimer.h"
//#include "../../Common/imglib.h"

#include "gl_helper.h"

int gl_validate(const int n);
int gl_2d_validate(const int n, bool write_img);

double gl_performance(const int n);
double gl_2d_performance(const int n);

#endif