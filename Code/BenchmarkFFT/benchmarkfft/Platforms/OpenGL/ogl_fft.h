#pragma once
#ifndef OGL_FFT_H
#define OGL_FFT_H

/*
#include "../../Common/mymath.h"
#include "../../Common/mycomplex.h"
#include "../../Common/mytimer.h"
#include "../../Common/imglib.h"
*/
#include "ogl_helper.h"

int ogl_validate(const int n);
int ogl_2d_validate(const int n, bool write_img);

double ogl_performance(const int n);
double ogl_2d_performance(const int n);

#endif