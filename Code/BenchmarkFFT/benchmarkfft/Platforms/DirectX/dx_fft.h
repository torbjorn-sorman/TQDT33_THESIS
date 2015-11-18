#pragma once
#ifndef DX_FFT_H
#define DX_FFT_H

#include "../../Definitions.h"
#include "../../Common/fftutil.h"
#include "../../Common/mycomplex.h"
#include "../../Common/mytimer.h"
#include "../../Common/imglib.h"
#include "dx_helper.h"

int dx_validate(const int n);
int dx_2d_validate(const int n, bool write_img);

double dx_performance(const int n);
double dx_2d_performance(const int n);

#endif