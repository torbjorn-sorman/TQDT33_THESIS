#ifndef DX_FFT_H
#define DX_FFT_H

#define _WIN32_WINNT 0x600
#include <stdio.h>

#include <d3d11.h>
#include <d3dcompiler.h>

#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")

#include <stdlib.h>
#include "../../Definitions.h"
#include "../../Common/mymath.h"
#include "../../Common/mycomplex.h"
#include "../../Common/mytimer.h"
#include "../../Common/imglib.h"
#include "dx_helper.h"

void dx_fft();

int dx_validate(const int n);
int dx_2d_validate(const int n);

double dx_performance(const int n);
double dx_2d_performance(const int n);

#endif