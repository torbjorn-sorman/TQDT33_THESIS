#ifndef DX_FFT_H
#define DX_FFT_H

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>

#include <d3d11.h>
#include <d3dcompiler.h>


#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")

#include "../../Definitions.h"
#include "../../Common/mymath.h"
#include "../../Common/mycomplex.h"
#include "../../Common/mytimer.h"
#include "../../Common/imglib.h"
#include "../../Common/cpx_debug.h"
#include "dx_helper.h"

int dx_validate(const int n);
int dx_2d_validate(const int n, bool write_img);

double dx_performance(const int n);
double dx_2d_performance(const int n);

#endif