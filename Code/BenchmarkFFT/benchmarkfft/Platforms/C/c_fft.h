#ifndef C_FFT_H
#define C_FFT_H

#include <stdio.h>
#include <stdlib.h>
#include "../../Definitions.h"
#include "../../Common/mymath.h"
#include "../../Common/mycomplex.h"
#include "../../Common/mytimer.h"
#include "c_helper.h"

int cConstantGeometry_validate(const int n);
int cConstantGeometry2D_validate(const int n);

double cConstantGeometry_runPerformance(const int n);
double cConstantGeometry2D_runPerformance(const int n);

void cConstantGeometry(fftDir dir, cpx **in, cpx **out, const int n);
void cConstantGeometryAlternate(fftDir dir, cpx **in, cpx **out, const int n);
void cConstantGeometry2D(fftDir dir, cpx** seq, const int n);

#endif