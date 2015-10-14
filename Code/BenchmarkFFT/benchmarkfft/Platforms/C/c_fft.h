#ifndef C_FFT_H
#define C_FFT_H

#include <stdio.h>
#include <stdlib.h>
#include "../../Definitions.h"
#include "../../Common/mymath.h"
#include "../../Common/mycomplex.h"
#include "../../Common/mytimer.h"
#include "../../Common/imglib.h"
#include "c_helper.h"

int c_validate(const int n);
int c_2d_validate(const int n);

double c_performance(const int n);
double c_2d_performance(const int n);

void c_const_geom(fftDir dir, cpx **in, cpx **out, const int n);
void c_const_geom_alt(fftDir dir, cpx **in, cpx **out, const int n);
void c_const_geom_2d(fftDir dir, cpx** in, cpx** out, const int n);

#endif