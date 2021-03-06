#pragma once
#ifndef C_FFT_H
#define C_FFT_H

#include "../../Definitions.h"
#include "../../Common/mathutil.h"
#include "../../Common/mycomplex.h"
#include "../../Common/mytimer.h"
#include "../../Common/imglib.h"
#include "c_helper.h"

int c_validate(const int n);
int c_2d_validate(const int n, bool write_img);

double c_performance(const int n);
double c_2d_performance(const int n);

void c_const_geom(transform_direction dir, cpx **in, cpx **out, const int n);
void c_const_geom_alt(transform_direction dir, cpx **in, cpx **out, const int n);
void c_const_geom_2d(transform_direction dir, cpx** in, cpx** out, const int n);

#endif