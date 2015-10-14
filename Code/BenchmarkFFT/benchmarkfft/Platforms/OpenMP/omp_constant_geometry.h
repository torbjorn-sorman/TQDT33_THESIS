#ifndef OMP_CONSTANT_GEOMETRY_H
#define OMP_CONSTANT_GEOMETRY_H

#ifdef _OPENMP
#include <omp.h> 
#endif
#include <stdlib.h>
#include <iostream>
#include "../../Definitions.h"
#include "../../Common/mymath.h"
#include "../../Common/mycomplex.h"
#include "../../Common/mytimer.h"
#include "../C/c_helper.h"

int openmp_validate(const int n);
int openmp_2d_validate(const int n);

double openmp_performance(const int n);
double openmp_2d_performance(const int n);

void openmp_const_geom(fftDir dir, cpx **in, cpx **out, const int n);
void openmp_const_geom_alt(fftDir dir, cpx **in, cpx **out, const int n);
void openmp_const_geom_2d(fftDir dir, cpx **in, cpx **out, const int n);

#endif