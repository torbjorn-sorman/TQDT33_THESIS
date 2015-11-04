#ifndef OMP_CONSTANT_GEOMETRY_H
#define OMP_CONSTANT_GEOMETRY_H

#ifdef _OPENMP
#include <omp.h> 
#endif
#include "../../Definitions.h"
#include "../../Common/mathutil.h"
#include "../../Common/mycomplex.h"
#include "../../Common/mytimer.h"
#include "../C/c_helper.h"

int openmp_validate(const int n);
int openmp_2d_validate(const int n, bool write_img);

double openmp_performance(const int n);
double openmp_2d_performance(const int n);

void openmp_const_geom(transform_direction dir, cpx **in, cpx **out, const int n);
void openmp_const_geom_alt(transform_direction dir, cpx **in, cpx **out, const int n);
void openmp_const_geom_2d(transform_direction dir, cpx **in, cpx **out, const int n);

#endif