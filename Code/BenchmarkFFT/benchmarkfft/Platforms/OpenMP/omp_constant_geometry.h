#ifndef OMP_CONSTANT_GEOMETRY_H
#define OMP_CONSTANT_GEOMETRY_H

#ifdef _OPENMP
#include <omp.h> 
#endif
#include <stdlib.h>
#include "../../Definitions.h"
#include "../../Common/mymath.h"
#include "../../Common/mycomplex.h"
#include "../../Common/mytimer.h"
#include "../C/c_helper.h"

int ompConstantGeometry_validate(const int n);
int ompConstantGeometry2D_validate(const int n);

double ompConstantGeometry_runPerformance(const int n);
double ompConstantGeometry2D_runPerformance(const int n);

void ompConstantGeometry(fftDir dir, cpx **in, cpx **out, const int n);
void ompConstantGeometryAlternate(fftDir dir, cpx **in, cpx **out, const int n);
void ompConstantGeometry2D(fftDir dir, cpx** seq, const int n);

#endif