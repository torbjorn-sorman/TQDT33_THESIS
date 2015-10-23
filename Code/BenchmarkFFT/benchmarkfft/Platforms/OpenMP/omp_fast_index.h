#ifndef OMP_FAST_INDEX_H
#define OMP_FAST_INDEX_H

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

int openmp_fast_index_validate(const int n);
int openmp_fast_index_2d_validate(const int n);

double openmp_fast_index_performance(const int n);
double openmp_fast_index_2d_performance(const int n);

void openmp_fast_index(transform_direction dir, cpx **in, cpx **out, const int n);
void openmp_fast_index_2d(transform_direction dir, cpx** seq, const int n);

#endif