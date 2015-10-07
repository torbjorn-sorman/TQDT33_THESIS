#ifndef OMP_FAST_INDEX_H
#define OMP_FAST_INDEX_H

#ifdef _OPENMP
#include <omp.h> 
#endif
#include <stdlib.h>
#include "../../Definitions.h"
#include "../../Common/mymath.h"
#include "../../Common/mycomplex.h"
#include "../../Common/mytimer.h"
#include "../C/c_helper.h"

int ompFastIndex_validate(const int n);
int ompFastIndex2D_validate(const int n);

double ompFastIndex_runPerformance(const int n);
double ompFastIndex2D_runPerformance(const int n);

void ompFastIndex(fftDir dir, cpx **in, cpx **out, const int n);
void ompFastIndex2D(fftDir dir, cpx** seq, const int n);

#endif