#ifndef FFT_REG_OMP_H
#define FFT_REG_OMP_H

#include "tb_definitions.h"

void fft_reg_omp(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n);

#endif