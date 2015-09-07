#ifndef FFT_REG_H
#define FFT_REG_H

#include "tb_definitions.h"

void fft_reg(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n);

#endif