#ifndef FFT_REG_H
#define FFT_REG_H

#include "tb_definitions.h"

void fft_reg(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n);
void fft2d_reg(fft_direction dir, cpx** seq, const int n_threads, const int n);

#endif