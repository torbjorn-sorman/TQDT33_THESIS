#ifndef FFT_TOBB_H
#define FFT_TOBB_H

#include "tb_definitions.h"

void fft_tobb(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n);
void fft2d_tobb(fft_direction dir, cpx** seq, const int n_threads, const int n);

#endif