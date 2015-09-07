#ifndef FFT_TOBB_H
#define FFT_TOBB_H

#include "tb_definitions.h"

void fft_tobb(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n);

#endif