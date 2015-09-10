#ifndef FFT_FIXED_H
#define FFT_FIXED_H

#include "tb_definitions.h"

void fft_fixed(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n);

#endif