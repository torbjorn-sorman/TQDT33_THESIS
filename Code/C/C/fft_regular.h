#ifndef FFT_REGULAR_H
#define FFT_REGULAR_H

#include "tb_definitions.h"

void fft_regular(const double dir, cpx **in, cpx **out, const int n_threads, const int n);

#endif