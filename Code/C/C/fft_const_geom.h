#ifndef FFT_CONST_GEOM_H
#define FFT_CONST_GEOM_H

#include "tb_definitions.h"

void fft_const_geom(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n);
void fft_const_geom_2(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n);

void fft2d_const_geom(fft_direction dir, cpx** seq, const int n_threads, const int n);

#endif