#ifndef TB_FFT_HELPER_H
#define TB_FFT_HELPER_H

#include "tb_definitions.h"

void twiddle_factors(cpx *W, fft_direction dir, const int n); 
void twiddle_factors_s(cpx *W, fft_direction dir, const int n);

/* Twiddle Factors Inverse take an already calculated twiddle factor list and invert the imaginary values */

void twiddle_factors_fast_inverse(cpx *W, const int n);
void bit_reverse(cpx *x, fft_direction dir, const int lead, const int n);

void transpose(cpx **seq, const int n);
void transpose_block(cpx **seq, const int b, const int n);
void transpose_block2(cpx **seq, const int b, const int n);
void transpose_block3(cpx **seq, const int b, const int n);

/*  */

void fft_shift(cpx **seq, const int n);
void fft_shift_alt(cpx **seq, const int n);

#endif