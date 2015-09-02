#ifndef TB_FFT_HELPER_H
#define TB_FFT_HELPER_H

#include "tb_definitions.h"

void twiddle_factors(cpx *W, const int n);
void twiddle_factors_alt(cpx *W, const int lead, const int n);
void twiddle_factors(cpx *W, const int lead, const int n);
void twiddle_factors_s(cpx *W, const int lead, const int n);

/* Twiddle Factors Inverse take an already calculated twiddle factor list and invert the imaginary values */

void twiddle_factors_inverse(cpx *W, const int n);
void bit_reverse(cpx *x, const double dir, const int lead, const int n);

/*  */

void fft_shift(cpx **seq, const int n);
void fft_shift_alt(cpx **seq, const int n);

#endif