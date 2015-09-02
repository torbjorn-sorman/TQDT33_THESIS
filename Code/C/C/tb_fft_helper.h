#ifndef TB_FFT_HELPER_H
#define TB_FFT_HELPER_H

#include "tb_definitions.h"

void twiddle_factors(cpx *W, const int n);

void twiddle_factors_alt(PARAMS_TWIDDLE);
void twiddle_factors_alt_omp(PARAMS_TWIDDLE);

void twiddle_factors(PARAMS_TWIDDLE);
void twiddle_factors_omp(PARAMS_TWIDDLE);

void twiddle_factors_s(PARAMS_TWIDDLE);
void twiddle_factors_s_omp(PARAMS_TWIDDLE);

/* Twiddle Factors Inverse take an already calculated twiddle factor list and invert the imaginary values */

void twiddle_factors_inverse(cpx *W, const int n);
void twiddle_factors_inverse_omp(cpx *W, const int n);

void bit_reverse(cpx *x, const double dir, const int lead, const int n);
void bit_reverse_omp(cpx *X, const double dir, const int lead, const int n);

/*  */

void fft_shift(cpx **seq, const int n);
void fft_shift_omp(cpx **seq, const int n);

void fft_shift_alt(cpx **seq, const int n);
void fft_shift_alt_omp(cpx **seq, const int n);

#endif