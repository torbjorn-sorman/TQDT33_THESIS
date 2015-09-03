#ifndef TB_FFT_HELPER_H
#define TB_FFT_HELPER_H

#include "tb_definitions.h"

__inline void twiddle_factors(cpx *W, const int n_threads, const int n);
__inline void twiddle_factors(cpx *W, const int lead, const int n_threads, const int n);
__inline void twiddle_factors_alt(cpx *W, const int lead, const int n_threads, const int n);
__inline void twiddle_factors_s(cpx *W, const int lead, const int n_threads, const int n);

/* Twiddle Factors Inverse take an already calculated twiddle factor list and invert the imaginary values */

__inline void twiddle_factors_inverse(cpx *W, const int n_threads, const int n);
__inline void bit_reverse(cpx *x, const double dir, const int lead, const int n_threads, const int n);

/*  */

__inline void fft_shift(cpx **seq, const int n_threads, const int n);
__inline void fft_shift_alt(cpx **seq, const int n_threads, const int n);

#endif