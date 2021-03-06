#ifndef TB_FFT_HELPER_H
#define TB_FFT_HELPER_H

#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_definitions.h"
#include "tb_math.h"
#include "tb_print.h"

void twiddle_factors(cpx *W, fft_direction dir, const int n); 
void twiddle_factors_s(cpx *W, fft_direction dir, const int n);

/* Twiddle Factors Inverse take an already calculated twiddle factor list and invert the imaginary values */

void twiddle_factors_fast_inverse(cpx *W, const int n);
void bit_reverse(cpx *x, fft_direction dir, const int lead, const int n);

void transpose(cpx **seq, const int n);
void transpose_block(cpx **seq, const int b, const int n);

static __inline void cpxAddSubMul(cpx* in, int l, int u, cpx *outL, cpx *outU, cpx W)
{
    float x = in[l].r - in[u].r;
    float y = in[l].i - in[u].i;
    outL->r = in[l].r + in[u].r;
    outL->i = in[l].i + in[u].i;
    outU->r = (W.r * x) - (W.i * y);
    outU->i = (W.i * x) + (W.r * y);
}

__inline static void swap(cpx **in, cpx **out)
{
    cpx *tmp = *in;
    *in = *out;
    *out = tmp;
}

void fft_shift(cpx **seq, const int n);
void fft_shift_alt(cpx **seq, const int n);

#endif