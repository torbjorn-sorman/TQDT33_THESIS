#include "fft_fixed.h"

#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_math.h"
#include "tb_fft_helper.h"
#include "tb_print.h"

#include "fft_generated_fixed.h"
//#include "fft_generated_fixed_const.h"

__inline static void fft_xn(fft_direction dir, cpx *in, cpx *out, cpx *W, const int n);

void fft_fixed(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n)
{
#ifdef GENERATED_FIXED_4
    if (n == 4) {
        if (dir == FORWARD_FFT) {
            fft_x4(*in, *out);
        }
        else  {
            fft_x4inv(*in, *out);
        }
    }
#endif
#ifdef GENERATED_FIXED_8
    else if (n == 8) {
        if (dir == FORWARD_FFT)
            fft_x8(*in, *out);
        else
            fft_x8inv(*in, *out);
    }
#endif
#ifdef GENERATED_FIXED_16
    else if (n == 16) {
        if (dir == FORWARD_FFT)
            fft_x16(*in, *out);
        else
            fft_x16inv(*in, *out);
    }
#endif
#ifdef GENERATED_FIXED_32
    else if (n == 32) {
        if (dir == FORWARD_FFT)
            fft_x32(*in, *out);
        else
            fft_x32inv(*in, *out);
    }
#endif
    if (n < 32)
        return;
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);

    fft_xn(dir, *in, *out, W, n);
    bit_reverse(*out, dir, 32 - log2_32(n), n);

    fft_xn(dir, *in, *out, W, n);
    bit_reverse(*out, dir, 32 - log2_32(n), n);

    free(W);

}

__inline void _fft_tbbody_f(cpx *in, cpx *out, cpx *W, int bit, int steps, int dist, int dist2, const int n2);

#define FIXED_4

__inline static void fft_xn(fft_direction dir, cpx *in, cpx *out, cpx *W, const int n)
{
#ifdef FIXED_4
    const int lg = 2;
    const int len = 4;
#else
#ifdef FIXED_8
    const int lg = 3;
    const int len = 8;
#else 
#ifdef FIXED_16
    const int lg = 4;
    const int len = 16;
#else
    const int lg = 5;
    const int len = 32;
#endif
#endif
#endif
    const int n2 = n / 2;
    int dist2 = n;
    int dist = n2;
    int steps = 0;
    int bit = log2_32(n) - 1;
    _fft_tbbody_f(in, out, W, bit, steps, dist, dist2, n2);
    while (bit-- > lg)
    {
        dist2 = dist;
        dist = dist >> 1;
        _fft_tbbody_f(out, out, W, bit, ++steps, dist, dist2, n2);
    }
    if (dir == FORWARD_FFT) {
        for (int i = 0; i < n; i += len) {
#ifdef FIXED_4
            fft_fixed_x4(&(out[i]), &(out[i]));
#else
#ifdef FIXED_8
            fft_fixed_x8(&(out[i]), &(out[i]));
#else 
#ifdef FIXED_16
            fft_fixed_x16(&(out[i]), &(out[i]));
#else
            fft_fixed_x32(&(out[i]), &(out[i]));
#endif
#endif
#endif
        }
    }
    else {
        for (int i = 0; i < n; i += len) {
#ifdef FIXED_4
            fft_fixed_x4inv(&(out[i]), &(out[i]));
#else
#ifdef FIXED_8
            fft_fixed_x8inv(&(out[i]), &(out[i]));
#else 
#ifdef FIXED_16
            fft_fixed_x16inv(&(out[i]), &(out[i]));
#else
            fft_fixed_x32inv(&(out[i]), &(out[i]));
#endif
#endif
#endif
        }
    }
}

__inline void _fft_tbbody_f(cpx *in, cpx *out, cpx *W, int bit, int steps, int dist, int dist2, const int n2)
{
    const unsigned int pmask = (dist - 1) << steps;
    const unsigned int lmask = 0xFFFFFFFF << bit;
    int l, u, p;
    float tmp_r, tmp_i;
#pragma omp parallel for schedule(static) private(l, u, p, tmp_r, tmp_i)    
    for (int i = 0; i < n2; ++i)
    {
        l = i + (i & lmask);
        u = l + dist;
        p = (i << steps) & pmask;
        tmp_r = in[l].r - in[u].r;
        tmp_i = in[l].i - in[u].i;
        out[l].r = in[l].r + in[u].r;
        out[l].i = in[l].i + in[u].i;
        out[u].r = (W[p].r * tmp_r) - (W[p].i * tmp_i);
        out[u].i = (W[p].i * tmp_r) + (W[p].r * tmp_i);
    }
}