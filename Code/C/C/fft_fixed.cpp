#include "fft_fixed.h"

#define PRECALC_TWIDDLE

#ifdef PRECALC_TWIDDLE
__inline static void fft_xn(fft_direction dir, cpx *in, cpx *out, cpx *W, const int n);
#else
__inline static void fft_xn(fft_direction dir, cpx *in, cpx *out, const int n);
#endif


void fft_fixed(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    if (dir == FORWARD_FFT) {
        switch (n)
        {
#ifdef GENERATED_FIXED_4
        case 4:
            fft_x4(*in, *out);
            return;
#endif
#ifdef GENERATED_FIXED_8
        case 8:
            fft_x8(*in, *out);
            return;
#endif
#ifdef GENERATED_FIXED_16
        case 16:
            fft_x16(*in, *out);
            return;
#endif
#ifdef GENERATED_FIXED_32
        case 32:
            fft_x32(*in, *out);
            return;
#endif
        default:
            break;
        }
    }
    else {
        switch (n)
        {
#ifdef GENERATED_FIXED_4
        case 4:
            fft_x4inv(*in, *out);
            return;
#endif
#ifdef GENERATED_FIXED_8
        case 8:
            fft_x8inv(*in, *out);
            return;
#endif
#ifdef GENERATED_FIXED_16
        case 16:
            fft_x16inv(*in, *out);
            return;
#endif
#ifdef GENERATED_FIXED_32
        case 32:
            fft_x32inv(*in, *out);
            return;
#endif
        default:
            break;
        }
    }
#ifdef PRECALC_TWIDDLE
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);
    fft_xn(dir, *in, *out, W, n);
    free(W);
#else
    fft_xn(dir, *in, *out, n);
#endif
    bit_reverse(*out, dir, 32 - log2_32(n), n);
}

__inline static void _fft_tbbody_f(cpx *in, cpx *out, cpx *W, const int bit, const int steps, const int dist, const int n2);
__inline static void _fft_tbbody_f(cpx *in, cpx *out, const float ang, const int bit, const int steps, const int dist, const int n2);

#define FIXED_8

#ifdef PRECALC_TWIDDLE
__inline static void fft_xn(fft_direction dir, cpx *in, cpx *out, cpx *W, const int n)
#else
__inline static void fft_xn(fft_direction dir, cpx *in, cpx *out, const int n)
#endif
{
#if defined(FIXED_4)
    const int lg = 2;
    const int len = 4;
#elif defined(FIXED_8)
    const int lg = 3;
    const int len = 8;
#elif defined(FIXED_16)
    const int lg = 4;
    const int len = 16;
#elif defined(FIXED_32)
    const int lg = 5;
    const int len = 32;
#endif
    const float angle = (dir * M_2_PI) / ((float)n);
    const int n2 = n / 2;
    int dist = n2;
    int steps = 0;
    int bit = log2_32(n) - 1;
#ifdef PRECALC_TWIDDLE
    _fft_tbbody_f(in, out, W, bit, steps, dist, n2);
#else
    _fft_tbbody_f(in, out, angle, bit, steps, dist, dist2, n2);
#endif
    while (bit-- > lg)
    {
        dist = dist >> 1;
        ++steps;
#ifdef PRECALC_TWIDDLE
        _fft_tbbody_f(out, out, W, bit, steps, dist, n2);
#else
        _fft_tbbody_f(out, out, angle, bit, steps, dist, dist2, n2);
#endif

    }
    if (dir == FORWARD_FFT) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i += len) {
#if defined(FIXED_4)
            fft_fixed_x4(&(out[i]), &(out[i]));
#elif defined(FIXED_8)
            fft_fixed_x8(&(out[i]), &(out[i]));
#elif defined(FIXED_16)
            fft_fixed_x16(&(out[i]), &(out[i]));
#elif defined(FIXED_32)
            fft_fixed_x32(&(out[i]), &(out[i]));
#endif
        }
    }
    else {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i += len) {
#if defined(FIXED_4)
            fft_fixed_x4inv(&(out[i]), &(out[i]));
#elif defined(FIXED_8)
            fft_fixed_x8inv(&(out[i]), &(out[i]));
#elif defined(FIXED_16)
            fft_fixed_x16inv(&(out[i]), &(out[i]));
#elif defined(FIXED_32)
            fft_fixed_x32inv(&(out[i]), &(out[i]));
#endif
        }
    }
}

static __inline void _fft_tbbody_f(cpx *in, cpx *out, cpx *W, const int bit, const int steps, const int dist, const int n2)
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

static __inline void _fft_tbbody_f(cpx *in, cpx *out, const float ang, const int bit, const int steps, const int dist, const int n2)
{
    const unsigned int pmask = (dist - 1) << steps;
    const unsigned int lmask = 0xFFFFFFFF << bit;
    int l, u, p, o;
    float tmp_r, tmp_i, a;
    cpx w;
    o = -1;
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
        if (p != o) {
            a = ang * p;
            w.i = sin(a);
            w.r = cos(a);
            o = p;
        }
        out[u].r = (w.r * tmp_r) - (w.i * tmp_i);
        out[u].i = (w.i * tmp_r) + (w.r * tmp_i);
    }
}

#ifdef PRECALC_TWIDDLE
__inline static void fft_xn2(fft_direction dir, cpx *in, cpx *out, cpx *W, const int n)
#else
__inline static void fft_xn(fft_direction dir, cpx *in, cpx *out, const int n)
#endif
{
#if defined(FIXED_4)
    const int lg = 2;
    const int len = 4;
#elif defined(FIXED_8)
    const int lg = 3;
    const int len = 8;
#elif defined(FIXED_16)
    const int lg = 4;
    const int len = 16;
#elif defined(FIXED_32)
    const int lg = 5;
    const int len = 32;
#endif
    const float angle = (dir * M_2_PI) / ((float)n);
    const int n2 = n / 2;
    int dist = n2;
    int steps = 0;
    int bit = log2_32(n) - 1;
#ifdef PRECALC_TWIDDLE
    _fft_tbbody_f(in, out, W, bit, steps, dist, n2);
#else
    _fft_tbbody_f(in, out, angle, bit, steps, dist, dist2, n2);
#endif
    while (bit-- > lg)
    {
        dist = dist >> 1;
        ++steps;
#ifdef PRECALC_TWIDDLE
        _fft_tbbody_f(out, out, W, bit, steps, dist, n2);
#else
        _fft_tbbody_f(out, out, angle, bit, steps, dist, dist2, n2);
#endif

    }
    if (dir == FORWARD_FFT) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i += len) {
#if defined(FIXED_4)
            fft_fixed_x4(&(out[i]), &(out[i]));
#elif defined(FIXED_8)
            fft_fixed_x8(&(out[i]), &(out[i]));
#elif defined(FIXED_16)
            fft_fixed_x16(&(out[i]), &(out[i]));
#elif defined(FIXED_32)
            fft_fixed_x32(&(out[i]), &(out[i]));
#endif
        }
    }
    else {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i += len) {
#if defined(FIXED_4)
            fft_fixed_x4inv(&(out[i]), &(out[i]));
#elif defined(FIXED_8)
            fft_fixed_x8inv(&(out[i]), &(out[i]));
#elif defined(FIXED_16)
            fft_fixed_x16inv(&(out[i]), &(out[i]));
#elif defined(FIXED_32)
            fft_fixed_x32inv(&(out[i]), &(out[i]));
#endif
        }
    }
}