#include "fft_fixed.h"

__inline static void fft_xn(fft_direction dir, cpx *in, cpx *out, cpx *W, const int n);

void fft_fixed(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n)
{
#ifdef GENERATED_FIXED_SIZE    
    if (fixed_size_fft(dir, *in, *out, GEN_BIT_REVERSE_ORDER, n)) {
        return;
    }
#endif
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);
    fft_xn(dir, *in, *out, W, n);
    free(W);
    bit_reverse(*out, dir, 32 - log2_32(n), n);
}

__inline static void _fft_tbbody_f(cpx *in, cpx *out, cpx *W, const int bit, const int steps, const int dist, const int n2);

__inline static void fft_xn(fft_direction dir, cpx *in, cpx *out, cpx *W, const int n)
{
#ifdef GENERATED_FIXED_SIZE
    const int lg = log2_32(GENERATED_FIXED_SIZE);
#else
    const int lg = 0;
#endif
    const float angle = (dir * M_2_PI) / ((float)n);
    const int n2 = n / 2;
    int dist = n2;
    int steps = 0;
    int bit = log2_32(n) - 1;
    _fft_tbbody_f(in, out, W, bit, steps, dist, n2);  
    while (bit-- > lg)
    {
        dist = dist >> 1;
        ++steps;
        _fft_tbbody_f(out, out, W, bit, steps, dist, n2);
    }
#ifdef GENERATED_FIXED_SIZE
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i += GENERATED_FIXED_SIZE) {
        fixed_size_fft(dir, &(out[i]), &(out[i]), GEN_NORMAL_ORDER, GENERATED_FIXED_SIZE);
    }
#endif
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