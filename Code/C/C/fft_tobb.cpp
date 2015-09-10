
#include "fft_tobb.h"

#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_math.h"
#include "tb_fft_helper.h"
#include "tb_print.h"

__inline void _fft_tbbody(cpx *in, cpx *out, cpx *W, int bit, int steps, int dist, int dist2, const int n2);
__inline void _fft_tobb(fft_direction dir, cpx *seq, cpx *W, const int n);

void fft_tobb(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    const int n2 = (n / 2);
    int bit, dist, dist2;
    unsigned int steps;
    cpx *W;
    bit = log2_32(n);    
    const int lead = 32 - bit;

    W = (cpx *)malloc(sizeof(cpx) * n);    
    twiddle_factors(W, dir, n);
    steps = 0;
    dist2 = n;
    dist = n2;
    --bit;
    
    _fft_tbbody(*in, *out, W, bit, steps, dist, dist2, n2);
    while (bit-- > 0)
    {
        dist2 = dist;
        dist = dist >> 1;
        _fft_tbbody(*out, *out, W, bit, ++steps, dist, dist2, n2);
    }
    bit_reverse(*out, dir, lead, n);
    free(W);
}

_inline void _do_rows(fft_direction dir, cpx** seq, cpx *W, const int n)
{
#pragma omp parallel for schedule(static)
    for (int row = 0; row < n; ++row) {
        _fft_tobb(dir, seq[row], W, n);
    }
}

void fft2d_tobb(fft_direction dir, cpx** seq, const int n_threads, const int n)
{
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);

    _do_rows(dir, seq, W, n);
    transpose(seq, n);
    _do_rows(dir, seq, W, n);
    transpose(seq, n);

    free(W);
}

__inline void _fft_tobb(fft_direction dir, cpx *seq, cpx *W, const int n)
{
    const int n2 = (n / 2);
    int bit, dist, dist2;
    unsigned int steps;

    bit = log2_32(n);
    const int lead = 32 - bit;
    steps = 0;
    dist2 = n;
    dist = n2;
    --bit;

    _fft_tbbody(seq, seq, W, bit, steps, dist, dist2, n2);
    while (bit-- > 0)
    {
        dist2 = dist;
        dist = dist >> 1;
        _fft_tbbody(seq, seq, W, bit, ++steps, dist, dist2, n2);
    }
    bit_reverse(seq, dir, lead, n);
}

__inline void _fft_tbbody(cpx *in, cpx *out, cpx *W, int bit, int steps, int dist, int dist2, const int n2)
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