#include "omp_fast_index.h"

static void __inline _fft_tbbody(cpx *in, cpx *out, cpx *W, const unsigned int lmask, int steps, int dist, const int n2)
{
    const unsigned int pmask = (dist - 1) << steps;   
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n2; ++i) {
        int l = i + (i & lmask);
        int u = l + dist;
        cpxAddSubMul(&out[l], &out[u], in[l], in[u], W[(i << steps) & pmask]);
    }
}

__inline void _fft_tobb(fftDir dir, cpx *seq, cpx *W, const int n)
{
    const int n2 = (n / 2);        
    int bit = log2_32(n);
    const int lead = 32 - bit;
    int steps = 0;
    int dist = n2;
    --bit;
    _fft_tbbody(seq, seq, W, 0xFFFFFFFF << bit, steps, dist, n2);
    while (bit-- > 0) {
        dist >>= 1;
        _fft_tbbody(seq, seq, W, 0xFFFFFFFF << bit, ++steps, dist, n2);
    }
    bit_reverse(seq, dir, lead, n);
}

void ompFastIndex(fftDir dir, cpx **in, cpx **out, const int n)
{
    const int n2 = (n / 2);
    int bit = log2_32(n);
    const int lead = 32 - bit;
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);
    int steps = 0;
    int dist = n2;
    _fft_tbbody(*in, *out, W, --bit, steps, dist, n2);
    while (bit-- > 0) {
        dist >>= 1;
        _fft_tbbody(*out, *out, W, bit, ++steps, dist, n2);
    }
    bit_reverse(*out, dir, lead, n);
    free(W);
}

_inline void _do_rows(fftDir dir, cpx** seq, cpx *W, const int n)
{
#pragma omp parallel for schedule(static)
    for (int row = 0; row < n; ++row) {
        _fft_tobb(dir, seq[row], W, n);
    }
}

void ompFastIndex2D(fftDir dir, cpx** seq, const int n)
{
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);
    _do_rows(dir, seq, W, n);
    transpose(seq, n);
    _do_rows(dir, seq, W, n);
    transpose(seq, n);
    free(W);
}

