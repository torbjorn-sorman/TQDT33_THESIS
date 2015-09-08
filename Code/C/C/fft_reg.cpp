#include "fft_reg.h"

#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_math.h"
#include "tb_fft_helper.h"
#include "tb_print.h"

__inline void _fft_body(cpx *in, cpx *out, cpx *W, int dist, int dist2, const int n_threads, const int n);
__inline void _fft_inner_body(cpx *in, cpx *out, const cpx *W, const int lower, const int upper, const int dist, const int cnt);
__inline void _fft_reg(fft_direction dir, cpx *seq, cpx *W, const int n_threads, const int n);

void fft_reg(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    int dist, dist2;
    cpx *W;
    dist2 = n;
    dist = (n / 2);
    W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);
    _fft_body(*in, *out, W, dist, dist2, n_threads, n);
    while ((dist2 = dist) > 1) {
        dist = dist >> 1;
        _fft_body(*out, *out, W, dist, dist2, n_threads, n);
    }
    bit_reverse(*out, dir, 32 - log2_32(n), n);
    free(W);
}

_inline void _do_rows(fft_direction dir, cpx** seq, cpx *W, const int n_threads, const int n)
{
#pragma omp parallel for schedule(static)
    for (int row = 0; row < n; ++row) {
        _fft_reg(dir, seq[row], W, n_threads, n);
    }
}

void fft2d_reg(fft_direction dir, cpx** seq, const int n_threads, const int n)
{
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);

    _do_rows(dir, seq, W, n_threads, n);
    transpose(seq, n);
    _do_rows(dir, seq, W, n_threads, n);
    transpose(seq, n);

    free(W);
}

_inline void _fft_reg(fft_direction dir, cpx *seq, cpx *W, const int n_threads, const int n)
{
    int dist, dist2;
    dist2 = n;
    dist = (n / 2);
    _fft_body(seq, seq, W, dist, dist2, n_threads, n);
    while ((dist2 = dist) > 1) {
        dist = dist >> 1;
        _fft_body(seq, seq, W, dist, dist2, n_threads, n);
    }
    bit_reverse(seq, dir, 32 - log2_32(n), n);
}

__inline void _fft_body(cpx *in, cpx *out, cpx *W, int dist, int dist2, const int n_threads, const int n)
{
    const int count = n / dist2;
#ifdef _OPENMP        
    if (count >= n_threads) {
#pragma omp parallel for schedule(static)              
        for (int lower = 0; lower < n; lower += dist2) {
            _fft_inner_body(in, out, W, lower, dist + lower, dist, count);
        }
    }
    else
    {
        int u, p, upper;
        float tmp_r, tmp_i;
        for (int lower = 0; lower < n; lower += dist2) {
            upper = dist + lower;
#pragma omp parallel for schedule(static) private(u, p, tmp_r, tmp_i)
            for (int l = lower; l < upper; ++l) {
                u = l + dist;
                p = (l - lower) * count;
                tmp_r = in[l].r - in[u].r;
                tmp_i = in[l].i - in[u].i;
                out[l].r = in[l].r + in[u].r;
                out[l].i = in[l].i + in[u].i;
                out[u].r = (W[p].r * tmp_r) - (W[p].i * tmp_i);
                out[u].i = (W[p].i * tmp_r) + (W[p].r * tmp_i);
            }
        }
    }
#else
    for (int lower = 0; lower < n; lower += dist2) {
        _fft_inner_body(in, out, W, lower, dist + lower, dist, count);
    }
#endif
}

__inline void _fft_inner_body(cpx *in, cpx *out, const cpx *W, const int lower, const int upper, const int dist, const int mul)
{
    int u, p;
    float tmp_r, tmp_i;
    for (int l = lower; l < upper; ++l) {
        u = l + dist;
        p = (l - lower) * mul;
        tmp_r = in[l].r - in[u].r;
        tmp_i = in[l].i - in[u].i;
        out[l].r = in[l].r + in[u].r;
        out[l].i = in[l].i + in[u].i;
        out[u].r = (W[p].r * tmp_r) - (W[p].i * tmp_i);
        out[u].i = (W[p].i * tmp_r) + (W[p].r * tmp_i);
    }
}