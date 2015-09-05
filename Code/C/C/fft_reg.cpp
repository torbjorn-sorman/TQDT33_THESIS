#include "fft_reg.h"

#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_math.h"
#include "tb_fft_helper.h"
#include "tb_print.h"

void fft_body(cpx *in, cpx *out, cpx *W, const unsigned mask, int dist, int dist2, const int n_threads, const int n);
void fft_inner_body(cpx *in, cpx *out, const cpx *W, const int lower, const int upper, const unsigned int mask, const int dist, const int cnt);

void fft_reg(const double dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    int dist;
    unsigned int steps, lim;
    cpx *W;
    lim = log2_32(n);
    const int lead = 32 - lim;
    dist = (n / 2);
    W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n_threads, n);
    steps = 0;  
    fft_body(*in, *out, W, 0xffffffff << steps, dist, n >> steps, n_threads, n);
    while (++steps < lim) {
        dist = dist >> 1;
        fft_body(*out, *out, W, 0xffffffff << steps, dist, n >> steps, n_threads, n);
    }
    bit_reverse(*out, dir, lead, n_threads, n);
    free(W);
}

void fft_body(cpx *in, cpx *out, cpx *W, const unsigned mask, int dist, int dist2, const int n_threads, const int n)
{
    int lower, upper, count;
    count = n / dist2;
#ifdef _OPENMP    
    int l, u, p, chunk;
    cpx tmp;
    if (count > n_threads) {
        chunk = count / n_threads;
#pragma omp parallel for schedule(static, chunk) private(lower, upper) shared(in, out, W, mask, dist, dist2, n, count)                
        for (lower = 0; lower < n; lower += dist2) {
            upper = dist + lower;
            fft_inner_body(in, out, W, lower, upper, mask, dist, count);
        }
    }
    else
    {
        chunk = dist / n_threads;
        for (lower = 0; lower < n; lower += dist2) {
            upper = dist + lower;
#pragma omp parallel for schedule(static, chunk) private(l, u, p, tmp) shared(in, out, W, mask, dist, lower, upper, chunk, count)
            for (l = lower; l < upper; ++l) {
                u = l + dist;
                p = ((l - lower) * count) & mask;
                tmp.r = in[l].r - in[u].r;
                tmp.i = in[l].i - in[u].i;
                out[l].r = in[l].r + in[u].r;
                out[l].i = in[l].i + in[u].i;
                out[u].r = (W[p].r * tmp.r) - (W[p].i * tmp.i);
                out[u].i = (W[p].i * tmp.r) + (W[p].r * tmp.i);
            }
        }
    }
#else
    for (lower = 0; lower < n; lower += dist2) {
        upper = dist + lower;
        fft_inner_body(in, out, W, lower, upper, mask, dist, count);
    }
#endif
}

void fft_inner_body(cpx *in, cpx *out, const cpx *W, const int lower, const int upper, const unsigned int mask, const int dist, const int mul)
{
    int l, u, p;
    cpx tmp;
    for (l = lower; l < upper; ++l) {
        u = l + dist;
        p = ((l - lower) * mul) & mask;
        tmp.r = in[l].r - in[u].r;
        tmp.i = in[l].i - in[u].i;
        out[l].r = in[l].r + in[u].r;
        out[l].i = in[l].i + in[u].i;        
        out[u].r = (W[p].r * tmp.r) - (W[p].i * tmp.i);
        out[u].i = (W[p].i * tmp.r) + (W[p].r * tmp.i);
    }
}