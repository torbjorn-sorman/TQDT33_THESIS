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
    int bit, steps, dist, dist2;
    unsigned int mask;
    cpx *W;
    bit = log2_32(n);
    const int lead = 32 - bit;
    bit -= 1;
    steps = bit;

    dist2 = n;
    dist = (n / 2);
    W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n_threads, n);    

    mask = 0xffffffff << (steps - bit);
    fft_body(*in, *out, W, mask, dist, dist2, n_threads, n);
    while (bit-- >= 0) {
        dist2 = dist;
        dist = dist >> 1;
        mask = 0xffffffff << (steps - bit);
        fft_body(*out, *out, W, mask, dist, dist2, n_threads, n);
    }
    bit_reverse(*out, dir, lead, n_threads, n);
    free(W);
}

void fft_body(cpx *in, cpx *out, cpx *W, const unsigned mask, int dist, int dist2, const int n_threads, const int n)
{
    int lower, upper, count;
#ifdef _OPENMP    
    int l, u, p, chunk;
    float real, imag;
    cpx tmp;
    chunk = (n / dist2) / n_threads;
    if (chunk > 0) {
        int start, end;
#pragma omp parallel for schedule(static, chunk) private(start, end) shared(in, out, W, bit, dist, dist2, n)        
        for (start = 0; start < n; start += dist2) {
            end = dist + start;
            fft_inner_body(in, out, W, start, end, bit, dist);            
        }
    }
    else
    {
        chunk = (dist > n_threads) ? (dist / n_threads) : dist;
        for (lower = 0; lower < n; lower += dist2) {
            upper = dist + lower;
#pragma omp parallel for schedule(static, chunk) private(l, u, p, real, imag, tmp) shared(in, out, W, bit, dist, lower, upper)
            for (l = lower; l < upper; ++l) {
                u = l + dist;
                p = (u >> bit);
                tmp = in[l];
                real = in[u].r * W[p].r - in[u].i * W[p].i;
                imag = in[u].i * W[p].r + in[u].r * W[p].i;
                out[l].r = tmp.r - real;
                out[l].i = tmp.i - imag;
                out[u].r = tmp.r + real;
                out[u].i = tmp.i + imag;
            }
        }
    }
#else
    count = n / dist2;//n / dist2;
    for (lower = 0; lower < n; lower += dist2) {
        upper = dist + lower;
        fft_inner_body(in, out, W, lower, upper, mask, dist, count);
    }
    //printf("\n");
#endif
}

void fft_inner_body(cpx *in, cpx *out, const cpx *W, const int lower, const int upper, const unsigned int mask, const int dist, const int mul)
{
    int l, u, p, cnt;
    //float real, imag;
    cpx tmp;
    cnt = -mul;
    for (l = lower; l < upper; ++l) {
        u = l + dist;
        
        // do better?
        //p = ((l - lower) * mul) & mask;
        p = (cnt += mul) & mask;

        //printf("p: %d\n", p);

        tmp.r = in[l].r - in[u].r;
        tmp.i = in[l].i - in[u].i;
        out[l].r = in[l].r + in[u].r;
        out[l].i = in[l].i + in[u].i;        
        out[u].r = (W[p].r * tmp.r) - (W[p].i * tmp.i);
        out[u].i = (W[p].i * tmp.r) + (W[p].r * tmp.i);
    }
}