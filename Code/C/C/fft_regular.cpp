#include "fft_regular.h"

#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_math.h"
#include "tb_fft_helper.h"
#include "tb_print.h"

void fft_body(cpx *in, cpx *out, cpx *W, int bit, int dist, int dist2, const int n_threads, const int n);
void fft_inner_body(cpx *in, cpx *out, cpx *W, int start, int end, int bit, int dist);
//void fft_body2(cpx *in, cpx *out, cpx *W, int bit, int dist, int dist2, int n_threads, const int n);

void fft_regular(const double dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    int bit, dist, dist2, lead;
    cpx *W;

    bit = log2_32(n);
    dist2 = n;
    dist = (n / 2);
    lead = 32 - bit;
    --bit;
    W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, lead, n_threads, n);
    //cpx *Ws = (cpx *)malloc(sizeof(cpx) * n);
    //twiddle_factors_short(Ws, dir, lead, n_threads, n);

    //console_print_cmp(W, Ws, n);
    //getchar();
    //mask = 0xffffffff << (steps - bit);

    fft_body(*in, *out, W, bit, dist, dist2, n_threads, n);
    while (bit-- > 0) {
        dist2 = dist;
        dist = dist >> 1;
        fft_body(*out, *out, W, bit, dist, dist2, n_threads, n);
    }
    bit_reverse(*out, dir, lead, n_threads, n);
    free(W);
}

void fft_body(cpx *in, cpx *out, cpx *W, int bit, int dist, int dist2, const int n_threads, const int n)
{
    int lower, upper;
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
    for (lower = 0; lower < n; lower += dist2) {
        upper = dist + lower;
        fft_inner_body(in, out, W, lower, upper, bit, dist);
    }
#endif
}

void fft_inner_body(cpx *in, cpx *out, cpx *W, int start, int end, int bit, int dist)
{
    int l, u, p;
    float real, imag;
    cpx tmp;
    for (l = start; l < end; ++l) {
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

/*
void fft_body2(cpx *in, cpx *out, cpx *W, int bit, int dist, int dist2, int n_threads, const int n)
{
    int start, end, l, u, p, chunk;
    float imag, real;
    cpx tmp;
    chunk = (n / dist2) / n_threads;
#pragma omp parallel for schedule(static, chunk) private(start, end, l, u, p, imag, real, tmp) shared(in, out, W, bit, dist, dist2, n)
    for (start = 0; start < n; start += dist2) {
        end = dist + start;
        for (l = start; l < end; ++l) {
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
*/