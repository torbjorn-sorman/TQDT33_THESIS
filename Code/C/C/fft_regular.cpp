#include "fft_regular.h"

#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_math.h"
#include "tb_fft_helper.h"

__inline void fft_body(cpx *in, cpx *out, cpx *W, int bit, int dist, int dist2, const int n_threads, const int n);
__inline void fft_inner_body(cpx *in, cpx *out, cpx *W, int start, int end, int bit, int dist);

void fft_regular(const double dir, cpx *in, cpx *out, const int n_threads, const int n)
{
    int bit, dist, dist2, lead;
    cpx *W;
    bit = log2_32(n);
    dist2 = n;
    dist = (n / 2);
    lead = 32 - bit;
    --bit;
    W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, lead, n_threads, n);
    fft_body(in, out, W, bit, dist, dist2, n_threads, n);
    while (bit-- > 0)
    {
        dist2 = dist;
        dist = dist >> 1;
        fft_body(out, out, W, bit, dist, dist2, n_threads, n);
    }
    bit_reverse(out, dir, lead, n_threads, n);
    free(W);
}

__inline void fft_body(cpx *in, cpx *out, cpx *W, int bit, int dist, int dist2, const int n_threads, const int n)
{
    int start, end, l, u, p;
    float imag, real;
    cpx tmp;
#ifdef _OPENMP
#pragma omp parallel shared(in, out, W, bit, dist, dist2, n)
    {
        if ((n / dist2) / n_threads > 1)
        {
#pragma omp for schedule(static, (n / dist2) / n_threads) private(start, end, l, u, p, imag, real, tmp)
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
        else {
            for (start = 0; start < n; start += dist2) {
                end = dist + start;
#pragma omp for schedule(static, (n / dist2) / n_threads) private(l, u, p, imag, real, tmp)
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
    }
#else
    for (start = 0; start < n; start += dist2) {
        end = dist + start;
        fft_inner_body(in, out, W, start, end, bit, dist);
        /*
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
        */
    }
#endif
}

__inline void fft_inner_body(cpx *in, cpx *out, cpx *W, int start, int end, int bit, int dist)
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