#include "fft_reg_omp.h"

#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_math.h"
#include "tb_fft_helper.h"
#include "tb_print.h"

void fft_body(cpx *in, cpx *out, cpx *W, const unsigned mask, int dist, int dist2, const int n_threads, const int n);
void fft_inner_body_(cpx *in, cpx *out, const cpx *W, const int lower, const int upper, const unsigned int mask, const int dist, const int cnt);

void fft_reg_omp(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    int dist, dist2;
    int lower, upper, start, end, count;
    int l, u, p, chunk;
    unsigned int steps, lim, mask;

    cpx *W, tmp;
    lim = log2_32(n);
    const int lead = 32 - lim;

    dist = (n / 2);
    W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);
    steps = 0;
    mask = 0xffffffff << steps;
    dist2 = n;
    count = n / dist2;
#pragma omp parallel
    {

        if (count > n_threads) {
            chunk = count / n_threads;
#pragma omp for schedule(static, chunk) private(start, end)
            for (start = 0; start < n; start += dist2) {
                end = dist + start;
                fft_inner_body_(*in, *out, W, start, end, mask, dist, count);
            }
        }
        else
        {
            chunk = dist / n_threads;
            for (lower = 0; lower < n; lower += dist2) {
                upper = dist + lower;
#pragma omp for schedule(static, chunk) private(l, u, p, tmp)
                for (l = lower; l < upper; ++l) {
                    u = l + dist;
                    p = ((l - lower) * count) & mask;
                    tmp.r = (*in)[l].r - (*in)[u].r;
                    tmp.i = (*in)[l].i - (*in)[u].i;
                    (*out)[l].r = (*in)[l].r + (*in)[u].r;
                    (*out)[l].i = (*in)[l].i + (*in)[u].i;
                    (*out)[u].r = (W[p].r * tmp.r) - (W[p].i * tmp.i);
                    (*out)[u].i = (W[p].i * tmp.r) + (W[p].r * tmp.i);
                }
            }
        }
        steps = 1;
#pragma omp barrier
        while (steps < lim) {
#pragma omp single
            {
                dist2 = dist;
                dist = dist >> 1;
                mask = 0xffffffff << steps;
                count = n / dist2;
                ++steps;
            }
#pragma omp barrier
            if (count > n_threads) {
                //printf("hej count > 4 !\n");
                chunk = count / n_threads;
#pragma omp for schedule(static, chunk) private(start, end)
                for (start = 0; start < n; start += dist2) {
                    end = dist + start;
                    fft_inner_body_(*out, *out, W, start, end, mask, dist, count);
                }
            }
            else
            {
                chunk = dist / n_threads;
                lower = 0;
#pragma omp barrier
                while (lower < n) {
#pragma omp single
                    {
                        lower += dist2;
                        upper = dist + lower;
                    }
                    //for (lower = 0; lower < n; lower += dist2) {
                    //upper = dist + lower;
#pragma omp for schedule(static, chunk) private(l, u, p, tmp)
                    for (l = lower; l < upper; ++l) {
                        u = l + dist;
                        p = ((l - lower) * count) & mask;
                        tmp.r = (*out)[l].r - (*out)[u].r;
                        tmp.i = (*out)[l].i - (*out)[u].i;
                        (*out)[l].r = (*out)[l].r + (*out)[u].r;
                        (*out)[l].i = (*out)[l].i + (*out)[u].i;
                        (*out)[u].r = (W[p].r * tmp.r) - (W[p].i * tmp.i);
                        (*out)[u].i = (W[p].i * tmp.r) + (W[p].r * tmp.i);
                    }
                }
            }
        }
#pragma omp barrier
    }
    bit_reverse((*out), dir, lead, n);
    free(W);
}

void fft_inner_body_(cpx *in, cpx *out, const cpx *W, const int lower, const int upper, const unsigned int mask, const int dist, const int mul)
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