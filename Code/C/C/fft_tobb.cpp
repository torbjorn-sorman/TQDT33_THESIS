
#include "fft_tobb.h"

#include "tb_math.h"
#include "tb_fft_helper.h"

void _fft_body(cpx *in, cpx *out, cpx *W, int bit, int dist, int dist2, const int n_threads, const int n2);
void inner_fft2(cpx *in, cpx *out, cpx *W, int bit, int dist, int dist2, const int n_threads, const int n2);

void fft_tobb(const double dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    const int n2 = (n / 2);
    int bit, lead, dist, dist_2;
    bit = log2_32(n);
    cpx *W;
    W = (cpx *)malloc(sizeof(cpx) * n);
    lead = 32 - bit;
    twiddle_factors(W, dir, lead, n_threads, n);

    dist_2 = n;
    dist = n2;
    _fft_body(*in, *out, W, --bit, dist, dist_2, n_threads, n2);
    while (bit-- > 0)
    {
        dist_2 = dist;
        dist = dist >> 1;
        _fft_body(*out, *out, W, bit, dist, dist_2, n_threads, n2);
    }
    bit_reverse(*out, dir, lead, n_threads, n);
    free(W);
}

void _fft_body(cpx *in, cpx *out, cpx *W, int bit, int dist, int dist2, const int n_threads, const int n2)
{
    int n, l, u, p;
    float imag, real;
    cpx tmp;
    /* START OMP CONDITIONAL */
#ifdef _OPENMP
    int *offset, *step, tid, chunk;
    offset = (int *)malloc(sizeof(int) * n_threads);
    step = (int *)malloc(sizeof(int) * n_threads);
    chunk = n2 / n_threads;
#pragma omp parallel private(l, u, p, imag, real, tmp, tid) shared(in, out, W, bit, dist, dist2, step, offset, n_threads, n2)
    {
        tid = omp_get_thread_num();
        offset[tid] = ((tid * chunk) / dist2) * dist2;
        step[tid] = dist + offset[tid];
#pragma omp for schedule(static, chunk)
#else
    int offset, step;
    offset = 0;
    step = dist;
#endif
    /* END OPENMP CONDITIONAL */

    for (n = 0; n < n2; ++n)
    {
        /* START OMP OPS*/
#ifdef _OPENMP
        if (i >= step[tid]) {
            offset[tid] += dist2;
            step[tid] += dist2;
        }
        l = (i & ~(1 << bit)) + offset[tid];
#else
        if (n >= step) {
            offset += dist2;
            step += dist2;
        }
        l = (n & ~(1 << bit)) + offset;
#endif    
        /* END OMP OPS*/
        u = l + dist;
        tmp = in[l];
        p = (u >> bit);
        real = in[u].r * W[p].r - in[u].i * W[p].i;
        imag = in[u].i * W[p].r + in[u].r * W[p].i;
        out[l].r = tmp.r - real;
        out[l].i = tmp.i - imag;
        out[u].r = tmp.r + real;
        out[u].i = tmp.i + imag;
    }
    /* START OMP CLEAN UP */
#ifdef _OPENMP
    }
    free(offset);
    free(step);
#endif
    /* END OMP CLEAN UP */
}

/*
void hulhuala_fft_body(cpx *in, cpx *out, cpx *W, int bit, int dist, int dist2, const int n_threads, const int n2)
{
int i, l, u, p;
float imag, real;
cpx tmp;
#ifdef _OPENMP
int *offset, *step, tid;
offset = (int *)malloc(sizeof(int) * n_threads);
step = (int *)malloc(sizeof(int) * n_threads);
#pragma omp parallel private(l, u, p, imag, real, tmp, tid) shared(in, out, W, bit, dist, dist2, step, offset, n_threads, n2)
{
tid = omp_get_thread_num();
offset[tid] = (((tid * n2) / n_threads) / dist2) * dist2;
step[tid] = dist + offset[tid];
#pragma omp for schedule(static, n2 / n_threads)
#else
int offset, step;
offset = 0;
step = dist;
#endif
for (i = 0; i < n2; ++i)
{
#ifdef _OPENMP
if (i >= step[tid]) {
offset[tid] += dist2;
step[tid] += dist2;
}
l = (i & ~(1 << bit)) + offset[tid];
// offset[tid] += (i >= (dist + offset[tid])) * dist2;
#else
if (i >= step) {
offset += dist2;
step += dist2;
}
l = (i & ~(1 << bit)) + offset;
#endif

u = l + dist;
tmp = in[l];
p = (u >> bit);
real = in[u].r * W[p].r - in[u].i * W[p].i;
imag = in[u].i * W[p].r + in[u].r * W[p].i;
out[l].r = tmp.r - real;
out[l].i = tmp.i - imag;
out[u].r = tmp.r + real;
out[u].i = tmp.i + imag;
}
#ifdef _OPENMP
}
free(offset);
free(step);
#endif
}
*/