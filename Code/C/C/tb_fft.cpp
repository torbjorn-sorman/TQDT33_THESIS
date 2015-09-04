#include "tb_fft.h"

#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_math.h"
#include "tb_transpose.h"
#include "tb_print.h"
#include "tb_fft_helper.h"
/*
 void fft_template(fft_body_fn dif, const double dir, cpx *in, cpx *out, cpx *W, const int n_threads, const int n)
{
    int bit, dist, dist2, lead;
    bit = log2_32(n);
    dist2 = n;
    dist = (n / 2);
    lead = 32 - bit;
    --bit;
    dif(in, out, W, bit, dist, dist2, n_threads, n);
    while (bit-- > 0)
    {
        dist2 = dist;
        dist = dist >> 1;
        dif(out, out, W, bit, dist, dist2, n_threads, n);
    }
    bit_reverse(out, dir, lead, n_threads, n);
}

void tb_fft2d(fft_body_fn dif, const double dir, cpx** seq, const int n_threads, const int n)
{
    int row, col, chunk;
    cpx tmp, *W;
    W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, 32 - log2_32(n), n_threads, n);
    if (dir == INVERSE_FFT)
        twiddle_factors_inverse(W, n_threads, n);
    chunk = n / n_threads;
#pragma omp parallel shared(W, dif, dir, seq, n, chunk)
    {
#pragma omp for schedule(static, chunk) private(row)
        for (row = 0; row < n; ++row) {
            fft_template(dif, dir, seq[row], seq[row], W, n_threads, n);
        }

#pragma omp for schedule(static, chunk) private(row, col, tmp)
        for (row = 0; row < n; ++row) {
            for (col = row + 1; col < n; ++col) {
                tmp = seq[row][col];
                seq[row][col] = seq[col][row];
                seq[col][row] = tmp;
            }
        }

#pragma omp for schedule(static, chunk) private(row)
        for (row = 0; row < n; ++row) {
            fft_template(dif, dir, seq[row], seq[row], W, n_threads, n);
        }

#pragma omp for schedule(static, chunk) private(row, col, tmp)
        for (row = 0; row < n; ++row) {
            for (col = row + 1; col < n; ++col) {
                tmp = seq[row][col];
                seq[row][col] = seq[col][row];
                seq[col][row] = tmp;
            }
        }
    }
    if (W != NULL)
        free(W);
}

 void fft_body(cpx *in, cpx *out, cpx *W, int bit, int dist, int dist2, const int n_threads, const int n)
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
#endif
}

 void fft_body_alt1(cpx *in, cpx *out, cpx *W, int bit, int dist, int dist2, const int n_threads, const int n)
{
    int i, l, u, p, n2;
    float imag, real;
    cpx tmp;
    n2 = n / 2;
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
    step = 0;
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

// Must be supplied with two buffers
 void fft_const_geom(const double dir, cpx **in, cpx **out, cpx *W, const int n_threads, const int n)
{
    int bit, steps;
    unsigned int mask;
    cpx *tmp;
    bit = log2_32(n);
    const int lead = 32 - bit;
    steps = --bit;
    mask = 0xffffffff << (steps - bit);
    fft_body_const_geom(*in, *out, W, mask, n_threads, n);
    while (bit-- > 0)
    {
        tmp = *in;
        *in = *out;
        *out = tmp;
        mask = 0xffffffff << (steps - bit);
        fft_body_const_geom(*in, *out, W, mask, n_threads, n);
    }
    bit_reverse(*out, dir, lead, n_threads, n);
}

 void fft_body_const_geom(cpx *in, cpx *out, cpx *W, unsigned int mask, const int n_threads, const int n)
{
    int i, l, u, p, n2;
    cpx tmp;
    n2 = n / 2;
#pragma omp parallel for schedule(static, n2 / n_threads) private(i, l, u, p, tmp) shared(in, out, W, mask, n, n2)
    for (i = 0; i < n; i += 2) {
        l = i / 2;
        u = n2 + l;
        p = l & mask;

        tmp.r = in[l].r - in[u].r;
        tmp.i = in[l].i - in[u].i;

        out[i].r = in[l].r + in[u].r;
        out[i].i = in[l].i + in[u].i;
        out[i + 1].r = (W[p].r * tmp.r) - (W[p].i * tmp.i);
        out[i + 1].i = (W[p].i * tmp.r) + (W[p].r * tmp.i);
    }
}

// Must be supplied with two buffers
 void fft_const_geom(const double dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    int bit, steps;
    unsigned int mask;
    float w_angle;
    cpx *tmp;

    bit = log2_32(n);
    const int lead = 32 - bit;
    steps = --bit;
    mask = 0xffffffff << (steps - bit);
    w_angle = (float)dir * M_2_PI / n;
    fft_body_const_geom(*in, *out, w_angle, mask, n_threads, n);
    while (bit-- > 0)
    {
        tmp = *in;
        *in = *out;
        *out = tmp;
        mask = 0xffffffff << (steps - bit);
        fft_body_const_geom(*in, *out, w_angle, mask, n_threads, n);
    }
    bit_reverse(*out, dir, lead, n_threads, n);
}

// If old == p evaluates to true the first round, that call will not calculate the correct values.
#pragma warning(disable:4700) 
 void fft_body_const_geom(cpx *in, cpx *out, float w_angle, unsigned int mask, const int n_threads, const int n)
{
    int i, l, u, p, n2, old;
    float cv, sv;
    cpx tmp;
    n2 = n / 2;
    old = -1;
#pragma omp parallel for schedule(static, n2 / n_threads) private(i, l, u, p, tmp, cv, sv, old) shared(in, out, w_angle, mask, n, n2)
    for (i = 0; i < n; i += 2) {
        l = i / 2;
        u = n2 + l;
        p = l & mask;
        if (old != p) {
            cv = cos(p * w_angle);
            sv = sin(p * w_angle);
            old = p;
        }
        tmp.r = in[l].r - in[u].r;
        tmp.i = in[l].i - in[u].i;

        out[i].r = in[l].r + in[u].r;
        out[i].i = in[l].i + in[u].i;

        out[i + 1].r = (cv * tmp.r) - (sv * tmp.i);
        out[i + 1].i = (sv * tmp.r) + (cv * tmp.i);
    }
#pragma warning(default:4700)
}
*/