#include "tb_fft.h"

#include <omp.h>

#include "tb_math.h"
#include "tb_transpose.h"
#include "tb_print.h"
#include "tb_fft_helper.h"

void fft_template(PARAMS_FFT)
{
    int bit, dist, dist2, lead;
    bit = log2_32(n);
    dist2 = n;
    dist = (n / 2);
    lead = 32 - bit;
    --bit;
    console_print(in, n);
    dif(in, out, W, bit, dist, dist2, n);
    console_print(in, n);
    while (bit-- > 0)
    {
        dist2 = dist;
        dist = dist >> 1;
        dif(out, out, W, bit, dist, dist2, n);
    }
    bit_reverse(out, dir, lead, n);
}

void tb_fft2d(PARAMS_FFT2D)
{
    int row;
    const int block_size = n < 16 ? n : n / 2;
    cpx *W;
    W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, 32 - log2_32(n), n);
    if (dir == INVERSE_FFT)
        twiddle_factors_inverse_omp(W, n);

    for (row = 0; row < n; ++row)
        fft_template(dif, dir, seq[row], seq[row], W, n);
    transpose_block(seq, block_size, n);

    for (row = 0; row < n; ++row)
        fft_template(dif, dir, seq[row], seq[row], W, n);
    transpose_block(seq, block_size, n);

    if (W != NULL)
        free(W);
}

void tb_fft2d_omp(PARAMS_FFT2D)
{
    int row, col;
    cpx tmp, *W;
    W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors_omp(W, 32 - log2_32(n), n);
    if (dir == INVERSE_FFT)
        twiddle_factors_inverse_omp(W, n);

#pragma omp parallel for private(row) shared(W, dif, dir, seq, n)
    for (row = 0; row < n; ++row) {
        fft_template(dif, dir, seq[row], seq[row], W, n);
    }
#pragma omp parallel for private(row, col, tmp) shared(n, seq)
    for (row = 0; row < n; ++row) {
        for (col = row + 1; col < n; ++col) {
            tmp = seq[row][col];
            seq[row][col] = seq[col][row];
            seq[col][row] = tmp;
        }
    }
#pragma omp parallel for private(row) shared(W, dif, dir, seq, n)
    for (row = 0; row < n; ++row) {
        fft_template(dif, dir, seq[row], seq[row], W, n);
    }
#pragma omp parallel for private(row, col, tmp) shared(n, seq)
    for (row = 0; row < n; ++row) {
        for (col = row + 1; col < n; ++col) {
            tmp = seq[row][col];
            seq[row][col] = seq[col][row];
            seq[col][row] = tmp;
        }
    }
    if (W != NULL)
        free(W);
}

void fft_body(PARAMS_BUTTERFLY)
{
    int start, end, l, u, p;
    float imag, real;
    cpx tmp;
    for (start = 0; start < n; start += dist2) {
        end = dist + start;
        for (l = start; l < end; ++l) {
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
    }
}

void fft_body_omp(PARAMS_BUTTERFLY)
{
    int start, end, l, u, p;
    float imag, real;
    cpx tmp;
#pragma omp parallel for private(start, end, l, u, p, imag, real, tmp) shared(in, out, W, bit, dist, dist2, n)
    for (start = 0; start < n; start += dist2) {
        end = dist + start;
        for (l = start; l < end; ++l) {
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
    }
}

void fft_body_alt1(PARAMS_BUTTERFLY)
{
    int i, l, u, p, offset, step, n2;
    float imag, real;
    cpx tmp;
    offset = 0;
    step = dist;
    n2 = n / 2;
    for (i = 0; i < n2; ++i)
    {
        if (i >= step) {
            offset += dist2;
            step += dist2;
        }
        l = (i & ~(1 << bit)) + offset;
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
}

void fft_body_alt1_omp(PARAMS_BUTTERFLY)
{
    int i, l, u, p, *offset, *step, threads, tid, n2;
    float imag, real;
    cpx tmp;
    n2 = n / 2;
    threads = omp_get_max_threads();
    offset = (int *)malloc(sizeof(int) * threads);
    step = (int *)malloc(sizeof(int) * threads);
#pragma omp parallel private(l, u, p, imag, real, tmp, tid) shared(in, out, W, bit, dist, dist2, step, offset, threads, n2)
    {
        tid = omp_get_thread_num();
        offset[tid] = (((tid * n2) / threads) / dist2) * dist2;
        step[tid] = dist + offset[tid];
#pragma omp for
        for (i = 0; i < n2; ++i)
        {
            if (i >= step[tid]) {
                offset[tid] += dist2;
                step[tid] += dist2;
            }
            l = (i & ~(1 << bit)) + offset[tid];
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
    }
    free(offset);
    free(step);
}

void fft_body_alt2(PARAMS_BUTTERFLY)
{
    int i, l, u, p, offset, n2;
    float imag, real;
    cpx tmp;
    n2 = n / 2;
    offset = 0;
    for (i = 0; i < n2; ++i)
    {
        offset += (i >= (dist + offset)) * dist2;
        l = (i & ~(1 << bit)) + offset;
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
}

void fft_body_alt2_omp(PARAMS_BUTTERFLY)
{
    int i, l, u, p, *offset, threads, tid, n2;
    float imag, real;
    cpx tmp;
    n2 = (n / 2);
    threads = omp_get_max_threads();
    offset = (int *)malloc(sizeof(int) * threads);
#pragma omp parallel private(l, u, p, imag, real, tmp, tid) shared(in, out, W, bit, dist, dist2, offset, threads, n2)
    {
        tid = omp_get_thread_num();
        offset[tid] = (((tid * n2) / threads) / dist2) * dist2;
#pragma omp for
        for (i = 0; i < n2; ++i)
        {
            offset[tid] += (i >= (dist + offset[tid])) * dist2;
            l = (i & ~(1 << bit)) + offset[tid];
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
    }
    free(offset);
}

void fft_body_const_geom(PARAMS_BUTTERFLY)
{
    int i, start, end, l, u, p, n2;
    float imag, real;
    cpx tmp;
    n2 = n / 2;
    console_print(in, n);
    for (i = 0; i < n; ++i){        
        l = i / 2;
        u = n2 + l;
        p = (l >> bit);
        /*
        //cpx add
        out[i].r = in[l].r + in[u].r;
        out[i].i = in[l].i + in[u].i;
        // cpx sub
        out[i + 1].r = in[l].r - in[u].r;
        out[i + 1].i = in[l].i - in[u].i;
        // cpx mul
        out[i + 1].r = W[p].r * in[u].r - W[p].i * in[u].i;
        out[i + 1].i = W[p].r * in[u].i + W[p].i * in[u].r;
        */

        //printf("(%f, %f) och (%f, %f)\n", in[l].r, in[l].i, in[u].r, in[u].i);
        printf("%d: %d och %d, %d\n", i, l, u, p);

        tmp = in[l];
        real = in[u].r * W[p].r - in[u].i * W[p].i;
        imag = in[u].i * W[p].r + in[u].r * W[p].i;
        printf("re: %f im: %f\n", real, imag);
        out[i].r = tmp.r - real;
        out[i].i = tmp.i - imag;
        out[++i].r = tmp.r + real;
        out[i].i = tmp.i + imag;
    }
}