
#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_fft_helper.h"
#include "tb_math.h"
#include "tb_print.h"

void twiddle_factors(cpx *W, const double dir, const int n_threads, const int n)
{
    int i, n2, n4, chunk, chunk2, tmp;
    float w_ang;
    w_ang = -M_2_PI / n;
    n2 = n / 2;
    n4 = n / 4;
    chunk = n2 / n_threads;
    chunk2 = n4 / n_threads;
#pragma omp parallel shared(W, dir, n2, n4, w_ang, chunk, chunk2)
    {
#pragma omp for schedule(static, chunk) private(i)
        for (i = 0; i < n2; ++i) {
            W[i].r = cos(w_ang * i);
        }
#pragma omp barrier
        if (dir == FORWARD_FFT) {
#pragma omp for schedule(static, chunk2) private(i, tmp)  
            for (i = 0; i < n4; ++i) {
                tmp = i + n4;
                W[i].i = W[tmp].r;
                W[tmp].i = -W[i].r;
            }
        }
        else {
#pragma omp for schedule(static, chunk2) private(i, tmp)  
            for (i = 0; i < n4; ++i) {
                tmp = i + n4;
                W[i].i = -W[tmp].r;
                W[tmp].i = W[i].r;
            }
        }
    }
}

/* To be replaced */
void twiddle_factors(cpx *W, const double dir, const int lead, const int n_threads, const int n)
{
    int i, n2, n4, chunk;
    float w_ang;
    w_ang = -M_2_PI / n;
    n2 = n / 2;
    n4 = n / 4;
    chunk = n2 / n_threads;
#pragma omp parallel shared(W, n2, n4, w_ang)
    {
#pragma omp for schedule(static, chunk) private(i) 
        for (i = 0; i < n2; ++i) {
            W[i].r = cos(w_ang * i);
            W[i + n2].r = -W[i].r;
        }
#pragma omp barrier
#pragma omp for schedule(static, chunk) private(i)  
        for (i = 0; i < n4; ++i) {
            W[i].i = -dir * W[i + n4].r;
            W[i + n2].i = -W[i].i;
        }
    }
    bit_reverse(W, FORWARD_FFT, lead, n_threads, n);
}

// Better locality
void twiddle_factors_s(cpx *W, const double dir, const int n_threads, const int n)
{
    int i, chunk, n2;
    float w_ang, a;
    w_ang = dir * M_2_PI / n;
    chunk = n / n_threads;
    n2 = n / 2;
#pragma omp parallel for schedule(static, chunk) private(i, a) shared(W, n, n2, w_ang)  
    for (i = 0; i < n2; ++i) {
        a = w_ang * i;
        W[i].r = cos(a);
        W[i].i = sin(a);
    }
}

void twiddle_factors_fast_inverse(cpx *W, int n_threads, const int n)
{
    int i, chunk;
    chunk = n / n_threads;
#pragma omp parallel for schedule(static, chunk) private(i) shared(W, n)
    for (i = 0; i < n; ++i)
        W[i].i = -W[i].i;
}

void bit_reverse(cpx *x, const double dir, const int lead, const int n_threads, const int n)
{
    int i, p, chunk;
    cpx tmp_cpx;
    chunk = n / n_threads;
#pragma omp parallel
    {
#pragma omp for schedule(static, chunk) private(i, p, tmp_cpx)
        for (i = 0; i <= n; ++i) {
            p = BIT_REVERSE(i, lead);
            if (i < p) {
                tmp_cpx = x[i];
                x[i] = x[p];
                x[p] = tmp_cpx;
            }
        }
        if (dir == INVERSE_FFT) {
#pragma omp barrier
#pragma omp for schedule(static, chunk) private(i)
            for (i = 0; i < n; ++i) {
                x[i].r = x[i].r / (float)n;
                x[i].i = x[i].i / (float)n;
            }
        }
    }
}

void fft_shift(cpx **seq, const int n_threads, const int n)
{
    int x, y, n2, chunk;
    cpx tmp;
    n2 = n / 2;
    chunk = n2 / n_threads;
#pragma omp parallel for schedule(static, chunk) private(x, y, tmp) shared(n2, seq)
    for (y = 0; y < n2; ++y)
    {
        for (x = 0; x < n2; ++x)
        {
            tmp = seq[y][x];
            seq[y][x] = seq[y + n2][x + n2];
            seq[y + n2][x + n2] = tmp;
            tmp = seq[y][x + n2];
            seq[y][x + n2] = seq[y + n2][x];
            seq[y + n2][x] = tmp;
        }
    }
}

void fft_shift_alt(cpx **seq, const int n_threads, const int n)
{
    int x, y, n2, chunk;
    cpx tmp;
    n2 = n / 2;
    chunk = n2 / n_threads;
#pragma omp parallel for schedule(static, chunk) private(x, y, tmp) shared(n2, seq)
    for (y = 0; y < n2; ++y)
    {
        for (x = 0; x < n2; ++x)
        {
            tmp = seq[y][x];
            seq[y][x] = seq[y + n2][x + n2];
            seq[y + n2][x + n2] = tmp;
        }
        for (x = n2; x < n; ++x)
        {
            tmp = seq[y][x + n2];
            seq[y][x + n2] = seq[y + n2][x];
            seq[y + n2][x] = tmp;
        }
    }
}