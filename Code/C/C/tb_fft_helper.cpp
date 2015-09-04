
#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_fft_helper.h"
#include "tb_math.h"
#include "tb_print.h"

void twiddle_factors(cpx *W, const double dir, const int n_threads, const int n)
{
    int i, n2, n4, chunk;
    float w_ang;
    w_ang = -M_2_PI / n;
    n2 = n / 2;
    n4 = n / 4;
    chunk = n2 / n_threads;
#pragma omp parallel shared(W, n2, w_ang)
    {
#pragma omp for schedule(static, chunk) private(i)
        for (i = 0; i < n2; ++i) {
            W[i].r = cos(w_ang * i);
            W[i + n2].r = -W[i].r;
        }
#pragma omp for schedule(static, chunk) private(i)  
        for (i = 0; i < n2; ++i) {
            W[i].i = -dir * W[i + n4].r;
            W[i + n2].i = -W[i].i;
        }
    }
}

void twiddle_factors(cpx *W, const double dir, const int lead, const int n_threads, const int n)
{
    int i, n2, n4, chunk;
    float w_ang;
    w_ang = -M_2_PI / n;
    n2 = n / 2;
    n4 = n / 4;
    chunk = n2 / n_threads;
#pragma omp parallel shared(W, n2, w_ang)
    {
#pragma omp for schedule(static, chunk) private(i) 
        for (i = 0; i < n2; ++i) {
            W[i].r = cos(w_ang * i);
            W[i + n2].r = -W[i].r;
        }
#pragma omp for schedule(static, chunk) private(i)  
        for (i = 0; i < n2; ++i) {
            W[i].i = -dir * W[i + n4].r;
            W[i + n2].i = -W[i].i;
        }
    }
    bit_reverse(W, FORWARD_FFT, lead, n_threads, n);
}

void twiddle_factors_short(cpx *W, const double dir, const int lead, const int n_threads, const int n)
{
    int i, n2, n4, n8, chunk;
    float w_ang;
    w_ang = -M_2_PI / n;
    n2 = n / 2;
    n4 = n / 4;
    n8 = n / 8;
    chunk = n4 / n_threads;

    /*

    0 - 1
    1 - 3
    2 - 5
    3 - 7

    0 - 0
    1 - 1
    2 - 2
    3 - 3
    4 - (-0)
    5 - (-1)
    6 - (-2)
    7 - (-3)

    *//*
#pragma omp parallel shared(W, n4, n8, w_ang)
    {
#pragma omp for schedule(static, chunk) private(i) 
        for (i = 0; i <= n4; ++i) {            
            W[i].r = cos(w_ang * (i * 2 + 1));
            W[i + n4].r = -W[i].r;
        }        
#pragma omp for schedule(static, chunk) private(i)  
        for (i = 0; i <= n4; ++i) {
            W[i].i = -dir * W[i + n8].r;
            W[i + n4].i = -W[i].i;
        }
    }*/
#pragma omp parallel shared(W, n2, w_ang)
    {
#pragma omp for schedule(static, chunk) private(i) 
        for (i = 0; i < n2; ++i) {
            W[i].r = cos(w_ang * i);
            W[i + n2].r = -W[i].r;
        }
#pragma omp for schedule(static, chunk) private(i)  
        for (i = 0; i < n2; ++i) {
            W[i].i = -dir * W[i + n4].r;
            W[i + n2].i = -W[i].i;
        }
    }
    //bit_reverse(W, FORWARD_FFT, lead + 1, n_threads, (n / 2));
}

void twiddle_factors_s(cpx *W, const double dir, const int lead, const int n_threads, const int n)
{
    int i, chunk;
    float w_ang, a;
    w_ang = dir * M_2_PI / n;
    chunk = n / n_threads;
#pragma omp parallel for schedule(static, chunk) private(i, a) shared(W, lead, n, w_ang)  
    for (i = 0; i < n; ++i) {
        a = (w_ang * i);
        W[i].r = cos(a);
        W[i].i = sin(a);
    }
    bit_reverse(W, FORWARD_FFT, lead, n_threads, n);
}

void twiddle_factors_inverse(cpx *W, int n_threads, const int n)
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
#pragma omp parallel shared(x, lead, n)
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