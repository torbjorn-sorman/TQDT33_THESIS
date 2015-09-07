
#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_fft_helper.h"
#include "tb_math.h"
#include "tb_print.h"

void twiddle_factors(cpx *W, fft_direction dir, const int n)
{
    int len, tmp;
    float w_ang;
    w_ang = -M_2_PI / n;
    len = n / 2;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < len; ++i) {
        W[i].r = cos(w_ang * i);
    }
    len = n / 4;
    if (dir == FORWARD_FFT) {
#pragma omp parallel for schedule(static) private(tmp)  
        for (int i = 0; i < len; ++i) {
            tmp = i + len;
            W[i].i = W[tmp].r;
            W[tmp].i = -W[i].r;
        }
    }
    else {
#pragma omp parallel for schedule(static) private(tmp)  
        for (int i = 0; i < len; ++i) {
            tmp = i + len;
            W[i].i = -W[tmp].r;
            W[tmp].i = W[i].r;
        }
    }
}

// Better locality(?)
void twiddle_factors_s(cpx *W, fft_direction dir, const int n)
{
    int n2;
    float w_ang, a;
    w_ang = dir * M_2_PI / n;
    n2 = n / 2;
#pragma omp parallel for schedule(static) private(a)
    for (int i = 0; i < n2; ++i) {
        a = w_ang * i;
        W[i].r = cos(a);
        W[i].i = sin(a);
    }
}

void twiddle_factors_fast_inverse(cpx *W, const int n)
{
#pragma omp parallel for schedule(static) 
    for (int i = 0; i < n; ++i)
        W[i].i = -W[i].i;
}

void bit_reverse(cpx *x, fft_direction dir, const int lead, const int n)
{
    cpx tmp_cpx;
#pragma omp parallel for schedule(static) private(tmp_cpx)
    for (int i = 0; i <= n; ++i) {
        int p = BIT_REVERSE(i, lead);
        if (i < p) {
            tmp_cpx = x[i];
            x[i] = x[p];
            x[p] = tmp_cpx;
        }
    }
    if (dir == INVERSE_FFT) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            x[i].r = x[i].r / (float)n;
            x[i].i = x[i].i / (float)n;
        }
    }
}

void transpose(cpx **seq, const int n)
{
    cpx tmp;
#pragma omp parallel for schedule(static) private(tmp)
    for (int y = 0; y < n; ++y) {
        for (int x = y + 1; x < n; ++x) {
            tmp = seq[y][x];
            seq[y][x] = seq[x][y];
            seq[x][y] = tmp;
        }
    }
}

void transpose_block(cpx **seq, const int b, const int n)
{
    cpx tmp;
#pragma omp parallel for schedule(static) private(tmp)
    for (int bly = 0; bly < n; bly += b) {
        for (int blx = bly; blx < n; blx += b) {
            for (int y = bly; y < b + bly; ++y) {
                for (int x = blx; x < b + blx; ++x) {
                    if (x > y) {
                        tmp = seq[y][x];
                        seq[y][x] = seq[x][y];
                        seq[x][y] = tmp;
                    }
                }
            }
        }
    }
}

void transpose_block2(cpx **seq, const int b, const int n)
{
    cpx tmp;
    for (int bly = 0; bly < n; bly += b) {
#pragma omp parallel for schedule(static) private(tmp)
        for (int blx = bly; blx < n; blx += b) {
            for (int y = bly; y < b + bly; ++y) {
                for (int x = blx; x < b + blx; ++x) {
                    if (x > y) {
                        tmp = seq[y][x];
                        seq[y][x] = seq[x][y];
                        seq[x][y] = tmp;
                    }
                }
            }
        }
    }
}

void transpose_block3(cpx **seq, const int b, const int n)
{
    cpx tmp;
    for (int bly = 0; bly < n; bly += b) {
        for (int blx = bly; blx < n; blx += b) {
#pragma omp parallel for schedule(static) private(tmp)
            for (int y = bly; y < b + bly; ++y) {
                for (int x = blx; x < b + blx; ++x) {
                    if (x > y) {
                        tmp = seq[y][x];
                        seq[y][x] = seq[x][y];
                        seq[x][y] = tmp;
                    }
                }
            }
        }
    }
}

void fft_shift(cpx **seq, const int n)
{
    int n2;
    cpx tmp;
    n2 = n / 2;
#pragma omp parallel for schedule(static) private(tmp)
    for (int y = 0; y < n2; ++y)
    {
        for (int x = 0; x < n2; ++x)
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

void fft_shift_alt(cpx **seq, const int n)
{
    int n2;
    cpx tmp;
    n2 = n / 2;
#pragma omp parallel for schedule(static) private(tmp)
    for (int y = 0; y < n2; ++y)
    {
        for (int x = 0; x < n2; ++x)
        {
            tmp = seq[y][x];
            seq[y][x] = seq[y + n2][x + n2];
            seq[y + n2][x + n2] = tmp;
        }
        for (int x = n2; x < n; ++x)
        {
            tmp = seq[y][x + n2];
            seq[y][x + n2] = seq[y + n2][x];
            seq[y + n2][x] = tmp;
        }
    }
}