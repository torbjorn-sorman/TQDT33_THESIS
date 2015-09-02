
#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_fft_helper.h"
#include "tb_math.h"
#include "tb_print.h"

void twiddle_factors(cpx *W, const int n)
{
    int i, n2, n4;
    float w_ang;
    w_ang = -M_2_PI / n;
    n2 = n / 2;
    n4 = n / 4;
#pragma omp parallel for schedule(static, n2 / omp_get_num_threads()) private(i) shared(W, n2, w_ang)
    for (i = 0; i < n2; ++i) {
        W[i].r = cos(w_ang * i);
        W[i + n2].r = -W[i].r;
    }
#pragma omp parallel for schedule(static, n2 / omp_get_num_threads()) private(i) shared(W, n2, n4)   
    for (i = 0; i < n2; ++i) {
        W[i].i = W[i + n4].r;
        W[i + n2].i = -W[i].i;
    }
}

void twiddle_factors_alt(cpx *W, const int lead, const int n)
{
    int i, n2, n4;
    float w_ang;
    w_ang = -M_2_PI / n;
    n2 = n / 2;
    n4 = n / 4;
#pragma omp parallel for schedule(static, n2 / omp_get_num_threads()) private(i) shared(W, n2, w_ang)   
    for (i = 0; i < n2; ++i) {
        W[i].r = cos(w_ang * i);
        W[i + n2].r = -W[i].r;
    }
#pragma omp parallel for schedule(static, n2 / omp_get_num_threads()) private(i) shared(W, n2, n4)   
    for (i = 0; i < n2; ++i) {
        W[i].i = W[i + n4].r;
        W[i + n2].i = -W[i].i;
    }
    bit_reverse(W, FORWARD_FFT, lead, n);
}

void twiddle_factors_s(cpx *W, const int lead, const int n)
{
    int i;
    float w_ang, a;
    w_ang = -M_2_PI / n;
#pragma omp parallel for schedule(static, n / omp_get_num_threads()) private(i, a) shared(W, lead, n, w_ang)  
    for (i = 0; i < n; ++i) {
        a = (w_ang * i);
        W[i].r = cos(a);
        W[i].i = sin(a);
    }
    bit_reverse(W, FORWARD_FFT, lead, n);
}

void twiddle_factors(cpx *W, const int lead, const int n)
{
    int i, n2, n4, _3n4;    
    float tmp, w_angle;
    w_angle = -M_2_PI / n;
    n2 = n / 2;
    n4 = n / 4;
    _3n4 = n2 + n4;
#pragma omp parallel for schedule(static, n2 / omp_get_num_threads()) private(i, tmp) shared(W, n2, n4, _3n4, n, w_angle)    
    for (i = n4; i < _3n4; ++i) {
        W[i].r = cos(w_angle * i);
        tmp = -W[i].r;
        W[(i + n2) % n].r = tmp;
        W[i - n4].i = W[i].r;
        W[i + n4].i = tmp;
    }
    bit_reverse(W, FORWARD_FFT, lead, n);
}

void twiddle_factors_inverse(cpx *W, const int n)
{
    int i;
#pragma omp parallel for schedule(static, n / omp_get_num_threads()) private(i) shared(W, n)
    for (i = 0; i < n; ++i)
        W[i].i = -W[i].i;
}

void bit_reverse(cpx *x, const double dir, const int lead, const int n)
{
    int i, p;
    cpx tmp_cpx;
#pragma omp parallel for schedule(static, n / omp_get_num_threads()) private(i, p, tmp_cpx) shared(x, lead, n)
    for (i = 0; i <= n; ++i) {
        p = BIT_REVERSE(i, lead);
        if (i < p) {
            tmp_cpx = x[i];
            x[i] = x[p];
            x[p] = tmp_cpx;
        }
    }
    if (dir == INVERSE_FFT) {
#pragma omp parallel for schedule(static, n / omp_get_num_threads()) private(i) shared(x, n)
        for (i = 0; i < n; ++i) {
            x[i].r = x[i].r / (float)n;
            x[i].i = x[i].i / (float)n;
        }
    }
}

void fft_shift(cpx **seq, const int n)
{
    int x, y, n2;
    cpx tmp;
    n2 = n / 2;
#pragma omp parallel for schedule(static, n2 / omp_get_num_threads()) private(x, y, tmp) shared(n2, seq)
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

void fft_shift_alt(cpx **seq, const int n)
{
    int x, y, n2;
    cpx tmp;
    n2 = n / 2;
#pragma omp parallel for schedule(static, n2 / omp_get_num_threads()) private(x, y, tmp) shared(n2, seq)
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