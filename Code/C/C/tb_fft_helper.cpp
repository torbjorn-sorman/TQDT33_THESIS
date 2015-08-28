
#include "tb_fft_helper.h"
#include "tb_math.h"

void twiddle_factors(tb_cpx *W, const double dir, const int lead, const int n)
{
    int i, n2, n4;
    double ang, w_angle;
    w_angle = (dir * M_2_PI) / (double)n;
    n2 = n / 2;
    n4 = n / 4;
    for (i = 0; i < n2; ++i) {
        ang = w_angle * i;
        W[i].r = (float)cos(ang);
        W[i + n2].r = -W[i].r;
    }
    for (i = 0; i < n2; ++i) {
        W[i].i = W[i + n4].r;
        W[i + n2].i = -W[i].i;
    }
    bit_reverse(W, dir, n, lead);
}

void twiddle_factors_omp(tb_cpx *W, const double dir, const int lead, const int n)
{
    int i, n2, n4;
    double ang, w_angle;
    w_angle = (dir * M_2_PI) / (double)n;
    n2 = n / 2;
    n4 = n / 4;
#pragma omp parallel for schedule(static) private(i, ang) shared(W, n2, w_angle)
    for (i = 0; i < n2; ++i) {
        ang = w_angle * i;
        W[i].r = (float)cos(ang);
        W[i + n2].r = -W[i].r;
    }
#pragma omp parallel for schedule(static) private(i) shared(W, n2, n4)
    for (i = 0; i < n2; ++i) {
        W[i].i = W[i + n4].r;
        W[i + n2].i = -W[i].i;
    }
    bit_reverse_omp(W, dir, n, lead);
}

void bit_reverse(tb_cpx *x, const double dir, const int n, const int lead)
{
    int i, p;
    tb_cpx tmp_cpx;
    for (i = 0; i <= n; ++i) {
        p = reverseBits(i, lead);
        if (i < p) {
            tmp_cpx = x[i];
            x[i] = x[p];
            x[p] = tmp_cpx;
        }
    }
    if (dir == INVERSE_FFT) {
        for (i = 0; i < n; ++i) {
            x[i].r = x[i].r / (float)n;
            x[i].i = x[i].i / (float)n;
        }
    }
}

void bit_reverse_omp(tb_cpx *X, const double dir, const int n, const int lead)
{
    int i, p;
    tb_cpx tmp_cpx;
#pragma omp parallel for private(i, p, tmp_cpx) shared(lead, X, n)
    for (i = 0; i <= n; ++i) {
        p = reverseBits(i, lead);
        if (i < p) {
            tmp_cpx = X[i];
            X[i] = X[p];
            X[p] = tmp_cpx;
        }
    }
    if (dir == INVERSE_FFT) {
#pragma omp parallel for private(i) shared(X, n)
        for (i = 0; i < n; ++i) {
            X[i].r = X[i].r / (float)n;
            X[i].i = X[i].i / (float)n;
        }
    }
}