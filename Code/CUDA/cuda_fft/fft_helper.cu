#include "fft_helper.cuh"
#include "math.h"
#include <stdio.h>

int log2_32(int value)
{
    value |= value >> 1; value |= value >> 2; value |= value >> 4; value |= value >> 8; value |= value >> 16;
    return tab32[(unsigned int)(value * 0x07C4ACDD) >> 27];
}

void console_print(cpx *seq, const int n)
{
    int i;
    for (i = 0; i < n; ++i)
        printf("%f\t%f\n", seq[i].x, seq[i].y);
}

int cpx_equal(cpx *c1, cpx *c2, const int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        if (c1[i].x != c2[i].x || c1[i].y != c2[i].y)
            return 0;
    }
    return 1;
}

int cpx_equal(cpx **c1, cpx **c2, const int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        if (cpx_equal(c1[i], c2[i], n) == 0)
            return 0;
    }
    return 1;
}

double cpx_diff(cpx a, cpx b)
{
    double re, im;
    re = abs((double)a.x - (double)b.x);
    im = abs((double)a.y - (double)b.y);
    return fmaxf(re, im);
}

double cpx_diff(cpx *a, cpx *b, const int n)
{
    int i;
    double m_diff;
    m_diff = -1;
    for (i = 0; i < n; ++i)
        m_diff = fmaxf(m_diff, cpx_diff(a[i], b[i]));
    return m_diff;
}

double cpx_diff(cpx **a, cpx **b, const int n)
{
    int i;
    double m_diff;
    m_diff = -1;
    for (i = 0; i < n; ++i)
        m_diff = fmaxf(m_diff, cpx_diff(a[i], b[i], n));
    return m_diff;
}

double cpx_avg_diff(cpx *a, cpx *b, const int n)
{
    int i;
    double sum;
    sum = 0.0;
    for (i = 0; i < n; ++i)
        sum += cpx_diff(a[i], b[i]);
    return sum / n;
}

double cpx_avg_diff(cpx **a, cpx **b, const int n)
{
    int i, j;
    double sum;
    sum = 0.0;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            sum += cpx_diff(a[i][j], b[i][j]);
        }
    }
    return sum / (n * n);
}

__global__ void twiddle_factors(cpx *W, const double dir, const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float ang = dir * (M_2_PI / n) * i;
    W[i].x = cosf(ang);
    W[i].y = sinf(ang);
    W[i + n / 4].y = dir * W[i].x;
}

__global__ void twiddle_factors_fast_inverse(cpx *W)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    W[i].y = -W[i].y;
}

__global__ void bit_reverse(cpx *x, const double dir, const int lead, const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int p = bitReverse32(i, lead);
    cpx tmp;
    if (i < p) {
        tmp = x[i];
        x[i] = x[p];
        x[p] = tmp;
    }
    if (dir > 0) {
        x[i].x = x[i].x / (float)n;
        x[i].y = x[i].y / (float)n;
    }
}

__device__ unsigned int bitReverse32(int x, const int l)
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return((x >> 16) | (x << 16)) >> (32 - l);
}
