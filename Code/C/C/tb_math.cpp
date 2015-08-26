#include <limits>
#include <cmath>
#include <Windows.h>

#include "tb_math.h"

int log2_32(uint32_t value)
{
    value |= value >> 1; value |= value >> 2; value |= value >> 4; value |= value >> 8; value |= value >> 16;
    return tab32[(uint32_t)(value * 0x07C4ACDD) >> 27];
}

uint32_t reverseBitsLowMem(uint32_t x, uint32_t l)
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return((x >> 16) | (x << 16)) >> (32 - l);
}

int cpx_equal(tb_cpx *c1, tb_cpx *c2, uint32_t N)
{
    uint32_t i;
    for (i = 0; i < N; ++i) {
        if (c1[i].r != c2[i].r || c1[i].i != c2[i].i)
            return 0;
    }
    return 1;
}

int cpx_equal(tb_cpx **c1, tb_cpx **c2, uint32_t N)
{
    uint32_t i;
    for (i = 0; i < N; ++i) {
        if (cpx_equal(c1[i], c2[i], N) == 0)
            return 0;
    }
    return 1;
}

double cpx_diff(tb_cpx a, tb_cpx b)
{
    double re, im;
    re = abs((double)a.r - (double)b.r);
    im = abs((double)a.i - (double)b.i);
    return max(re, im);
}

double cpx_diff(tb_cpx *a, tb_cpx *b, uint32_t N)
{
    uint32_t i;
    double m_diff;
    m_diff = DBL_MIN;
    for (i = 0; i < N; ++i)
        m_diff = max(m_diff, cpx_diff(a[i], b[i]));
    return m_diff;
}

double cpx_diff(tb_cpx **a, tb_cpx **b, uint32_t N)
{
    uint32_t i;
    double m_diff;
    m_diff = DBL_MIN;
    for (i = 0; i < N; ++i)
        m_diff = max(m_diff, cpx_diff(a[i], b[i], N));
    return m_diff;
}

double cpx_avg_diff(tb_cpx *a, tb_cpx *b, uint32_t N)
{
    uint32_t i;
    double sum;
    sum = 0.0;
    for (i = 0; i < N; ++i)
        sum += cpx_diff(a[i], b[i]);
    return sum / N;
}

double cpx_avg_diff(tb_cpx **a, tb_cpx **b, uint32_t N)
{
    uint32_t i, j;
    double sum;
    sum = 0.0;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            sum += cpx_diff(a[i][j], b[i][j]);
        }
    }
    return sum / (N * N);
}