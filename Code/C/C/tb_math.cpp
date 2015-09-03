#include <limits>
#include <cmath>
#include <Windows.h>

#include "tb_math.h"

__inline int log2_32(int value)
{
    value |= value >> 1; value |= value >> 2; value |= value >> 4; value |= value >> 8; value |= value >> 16;
    return tab32[(unsigned int)(value * 0x07C4ACDD) >> 27];
}

__inline unsigned int reverseBitsLowMem(int x, const int l)
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return((x >> 16) | (x << 16)) >> (32 - l);
}

__inline int cpx_equal(cpx *c1, cpx *c2, const int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        if (c1[i].r != c2[i].r || c1[i].i != c2[i].i)
            return 0;
    }
    return 1;
}

__inline int cpx_equal(cpx **c1, cpx **c2, const int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        if (cpx_equal(c1[i], c2[i], n) == 0)
            return 0;
    }
    return 1;
}

__inline double cpx_diff(cpx a, cpx b)
{
    double re, im;
    re = abs((double)a.r - (double)b.r);
    im = abs((double)a.i - (double)b.i);
    return max(re, im);
}

__inline double cpx_diff(cpx *a, cpx *b, const int n)
{
    int i;
    double m_diff;
    m_diff = DBL_MIN;
    for (i = 0; i < n; ++i)
        m_diff = max(m_diff, cpx_diff(a[i], b[i]));
    return m_diff;
}

__inline double cpx_diff(cpx **a, cpx **b, const int n)
{
    int i;
    double m_diff;
    m_diff = DBL_MIN;
    for (i = 0; i < n; ++i)
        m_diff = max(m_diff, cpx_diff(a[i], b[i], n));
    return m_diff;
}

__inline double cpx_avg_diff(cpx *a, cpx *b, const int n)
{
    int i;
    double sum;
    sum = 0.0;
    for (i = 0; i < n; ++i)
        sum += cpx_diff(a[i], b[i]);
    return sum / n;
}

__inline double cpx_avg_diff(cpx **a, cpx **b, const int n)
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