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

int compareComplex(my_complex *c1, my_complex *c2, uint32_t N)
{
    double m = 0.00001;
    double max_r = -DBL_MAX;
    double max_i = -DBL_MAX;
    for (int i = 0; i < N; ++i)
    {
        max_r = max(abs((double)c1[i].r - (double)c2[i].r), max_r);
        max_i = max(abs((double)c1[i].i - (double)c2[i].i), max_i);
    }
    if ((max_r > m || max_i > m))
    {
        //printf("\nNOT EQUAL\nDiff: (%f, %f)\n", max_r, max_i);
        return 0;
    }
    else
    {
        //printf("\nEQUAL\n");
        return 1;
    }
}