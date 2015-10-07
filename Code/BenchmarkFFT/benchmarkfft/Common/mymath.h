#ifndef MYMATH_H
#define MYMATH_H

static int tab32[32] = {
    0, 9, 1, 10, 13, 21, 2, 29, 
    11, 14, 16, 18, 22, 25, 3, 30,
    8, 12, 20, 28, 15, 17, 24, 7, 
    19, 27, 23, 6, 26, 5, 4, 31
};

// Integer, base 2 log
static __inline int log2_32(int value)
{
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return tab32[(unsigned int)(value * 0x07C4ACDD) >> 27];
}

static __inline int cmp(const void *in_l, const void *in_h)
{
    double xx = *(double*)in_l, yy = *(double*)in_h;
    if (xx < yy) return -1;
    if (xx > yy) return  1;
    return 0;
}

// Run average on 5 of n results (or n / 2 if less then five)
// Results sorted by ascending order.
static __inline double avg(double m[], int n)
{
    double sum = 0.0;
    int end = ((n < 5) ? ((n / 2) < 1 ? 1 : (n / 2)) : 5);
    qsort(m, n, sizeof(double), cmp);    
    for (int i = 0; i < end; ++i)
        sum += m[i];
    return (sum / (double)end);
}


// Integer power function
static __inline unsigned int power(const unsigned int base, const unsigned int exp)
{
    if (exp == 0)
        return 1;
    unsigned int value = base;
    for (unsigned int i = 1; i < exp; ++i) {
        value *= base;
    }
    return value;
}

// Integer power function with base 2
static __inline unsigned int power2(const unsigned int exp)
{
    return power(2, exp);
}

static __inline void cpxAddSubMul(cpx *out_l, cpx *out_h, cpx in_l, cpx in_h, cpx w)
{
    (*out_l) = cuCaddf(in_l, in_h);
    (*out_h) = cuCmulf(cuCsubf(in_l, in_h), w);
}

#endif