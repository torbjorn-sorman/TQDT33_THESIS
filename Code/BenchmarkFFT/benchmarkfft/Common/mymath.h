#ifndef MYMATH_H
#define MYMATH_H

static int tab32[32] = {
    0, 9, 1, 10, 13, 21, 2, 29, 
    11, 14, 16, 18, 22, 25, 3, 30,
    8, 12, 20, 28, 15, 17, 24, 7, 
    19, 27, 23, 6, 26, 5, 4, 31
};

static __inline int log2_32(int value)
{
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return tab32[(unsigned int)(value * 0x07C4ACDD) >> 27];
}

#endif