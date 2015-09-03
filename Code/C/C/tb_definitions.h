#ifndef TB_DEFINITIONS_H
#define TB_DEFINITIONS_H

#include "kiss_fft.h"

typedef kiss_fft_cpx cpx;
typedef void(*dif_fn)(cpx*, cpx*, cpx*, int, int, int, int, const int);
typedef void(*fft_fn)(dif_fn, const double, cpx*, cpx*, cpx*, int, const int);
typedef void(*fft2d_fn)(dif_fn, const double, cpx**, int, const int);
typedef void(*transpose_fn)(cpx**, const int, int, const int);
typedef void(*twiddle_fn)(cpx*, const int, int, const int);
typedef void(*bit_reverse_fn)(cpx*, const double, const int, int, const int);

#define M_2_PI 6.28318530718f
#define M_PI 3.14159265359f

#define FORWARD_FFT -1.0
#define INVERSE_FFT 1.0

#endif