#ifndef TB_DEFINITIONS_H
#define TB_DEFINITIONS_H

#include "kiss_fft.h"

typedef kiss_fft_cpx tb_cpx;
typedef void(*fft_function)(const double, tb_cpx*, tb_cpx*, tb_cpx*, const int);
typedef void(*fft2d_function)(const double, fft_function, tb_cpx**, const int);
typedef void(*transpose_function)(tb_cpx**, const int, const int);
typedef void(*twiddle_function)(tb_cpx*, const int, const int);

#define M_2_PI 6.28318530718
#define M_PI 3.14159265359

#define FORWARD_FFT -1.0
#define INVERSE_FFT 1.0

#endif