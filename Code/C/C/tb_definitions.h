#ifndef TB_DEFINITIONS_H
#define TB_DEFINITIONS_H

#include "kiss_fft.h"

typedef unsigned __int32 uint32_t;
typedef kiss_fft_cpx tb_cpx;
typedef void(*fft_function)(double, tb_cpx*, tb_cpx*, uint32_t);

#define M_2_PI 6.28318530718
#define M_PI 3.14159265359

#define FORWARD_FFT -1.0
#define INVERSE_FFT 1.0

#endif