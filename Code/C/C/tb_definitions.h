#ifndef TB_DEFINITIONS_H
#define TB_DEFINITIONS_H

#include "kiss_fft.h"

typedef kiss_fft_cpx cpx;
typedef const float fft_direction;

typedef void(*fft_func)(const float direction, cpx **in, cpx **out, int n_threads, const int n);
typedef void(*fft2d_func)(fft_direction direction, cpx **seq, int n_threads, const int n);

typedef void(*transpose_func)(cpx **seq, const int, const int n);
typedef void(*twiddle_func)(cpx *W, const float dir, const int n);
typedef void(*bit_reverse_func)(cpx *seq, const float dir, const int lead, const int n);

#define M_2_PI 6.28318530718f
#define M_PI 3.14159265359f

#define FORWARD_FFT -1.0f
#define INVERSE_FFT 1.0f

#endif