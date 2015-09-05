#ifndef TB_DEFINITIONS_H
#define TB_DEFINITIONS_H

#include "cuComplex.h"

typedef cuFloatComplex cpx;

typedef void(*fft_func)(const double direction, cpx **in, cpx **out, const int n);
typedef void(*fft2d_func)(fft_func func, const double direction, cpx **seq, const int n);

typedef void(*transpose_func)(cpx **seq, const int, const int n);
typedef void(*twiddle_func)(cpx *W, const int, const int n);
typedef void(*bit_reverse_func)(cpx *seq, const double dir, const int lead, const int n);

#define M_2_PI 6.28318530718f
#define M_PI 3.14159265359f

#define FORWARD_FFT -1.0f
#define INVERSE_FFT 1.0f

#endif