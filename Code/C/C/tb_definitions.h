#ifndef TB_DEFINITIONS_H
#define TB_DEFINITIONS_H

#include "kiss_fft.h"

#define PARAMS_FFT dif_fn dif, const double dir, cpx *in, cpx *out, cpx *W, const int n
#define PARAMS_FFT2D dif_fn dif, const double dir, cpx** seq, const int n
#define PARAMS_BUTTERFLY cpx *in, cpx *out, cpx *W, int bit, int dist, int dist2, const int n
#define PARAMS_TWIDDLE cpx *W, const int lead, const int n

typedef kiss_fft_cpx cpx;
typedef void(*dif_fn)(PARAMS_BUTTERFLY);
typedef void(*fft_fn)(PARAMS_FFT);
typedef void(*fft2d_fn)(PARAMS_FFT2D);
typedef void(*transpose_fn)(cpx**, const int, const int);
typedef void(*twiddle_fn)(PARAMS_TWIDDLE);
typedef void(*bit_reverse_fn)(cpx*, const double, const int, const int);

#define M_2_PI 6.28318530718f
#define M_PI 3.14159265359f

#define FORWARD_FFT -1.0
#define INVERSE_FFT 1.0

#endif