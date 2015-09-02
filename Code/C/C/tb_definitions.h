#ifndef TB_DEFINITIONS_H
#define TB_DEFINITIONS_H

#include "kiss_fft.h"

typedef kiss_fft_cpx cpx;
typedef void(*dif_fn)(cpx *in, cpx *out, cpx *W, int bit, int dist, int dist2, const int n);
typedef void(*fft_fn)(dif_fn dif, const double dir, cpx *in, cpx *out, cpx *W, const int n);
typedef void(*fft2d_fn)(dif_fn dif, const double dir, cpx** seq, const int n);
typedef void(*transpose_fn)(cpx**, const int, const int);
typedef void(*twiddle_fn)(cpx *W, const int lead, const int n);
typedef void(*bit_reverse_fn)(cpx*, const double, const int, const int);

#define M_2_PI 6.28318530718f
#define M_PI 3.14159265359f

#define FORWARD_FFT -1.0
#define INVERSE_FFT 1.0

#endif