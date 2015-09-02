#ifndef TB_FFT_H
#define TB_FFT_H

#include "tb_definitions.h"

void fft_template(dif_fn dif, const double dir, cpx *in, cpx *out, cpx *W, const int n);

void tb_fft2d(dif_fn dif, const double dir, cpx** seq, const int n);

void fft_body(cpx *in, cpx *out, cpx *W, int bit, int dist, int dist2, const int n);
void fft_body_alt1(cpx *in, cpx *out, cpx *W, int bit, int dist, int dist2, const int n);
void fft_const_geom(const double dir, cpx **in, cpx **out, cpx *W, const int n);
void fft_body_const_geom(cpx*, cpx*, cpx*, unsigned int, const int);

#endif