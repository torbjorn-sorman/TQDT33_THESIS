#ifndef TB_FFT_H
#define TB_FFT_H

#include "tb_definitions.h"

void fft_template(PARAMS_FFT);

void tb_fft2d(PARAMS_FFT2D);
void tb_fft2d_omp(PARAMS_FFT2D);

void fft_body(PARAMS_BUTTERFLY);
void fft_body_omp(PARAMS_BUTTERFLY);

void fft_body_alt1(PARAMS_BUTTERFLY);
void fft_body_alt1_omp(PARAMS_BUTTERFLY);

void fft_body_alt2(PARAMS_BUTTERFLY);
void fft_body_alt2_omp(PARAMS_BUTTERFLY);

void fft_const_geom(const double dir, cpx **in, cpx **out, cpx *W, const int n);
void fft_const_geom_omp(const double dir, cpx **in, cpx **out, cpx *W, const int n);

void fft_body_const_geom(const double dir, cpx *in, cpx *out, cpx *W, int bit, unsigned int mask, const int n);
void fft_body_const_geom_omp(cpx *in, cpx *out, cpx *W, int bit, const int n);

#endif