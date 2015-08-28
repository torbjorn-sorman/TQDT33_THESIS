#ifndef TB_FFT_H
#define TB_FFT_H

#include "tb_definitions.h"

void tb_fft(const double dir, tb_cpx *x, tb_cpx *X, const int n);
void tb_fft_alt(const double dir, tb_cpx *x, tb_cpx *X, const int n);
void tb_fft_inplace(const double dir, tb_cpx *x, tb_cpx *X, const int n);
void tb_fft_real(const double dir, tb_cpx *x, tb_cpx *X, const int n);
void tb_fft_openmp(const double dir, tb_cpx *x, tb_cpx *X, const int n);

void tb_fft2d(const double dir, fft_function fn, tb_cpx **seq2d, const int n);
void tb_fft2d_openmp(const double dir, fft_function fn, tb_cpx **seq2d, const int n);
void tb_fft2d_openmp_alt(const double dir, fft_function fn, tb_cpx **seq2d, const int n);

#endif