#ifndef TB_FFT_H
#define TB_FFT_H

#include "tb_definitions.h"

void tb_fft_old(double dir, tb_cpx *x, tb_cpx *X, uint32_t N);
void tb_fft(double dir, tb_cpx *x, tb_cpx *X, uint32_t N);
void tb_fft_inplace(double dir, tb_cpx *x, tb_cpx *X, uint32_t N);
void tb_fft_real(double dir, tb_cpx *x, tb_cpx *X, uint32_t N);

void tb_fft2d(double dir, fft_function fn, tb_cpx **seq2d, uint32_t N);

void tb_dft_naive(tb_cpx *x, tb_cpx *X, uint32_t N);

#endif