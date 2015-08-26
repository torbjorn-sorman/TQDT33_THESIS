#ifndef TB_FFT_H
#define TB_FFT_H

#include "tb_math.h"

#define FORWARD_FFT -1.0
#define INVERSE_FFT 1.0

typedef void(*fft_function)(int, tb_cpx*, tb_cpx*, uint32_t);

void tb_fft(int dir, tb_cpx *x, tb_cpx *X, uint32_t N);
void tb_fft_inplace(int dir, tb_cpx *x, tb_cpx *X, uint32_t N);
void tb_fft_real(int dir, tb_cpx *x, tb_cpx *X, uint32_t N);

void tb_fft2d(int dir, void(*fn)(int, tb_cpx*, tb_cpx*, uint32_t), tb_cpx **seq2d, uint32_t N);
void tb_fft2d_trans(int dir, void(*fn)(int, tb_cpx*, tb_cpx*, uint32_t), tb_cpx **seq2d, uint32_t N);
void tb_fft2d_inplace(int dir, void(*fn)(int, tb_cpx*, tb_cpx*, uint32_t), tb_cpx **seq2d, uint32_t N);

void tb_dft_naive(tb_cpx *x, tb_cpx *X, uint32_t N);

#endif