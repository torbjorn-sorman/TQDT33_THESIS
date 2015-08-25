#ifndef TB_FFT_H
#define TB_FFT_H

#include "tb_math.h"

#define FORWARD_FFT 1
#define INVERSE_FFT -1

void tb_fft(int dir, my_complex *x, my_complex *X, uint32_t N);
void tb_fft_inplace(int dir, my_complex *x, my_complex *X, uint32_t N);

void tb_fft2d(int dir, void(*fn)(int, my_complex*, my_complex*, uint32_t), my_complex **seq2d, uint32_t N);
void tb_dft_naive(my_complex *x, my_complex *X, uint32_t N);

void tb_fft_test(int dir, my_complex *x, my_complex *X, uint32_t N);

#endif