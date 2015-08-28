#ifndef TB_TEST_H
#define TB_TEST_H

#include "tb_definitions.h"

void simple();

unsigned char test_equal_dft(fft_function fn, fft_function ref_fn, const int inplace);
unsigned char test_equal_dft2d(fft2d_function fn2d, fft_function fn, fft_function ref_fn, const int inplace);
unsigned char test_image(fft2d_function fn2d, fft_function fn, char *filename, const int n);
unsigned char test_transpose(transpose_function fn, const int b, const int n);
unsigned char test_twiddle(twiddle_function fn, twiddle_function ref_fn, const int n);

double test_time_dft(fft_function fn, const int n);
double test_time_dft_2d(fft2d_function fn2d, fft_function fn, const int n);
double test_time_transpose(transpose_function fn, const int b, const int n);
double test_time_twiddle(twiddle_function fn, const int n);
double test_cmp_time(fft_function fn, fft_function ref);


/* External libraries to compare with. */
void kiss_fft(double dir, tb_cpx *in, tb_cpx *out, const int n);

#endif