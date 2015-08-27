#ifndef TB_TEST_H
#define TB_TEST_H

#include "tb_definitions.h"

void simple();

unsigned char test_equal_dft(fft_function fn, fft_function ref_fn, const int inplace);
unsigned char test_equal_dft2d(fft_function fn, fft_function ref_fn, const int inplace);
unsigned char test_image(fft_function fn, char *filename, const int n);
unsigned char test_transpose(transpose_function fn, const int b, const int n);

double test_time_dft(fft_function fn, const int n);
double test_time_dft_2d(fft_function fn, const int n);
double test_time_2d(const int openMP);

double test_cmp_time(fft_function fn, fft_function ref);

double test_time_transpose(transpose_function fn, const int b, const int n);

int run_fbtest(fft_function fn, const int n);
int run_fft2dinvtest(fft_function fn, const int n);
int run_fft2dTest(fft_function fn, const int n);

/* External libraries to compare with. */

void kiss_fft(double dir, tb_cpx *in, tb_cpx *out, const int n);

#endif