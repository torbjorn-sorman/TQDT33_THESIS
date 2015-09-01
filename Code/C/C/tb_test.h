#ifndef TB_TEST_H
#define TB_TEST_H

#include "tb_definitions.h"

unsigned char test_equal_dft(dif_fn fn, dif_fn ref, const int inplace);
unsigned char test_equal_dft2d(fft2d_fn fn2d, dif_fn fn, dif_fn ref, const int inplace);
unsigned char test_image(fft2d_fn fn2d, dif_fn fn, char *filename, const int n);
unsigned char test_transpose(transpose_fn fn, const int b, const int n);
unsigned char test_twiddle(twiddle_fn fn, twiddle_fn ref_fn, const int n);

double test_time_dft(dif_fn fn, const int n);
double test_time_dft_2d(fft2d_fn fn2d, dif_fn fn, const int n);
double test_time_transpose(transpose_fn fn, const int b, const int n);
double test_time_twiddle(twiddle_fn fn, const int n);
double test_time_reverse(bit_reverse_fn fn, const int n);
double test_cmp_time(dif_fn fn, dif_fn ref);

void test_complete_fft(char *name, dif_fn fn);
void test_complete_fft2d(char *name, fft2d_fn fn);


/* External libraries to compare with. */
void kiss_fft(double dir, cpx *in, cpx *out, const int n);

#endif