#ifndef TB_TEST_H
#define TB_TEST_H

#include "tb_definitions.h"

unsigned char test_equal_dft(fft_function fn, fft_function ref_fn, uint32_t inplace);
unsigned char test_equal_dft2d(fft_function fn, fft_function ref_fn, uint32_t inplace);
unsigned char test_image(fft_function fn, char *filename, uint32_t N);
unsigned char test_transpose(uint32_t N);

double test_time_dft(fft_function fn, uint32_t N);
double test_time_dft_2d(fft_function fn, uint32_t N);

double test_cmp_time(fft_function fn, fft_function ref);

double test_time_transpose(void(*transpose_function)(tb_cpx**, uint32_t), uint32_t N);
double test_time_transpose_block(void(*transpose_function)(tb_cpx**, uint32_t, uint32_t), uint32_t block_size, uint32_t N);

int run_fbtest(fft_function fn, uint32_t N);
int run_fft2dinvtest(fft_function fn, uint32_t N);
int run_fft2dTest(fft_function fn, uint32_t N);


/* External libraries to compare with. */

void kiss_fft(double dir, tb_cpx *in, tb_cpx *out, uint32_t N);

#endif