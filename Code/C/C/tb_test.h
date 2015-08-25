#ifndef TB_TEST_H
#define TB_TEST_H

#include "tb_math.h"

typedef void(*fft_function)(int, tb_cpx*, tb_cpx*, uint32_t);

unsigned char test_equal_dft(fft_function fn, fft_function ref_fn, uint32_t N, uint32_t inplace);
unsigned char test_image(fft_function fn, char *filename, uint32_t N);

double test_time_dft(fft_function fn, uint32_t N);
double test_time_dft_2d(fft_function fn, uint32_t N);



int run_fbtest(fft_function fn, uint32_t N);
int run_fft2dinvtest(fft_function fn, uint32_t N);
int run_fft2dTest(fft_function fn, uint32_t N);

/* External libraries to compare with. */

void kiss_fft(int dir, tb_cpx *in, tb_cpx *out, uint32_t N);

#endif