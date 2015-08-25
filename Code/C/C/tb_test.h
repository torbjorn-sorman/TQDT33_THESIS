#ifndef TB_TEST_H
#define TB_TEST_H

#include "tb_math.h"

#define FFT_FUNCTION(A) void(*A)(int, my_complex*, my_complex*, uint32_t)

// TODO: Clean up and make relevant tests...

unsigned char test_equal_dft(FFT_FUNCTION(fn), FFT_FUNCTION(ref_fn), uint32_t N, uint32_t inplace);

double test_time_dft(FFT_FUNCTION(fn), uint32_t N);
double test_time_dft_2d(FFT_FUNCTION(fn), uint32_t N);

unsigned char test_image(FFT_FUNCTION(fn), char *filename, uint32_t N);

int run_fbtest(FFT_FUNCTION(fn), uint32_t N);
int run_fft2dinvtest(FFT_FUNCTION(fn), uint32_t N);

int run_fft2dTest(FFT_FUNCTION(fn), uint32_t N);

/* External libraries to compare with. */

void kiss_fft(int dir, my_complex *in, my_complex *out, uint32_t N);

#endif