#ifndef FFT_RADIX4_H
#define FFT_RADIX4_H

#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_definitions.h"
#include "tb_math.h"
#include "tb_fft_helper.h"
#include "tb_print.h"

void fft_radix4(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n);

#endif