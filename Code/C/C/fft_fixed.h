#ifndef FFT_FIXED_H
#define FFT_FIXED_H

#include <amp_math.h>

#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_definitions.h"
#include "tb_math.h"
#include "tb_fft_helper.h"
#include "tb_print.h"

#include "fft_generated_fixed.h"
#include "fft_generated_fixed_const.h"
#include "fft_generated_fixed_const_w.h"

void fft_fixed(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n);

#endif