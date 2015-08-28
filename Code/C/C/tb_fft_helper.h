#ifndef TB_FFT_HELPER_H
#define TB_FFT_HELPER_H

#include "tb_definitions.h"

void twiddle_factors(tb_cpx *W, const double dir, const int lead, const int n);
void twiddle_factors_omp(tb_cpx *W, const double dir, const int lead, const int n);
void bit_reverse(tb_cpx *x, const double dir, const int n, const int lead);
void bit_reverse_omp(tb_cpx *X, const double dir, const int n, const int lead);

#endif