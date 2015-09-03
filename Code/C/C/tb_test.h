#ifndef TB_TEST_H
#define TB_TEST_H

#include "tb_definitions.h"

double simple(const unsigned int limit);

unsigned char test_equal_dft(dif_fn fn, dif_fn ref, const int inplace, int n_threads);
unsigned char test_equal_dft2d(fft2d_fn fn2d, dif_fn fn, dif_fn ref, const int inplace, int n_threads);
unsigned char test_image(fft2d_fn fn2d, dif_fn fn, char *filename, int n_threads, const int n);
unsigned char test_transpose(transpose_fn fn, const int b, int n_threads, const int n);
unsigned char test_twiddle(twiddle_fn fn, twiddle_fn ref_fn, int n_threads, const int n);

double test_time_dft(dif_fn fn, int n_threads, const int n);
double test_time_dft_2d(fft2d_fn fn2d, dif_fn fn, int n_threads, const int n);
double test_time_transpose(transpose_fn fn, const int b, int n_threads, const int n);
double test_time_twiddle(twiddle_fn fn, int n_threads, const int n);
double test_time_reverse(bit_reverse_fn fn, int n_threads, const int n);
double test_time_const_geom(int n_threads, const int);
double test_cmp_time(dif_fn fn, dif_fn ref, int n_threads);

void test_complete_fft(char *name, dif_fn fn, int n_threads);
void test_complete_fft_cg(char *name, int n_threads);
void test_complete_ext(char *name, void(*fn)(const double, cpx *, cpx *, int n_threads, const int));

void test_complete_fft2d(char *name, fft2d_fn fn, int n_threads);

/* External libraries to compare with. */
void kiss_fft(double dir, cpx *in, cpx *out, int n_threads, const int n);
void cgp_fft(double dir, cpx *in, cpx *out, int n_threads, const int n);

#endif