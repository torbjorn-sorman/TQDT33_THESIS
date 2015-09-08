#ifndef TB_TEST_H
#define TB_TEST_H

#include "tb_definitions.h"

double simple(const unsigned int limit);

void validate_fft(fft_func fn, const int n_threads, const unsigned int max_elements);
double timing(fft_func fn, const int n_threads, const int n);
double timing(twiddle_func fn, const int n_threads, const int n);

void mtime(char *name, fft_func fn, const int n_threads, int toFile, const unsigned int max_elements);
void mtime(char *name, twiddle_func fn, const int n_threads, int toFile, const unsigned int max_elements);

void test_fft(char *name, fft_func fn, const int n_threads, int toFile, const unsigned int max_elements);
void test_fft2d(char *name, fft2d_func fn, const int n_threads, int file, const unsigned int max_elem);

void test_fftw(unsigned int max_elem);
void test_fftw2d(unsigned int max_elem);

int test_image(fft2d_func fft2d, char *filename, const int n_threads, const int n);

/*
unsigned char test_equal_dft(fft_body_fn fn, fft_body_fn ref, const int inplace, const int n_threads);
unsigned char test_equal_dft2d(fft2d_fn fn2d, fft_body_fn fn, fft_body_fn ref, const int inplace, const int n_threads);
unsigned char test_image(fft2d_fn fn2d, fft_body_fn fn, char *filename, const int n_threads, const int n);
unsigned char test_transpose(transpose_fn fn, const int b, const int n_threads, const int n);
unsigned char test_twiddle(twiddle_fn fn, twiddle_fn ref_fn, const int n_threads, const int n);

double test_time_dft(fft_body_fn fn, const int n_threads, const int n);
double test_time_dft_2d(fft2d_fn fn2d, fft_body_fn fn, const int n_threads, const int n);
double test_time_transpose(transpose_fn fn, const int b, const int n_threads, const int n);
double test_time_twiddle(twiddle_fn fn, const int n_threads, const int n);
double test_time_reverse(bit_reverse_fn fn, const int n_threads, const int n);
double test_time_const_geom(int n_threads, const int n);
double test_cmp_time(fft_body_fn fn, fft_body_fn ref, const int n_threads);

void test_complete_fft(char *name, fft_body_fn fn, const int n_threads);
void test_complete_fft_cg(char *name, const int n_threads);
void test_complete_fft_cg_no_twiddle(char *name, const int n_threads);
void test_complete_ext(char *name, void(*fn)(const double, cpx *, cpx *, const int n_threads, const int), const int n_threads);

void test_complete_fft2d(char *name, fft2d_fn fn, const int n_threads);
*/

/* External libraries to compare with. */
void kiss_fft(double dir, cpx **in, cpx **out, const int n_threads, const int n);
void cgp_fft(double dir, cpx **in, cpx **out, const int n_threads, const int n);

#endif