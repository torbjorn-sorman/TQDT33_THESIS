#ifndef TB_TEST_H
#define TB_TEST_H

#ifdef _OPENMP
#include <omp.h> 
#endif

#include <Windows.h>
#include <limits>
#include "fftw-3.3.4-dll32\fftw3.h"
#include "cgp_fft.h"

#include "tb_definitions.h"
#include "tb_fft_helper.h"
#include "tb_image.h"
#include "tb_math.h"
#include "tb_print.h"
#include "tb_filter.h"
#include "fft_generated_fixed_const.h"

double simple(const unsigned int limit);

void validate_fft(fft_func fn, const int n_threads, const unsigned int max_elements);
double timing(fft_func fn, const int n_threads, const int n);
double timing(twiddle_func fn, const int n_threads, const int n);

void mtime(char *name, fft_func fn, const int n_threads, int toFile, const unsigned int max_elements);
void mtime(char *name, twiddle_func fn, const int n_threads, int toFile, const unsigned int max_elements);

void test_fft(char *name, fft_func fn, const int n_threads, int toFile, const unsigned int max_elements);
void test_fft2d(char *name, fft2d_func fn, const int n_threads, int file, const unsigned int max_elem);

void test_short_fft(fft_func fn, const int n_threads, unsigned int max_elem);
void test_short_fftw(unsigned int max_elem);

void test_fftw(unsigned int max_elem);
void test_fftw2d(unsigned int max_elem);

int test_image(fft2d_func fft2d, char *filename, const int n_threads, const int n);

/* External libraries to compare with. */
void kiss_fft(double dir, cpx **in, cpx **out, const int n_threads, const int n);
void cgp_fft(double dir, cpx **in, cpx **out, const int n_threads, const int n);

#define ERROR_MARGIN 0.0001

static _inline int checkError(cpx *seq, cpx *ref, const int n, int print)
{
    int j;
    double re, im, i_val, r_val;
    re = im = 0.0;
    for (j = 0; j < n; ++j) {
        r_val = abs(seq[j].r - ref[j].r);
        i_val = abs(seq[j].i - ref[j].i);
        re = re > r_val ? re : r_val;
        im = im > i_val ? im : i_val;
    }
    if (print == 1) printf("Error\tre(e): %f\t im(e): %f\t@%u\n", re, im, n);
    return re > ERROR_MARGIN || im > ERROR_MARGIN;
}

static _inline int checkError(cpx **seq, cpx **ref, const int n, int print)
{
    int x, y;
    double re, im, i_val, r_val;
    re = im = 0.0;
    for (y = 0; y < n; ++y) {
        for (x = 0; x < n; ++x) {
            r_val = abs(seq[y][x].r - ref[y][x].r);
            i_val = abs(seq[y][x].i - ref[y][x].i);
            re = re > r_val ? re : r_val;
            im = im > i_val ? im : i_val;
        }
    }
    if (print == 1) printf("Error\tre(e): %f\t im(e): %f\t@%u\n", re, im, n);
    return re > ERROR_MARGIN || im > ERROR_MARGIN;
}

static _inline int equal(cpx a, cpx b)
{
    return a.r == b.r && a.i == b.i;
}

static _inline int cmp(const void *x, const void *y)
{
    double xx = *(double*)x, yy = *(double*)y;
    if (xx < yy) return -1;
    if (xx > yy) return  1;
    return 0;
}

static _inline int cmp(unsigned char *a, unsigned char *b, const int n)
{
    int i, end;
    end = n * n * 3;
    for (i = 0; i < end; ++i)
        if (abs(a[i] - b[i]) > 1)
            return 0;
    return 1;
}

static _inline double avg(double m[], int n)
{
    int i, cnt, end;
    double sum;
    qsort(m, n, sizeof(double), cmp);
    sum = 0.0;
    cnt = 0;
    end = n < 5 ? n - 1 : 5;
    for (i = 0; i < end; ++i) {
        sum += m[i];
        ++cnt;
    }
    return (sum / (double)cnt);
}

static _inline double abs_diff(cpx a, cpx b)
{
    double r, i;
    r = a.r - b.r;
    i = a.i - b.i;
    return sqrt(r * r + i * i);
}

static _inline cpx *get_seq(const int n, const int sinus)
{
    int i;
    cpx *seq;
    seq = (cpx *)malloc(sizeof(cpx) * n);
    for (i = 0; i < n; ++i) {
        seq[i].r = sinus == 0 ? 0.f : (float)sin(M_2_PI * (((double)i) / n));
        seq[i].i = 0.f;
    }
    return seq;
}

static _inline cpx *get_seq(const int n, cpx *src)
{
    int i;
    cpx *seq;
    seq = (cpx *)malloc(sizeof(cpx) * n);
    for (i = 0; i < n; ++i) {
        seq[i].r = src[i].r;
        seq[i].i = src[i].i;
    }
    return seq;
}

static _inline cpx *get_seq(const int n)
{
    return get_seq(n, 0);
}

static _inline cpx **get_seq2d(const int n, const int type)
{
    int i, x, y;
    cpx **seq;
    seq = (cpx **)malloc(sizeof(cpx *) * n);
    if (type == 0 || type == 1) {
        for (i = 0; i < n; ++i) {
            seq[i] = get_seq(n, type);
        }
    }
    else {
        for (y = 0; y < n; ++y) {
            seq[y] = (cpx *)malloc(sizeof(cpx) * n);
            for (x = 0; x < n; ++x) {
                seq[y][x] = { (float)x, (float)y };
            }
        }
    }
    return seq;
}

static _inline cpx **get_seq2d(const int n, cpx **src)
{
    int i;
    cpx **seq;
    seq = (cpx **)malloc(sizeof(cpx *) * n);
    for (i = 0; i < n; ++i) {
        seq[i] = get_seq(n, src[i]);
    }
    return seq;
}

static _inline cpx **get_seq2d(const int n)
{
    return get_seq2d(n, 0);
}

static _inline void copy_seq2d(cpx **from, cpx **to, const int n)
{
    int i, j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            to[i][j].r = from[i][j].r;
            to[i][j].i = from[i][j].i;
        }
    }
}

static _inline unsigned char *get_empty_img(const int w, const int h)
{
    return (unsigned char *)malloc(sizeof(unsigned char) * w * h * 3);
}

static _inline void free_seq2d(cpx **seq, const int n)
{
    int i;
    for (i = 0; i < n; ++i)
        free(seq[i]);
    free(seq);
}

#endif