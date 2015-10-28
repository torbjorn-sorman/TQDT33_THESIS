#include "omp_fast_index.h"

//
// Testing
//

int openmp_fast_index_validate(const int n)
{
    cpx *in = get_seq(n, 1);
    cpx *out = get_seq(n);
    cpx *ref = get_seq(n, in);
    openmp_fast_index(FFT_FORWARD, &in, &out, n);
    openmp_fast_index(FFT_INVERSE, &out, &in, n);
    double diff = diff_seq(in, ref, n);
    free(in);
    free(out);
    free(ref);
    return diff < RELATIVE_ERROR_MARGIN;
}

int openmp_fast_index_2d_validate(const int n)
{
    cpx **in = get_seq2D(n, 1);
    cpx **ref = get_seq2D(n, in);
    openmp_fast_index_2d(FFT_FORWARD, in, n);
    openmp_fast_index_2d(FFT_INVERSE, in, n);
    double diff = diff_seq(in, ref, n);
    free_seq2D(in, n);
    free_seq2D(ref, n);
    return diff < RELATIVE_ERROR_MARGIN;
}

double openmp_fast_index_performance(const int n)
{
    double measures[NUM_TESTS];
    cpx *in = get_seq(n, 1);
    for (int i = 0; i < NUM_TESTS; ++i) {
        startTimer();
        openmp_fast_index(FFT_FORWARD, &in, &in, n);
        measures[i] = stopTimer();
    }
    free(in);
    return average_best(measures, NUM_TESTS);
}

double openmp_fast_index_2d_performance(const int n)
{
    double measures[NUM_TESTS];
    cpx **in = get_seq2D(n, 1);
    for (int i = 0; i < NUM_TESTS; ++i) {
        startTimer();
        openmp_fast_index_2d(FFT_FORWARD, in, n);
        measures[i] = stopTimer();
    }
    free_seq2D(in, n);
    return average_best(measures, NUM_TESTS);
}

//
// Algorithm
//

static void __inline openmp_fi_body(cpx *in, cpx *out, cpx *W, const unsigned int lmask, int steps, int dist, const int n_half)
{
    const unsigned int pmask = (dist - 1) << steps;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_half; ++i) {
        int l = i + (i & lmask);
        int u = l + dist;        
        c_add_sub_mul(in + l, in + u, out + l, out + u, &W[(i << steps) & pmask]);
    }
}

__inline void ompFI(transform_direction dir, cpx *seq, cpx *W, const int n)
{
    const int n_half = (n / 2);
    int bit = log2_32(n);
    const int leading_bits = 32 - bit;
    int steps = 0;
    int dist = n_half;
    --bit;
    openmp_fi_body(seq, seq, W, 0xFFFFFFFF << bit, steps, dist, n_half);
    while (bit-- > 0) {
        dist >>= 1;
        openmp_fi_body(seq, seq, W, 0xFFFFFFFF << bit, ++steps, dist, n_half);
    }
    openmp_bit_reverse(seq, dir, leading_bits, n);
}

void openmp_fast_index(transform_direction dir, cpx **in, cpx **out, const int n)
{
    const int n_half = (n / 2);
    int bit = log2_32(n);
    const int leading_bits = 32 - bit;
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    openmp_twiddle_factors(W, dir, n);
    int steps = 0;
    int dist = n_half;
    --bit;
    openmp_fi_body(*in, *out, W, 0xFFFFFFFF << bit, steps, dist, n_half);
    while (bit-- > 0) {
        dist >>= 1;
        openmp_fi_body(*out, *out, W, 0xFFFFFFFF << bit, ++steps, dist, n_half);
    }
    openmp_bit_reverse(*out, dir, leading_bits, n);
    free(W);
}

_inline void openmp_fi_rows(transform_direction dir, cpx** seq, cpx *W, const int n)
{
#pragma omp parallel for schedule(static)
    for (int row = 0; row < n; ++row) {
        ompFI(dir, seq[row], W, n);
    }
}

void openmp_fast_index_2d(transform_direction dir, cpx** seq, const int n)
{
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    openmp_twiddle_factors(W, dir, n);
    openmp_fi_rows(dir, seq, W, n);
    openmp_transpose(seq, n);
    openmp_fi_rows(dir, seq, W, n);
    openmp_transpose(seq, n);
    free(W);
}

