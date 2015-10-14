#include "omp_fast_index.h"

//
// Testing
//

int openmp_fast_index_validate(const int n)
{
    cpx *in = get_seq(n, 1);
    cpx *out = get_seq(n);
    cpx *ref = get_seq(n, in);
    ompFastIndex(FFT_FORWARD, &in, &out, n);
    ompFastIndex(FFT_INVERSE, &out, &in, n);
    double diff = diff_seq(in, ref, n);
    free(in);
    free(out);
    free(ref);
    return diff < RELATIVE_ERROR_MARGIN;
}

int ompFastIndex2D_validate(const int n)
{
    cpx **in = get_seq2D(n, 1);
    cpx **ref = get_seq2D(n, in);
    ompFastIndex2D(FFT_FORWARD, in, n);
    ompFastIndex2D(FFT_INVERSE, in, n);
    double diff = diff_seq(in, ref, n);
    free_seq2D(in, n);
    free_seq2D(ref, n);
    return diff < RELATIVE_ERROR_MARGIN;
}

double ompFastIndex_runPerformance(const int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in = get_seq(n, 1);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        ompFastIndex(FFT_FORWARD, &in, &in, n);
        measures[i] = stopTimer();
    }
    free(in);
    return average_best(measures, NUM_PERFORMANCE);
}

double ompFastIndex2D_runPerformance(const int n)
{
    double measures[NUM_PERFORMANCE];
    cpx **in = get_seq2D(n, 1);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        ompFastIndex2D(FFT_FORWARD, in, n);
        measures[i] = stopTimer();
    }
    free_seq2D(in, n);
    return average_best(measures, NUM_PERFORMANCE);
}

//
// Algorithm
//

static void __inline ompFIBody(cpx *in, cpx *out, cpx *W, const unsigned int lmask, int steps, int dist, const int n_half)
{
    const unsigned int pmask = (dist - 1) << steps;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_half; ++i) {
        int l = i + (i & lmask);
        int u = l + dist;        
        c_add_sub_mul(in + l, in + u, out + l, out + u, &W[(i << steps) & pmask]);
    }
}

__inline void ompFI(fftDir dir, cpx *seq, cpx *W, const int n)
{
    const int n_half = (n / 2);
    int bit = log2_32(n);
    const int leading_bits = 32 - bit;
    int steps = 0;
    int dist = n_half;
    --bit;
    ompFIBody(seq, seq, W, 0xFFFFFFFF << bit, steps, dist, n_half);
    while (bit-- > 0) {
        dist >>= 1;
        ompFIBody(seq, seq, W, 0xFFFFFFFF << bit, ++steps, dist, n_half);
    }
    openmp_bit_reverse(seq, dir, leading_bits, n);
}

void ompFastIndex(fftDir dir, cpx **in, cpx **out, const int n)
{
    const int n_half = (n / 2);
    int bit = log2_32(n);
    const int leading_bits = 32 - bit;
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    ompTwiddleFactors(W, dir, n);
    int steps = 0;
    int dist = n_half;
    --bit;
    ompFIBody(*in, *out, W, 0xFFFFFFFF << bit, steps, dist, n_half);
    while (bit-- > 0) {
        dist >>= 1;
        ompFIBody(*out, *out, W, 0xFFFFFFFF << bit, ++steps, dist, n_half);
    }
    openmp_bit_reverse(*out, dir, leading_bits, n);
    free(W);
}

_inline void ompFIDoRows(fftDir dir, cpx** seq, cpx *W, const int n)
{
#pragma omp parallel for schedule(static)
    for (int row = 0; row < n; ++row) {
        ompFI(dir, seq[row], W, n);
    }
}

void ompFastIndex2D(fftDir dir, cpx** seq, const int n)
{
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    ompTwiddleFactors(W, dir, n);
    ompFIDoRows(dir, seq, W, n);
    ompTranspose(seq, n);
    ompFIDoRows(dir, seq, W, n);
    ompTranspose(seq, n);
    free(W);
}

