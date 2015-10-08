#include "c_fft.h"

//
// Testing
//

//#define NO_TWIDDLE_TABLE

int cConstantGeometry_validate(const int n)
{
    cpx *in = get_seq(n, 1);
    cpx *out = get_seq(n);
    cpx *ref = get_seq(n, in);
#ifdef NO_TWIDDLE_TABLE
    cConstantGeometryAlternate(FFT_FORWARD, &in, &out, n);
    cConstantGeometryAlternate(FFT_INVERSE, &out, &in, n);
#else
    cConstantGeometry(FFT_FORWARD, &in, &out, n);
    cConstantGeometry(FFT_INVERSE, &out, &in, n);
#endif
    double diff = diff_seq(in, ref, n);
    free(in);
    free(out);
    free(ref);
    return diff < RELATIVE_ERROR_MARGIN;
}

int cConstantGeometry2D_validate(const int n)
{
    cpx **in = get_seq2D(n, 1);
    cpx **ref = get_seq2D(n, in);
    cConstantGeometry2D(FFT_FORWARD, in, n);
    cConstantGeometry2D(FFT_INVERSE, in, n);
    double diff = diff_seq(in, ref, n);
    free_seq2D(in, n);
    free_seq2D(ref, n);
    return diff < RELATIVE_ERROR_MARGIN;
}

double cConstantGeometry_runPerformance(const int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in = get_seq(n, 1);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
#ifdef NO_TWIDDLE_TABLE
        cConstantGeometryAlternate(FFT_FORWARD, &in, &in, n);
#else
        cConstantGeometry(FFT_FORWARD, &in, &in, n);
#endif
        measures[i] = stopTimer();
    }
    free(in);
    return avg(measures, NUM_PERFORMANCE);
}

double cConstantGeometry2D_runPerformance(const int n)
{
    double measures[NUM_PERFORMANCE];
    cpx **in = get_seq2D(n, 1);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        cConstantGeometry2D(FFT_FORWARD, in, n);
        measures[i] = stopTimer();
    }
    free_seq2D(in, n);
    return avg(measures, NUM_PERFORMANCE);
}

//
// Algorithm
//

_inline void cCGBody(cpx *in, cpx *out, const cpx *W, const unsigned int mask, const int n)
{
    for (int i = 0; i < n; i += 2) {
        int l = i / 2;
        cpxAddSubMul(in, l, (n / 2) + l, out++, out++, W[l & mask]);
    }
}

void cConstantGeometry(fftDir dir, cpx **in, cpx **out, const int n)
{
    int depth = log2_32(n);
    int steps = 0;
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);

    cCGBody(*in, *out, W, 0xffffffff << steps, n);
    while (++steps < depth) {
        swap(in, out);
        cCGBody(*in, *out, W, 0xffffffff << steps, n);
    }

    bit_reverse(*out, dir, 32 - depth, n);
    free(W);
}

_inline void cCG(fftDir dir, cpx **in, cpx **out, const cpx *W, const int n)
{
    int depth = log2_32(n);
    int steps = 0;
    cCGBody(*in, *out, W, 0xffffffff << steps, n);
    while (++steps < depth) {
        swap(in, out);
        cCGBody(*in, *out, W, 0xffffffff << steps, n);
    }
    bit_reverse(*out, dir, 32 - depth, n);
}

static void _inline cCGDoRows(fftDir dir, cpx** seq, const cpx *W, cpx *buffer, const int n)
{
    for (int row = 0; row < n; ++row) {
        cCG(dir, &seq[row], &buffer, W, n);
        swap(&seq[row], &buffer);
    }
}

void cConstantGeometry2D(fftDir dir, cpx** seq, const int n)
{
    cpx* buffer = (cpx *)malloc(sizeof(cpx) * n);
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);
    cCGDoRows(dir, seq, W, buffer, n);
    transpose(seq, n);
    cCGDoRows(dir, seq, W, buffer, n);
    transpose(seq, n);
    free(W);
    free(buffer);
}

_inline void cCGBody(cpx *in, cpx *out, float w_angle, unsigned int mask, const int n)
{
    cpx w;
    int old = -1;
    for (int i = 0; i < n; i += 2) {
        int l = i / 2;
        int p = l & mask;
        if (old != p) {
            float ang = p * w_angle;
            w = make_cuFloatComplex(cos(ang), sin(ang));
            old = p;
        }
        cpxAddSubMul(in, l, (n / 2) + l, out++, out++, w);
    }
}

void cConstantGeometryAlternate(fftDir dir, cpx **in, cpx **out, const int n)
{
    int depth = log2_32(n);
    int steps = 0;
    float w_angle = dir * M_2_PI / n;
    cCGBody(*in, *out, w_angle, 0xffffffff << steps, n);
    while (steps++ < depth) {
        swap(in, out);
        cCGBody(*in, *out, w_angle, 0xffffffff << steps, n);
    }
    bit_reverse(*out, dir, 32 - depth, n);
}