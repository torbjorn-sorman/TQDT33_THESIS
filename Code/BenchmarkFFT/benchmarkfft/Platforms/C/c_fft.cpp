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
    cpx *in, *buf, *ref;
    setup_seq2D(&in, &buf, &ref, n);
    cConstantGeometry2D(FFT_FORWARD, in, buf, n);
    cConstantGeometry2D(FFT_INVERSE, in, buf, n);
    double diff = diff_seq(in, ref, n);
    free(in);
    free(buf);
    free(ref);
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
    cpx *in = get_seq(n);
    cpx *out = get_seq(n);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        cConstantGeometry2D(FFT_FORWARD, in, out, n);
        measures[i] = stopTimer();
    }
    free(in);
    free(out);
    return avg(measures, NUM_PERFORMANCE);
}

//
// Algorithm
//

_inline void cCGBody(cpx *in, cpx *out, const cpx *W, const unsigned int mask, const int n)
{
    for (int i = 0; i < n; i += 2) {
        int l = i / 2;
        cpxAddSubMulCG(&in[l], &in[(n / 2) + l], out, W[l & mask]);
        out += 2;
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

_inline void cCG(fftDir dir, cpx **in, cpx **out, int offset, const cpx *W, const int n)
{
    int depth = log2_32(n);
    int steps = 0;
    cCGBody(*in + offset, *out + offset, W, 0xffffffff << steps, n);
    while (++steps < depth) {
        swap(in, out);
        cCGBody(*in + offset, *out + offset, W, 0xffffffff << steps, n);
    }
    bit_reverse(*out, dir, 32 - depth, n);
}

void cConstantGeometry2D(fftDir dir, cpx* in, cpx* out, const int n)
{
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);
    //for (int row = 0; row < n * n; row += n)
        //cCG(dir, &in, &out, row, W, n);
    transpose(out, n);
    //for (int row = 0; row < n * n; row += n)
        //cCG(dir, &in, &out, row, W, n);
    transpose(in, n);
    free(W);
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