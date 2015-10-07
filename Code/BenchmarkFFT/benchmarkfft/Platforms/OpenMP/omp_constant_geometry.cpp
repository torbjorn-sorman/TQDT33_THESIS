#include "omp_constant_geometry.h"

//
// Testing
//

int ompConstantGeometry_validate(const int n)
{
    cpx *in = get_seq(n, 1);
    cpx *out = get_seq(n);
    cpx *ref = get_seq(n, in);
   ompConstantGeometry(FFT_FORWARD, &in, &out, n);
   ompConstantGeometry(FFT_INVERSE, &out, &in, n);
    double diff = diff_seq(in, ref, n);
    free(in);
    free(out);
    free(ref);
    return diff < RELATIVE_ERROR_MARGIN;
}

int ompConstantGeometry2D_validate(const int n)
{
    cpx **in = get_seq2D(n, 1);
    cpx **ref = get_seq2D(n, in);
   ompConstantGeometry2D(FFT_FORWARD, in, n);
   ompConstantGeometry2D(FFT_INVERSE, in, n);
    double diff = diff_seq(in, ref, n);
    free_seq2D(in, n);
    free_seq2D(ref, n);
    return diff < RELATIVE_ERROR_MARGIN;
}

double ompConstantGeometry_runPerformance(const int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in = get_seq(n, 1);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
       ompConstantGeometry(FFT_FORWARD, &in, &in, n);
        measures[i] = stopTimer();
    }
    free(in);
    return avg(measures, NUM_PERFORMANCE);
}

double ompConstantGeometry2D_runPerformance(const int n)
{
    double measures[NUM_PERFORMANCE];
    cpx **in = get_seq2D(n, 1);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
       ompConstantGeometry2D(FFT_FORWARD, in, n);
        measures[i] = stopTimer();
    }
    free_seq2D(in, n);
    return avg(measures, NUM_PERFORMANCE);
}

//
// Algorithm
//

_inline void _fft_cgbody(cpx *in, cpx *out, const cpx *W, unsigned int mask, const int n)
{
    int const n2 = n / 2;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i += 2) {
        int l = i / 2;
        cpxAddSubMul(&out[i], &out[i + 1], in[l], in[n2 + l], W[l & mask]);
    }
}

void ompConstantGeometry(fftDir dir, cpx **in, cpx **out, const int n)
{
    int depth = log2_32(n);
    int steps = 0;
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);

    _fft_cgbody(*in, *out, W, 0xffffffff << steps, n);
    while (++steps < depth) {
        swap(in, out);
        _fft_cgbody(*in, *out, W, 0xffffffff << steps, n);
    }

    bit_reverse(*out, dir, 32 - depth, n);
    free(W);
}

_inline void _fft_const_geom(fftDir dir, cpx **in, cpx **out, const cpx *W, const int n)
{
    int depth = log2_32(n);
    int steps = 0;
    _fft_cgbody(*in, *out, W, 0xffffffff << steps, n);
    while (++steps < depth) {
        swap(in, out);
        _fft_cgbody(*in, *out, W, 0xffffffff << steps, n);
    }
    bit_reverse(*out, dir, 32 - depth, n);
}

_inline void _do_rows(fftDir dir, cpx** seq, const cpx *W, cpx **buffers, const int n)
{
#pragma omp parallel for schedule(static)
    for (int row = 0; row < n; ++row) {
        int tid = omp_get_thread_num();
        _fft_const_geom(dir, &seq[row], &buffers[tid], W, n);
        swap(&seq[row], &buffers[tid]);
    }
}

void ompConstantGeometry2D(fftDir dir, cpx** seq, const int n)
{
    int n_threads = omp_get_num_threads();
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);
    cpx** buffers = (cpx **)malloc(sizeof(cpx *) * n_threads);
#pragma omp for schedule(static)
    for (int i = 0; i < n_threads; ++i) {
        buffers[i] = (cpx *)malloc(sizeof(cpx) * n);
    }
    _do_rows(dir, seq, W, buffers, n);
    transpose(seq, n);
    _do_rows(dir, seq, W, buffers, n);
    transpose(seq, n);
#pragma omp for schedule(static)
    for (int i = 0; i < n_threads; ++i) {
        free(buffers[i]);
    }
    free(buffers);
    free(W);
}

_inline void _fft_cgbody(cpx *in, cpx *out, float w_angle, unsigned int mask, const int n)
{   
    cpx w;
    const int n2 = n / 2;
    int old = -1;
#pragma omp parallel for schedule(static) private(old, w)
    for (int i = 0; i < n; i += 2) {
        int l = i / 2;
        int p = l & mask;
        if (old != p) {
            float ang = p * w_angle;
            w = make_cuFloatComplex(cos(ang = p * w_angle), sin(ang));
            old = p;
        }
        cpxAddSubMul(&out[i], &out[i + 1], in[l], in[n2 + l], w);
    }
}

void ompConstantGeometryAlternate(fftDir dir, cpx **in, cpx **out, const int n)
{
    int bit = log2_32(n);
    const int lead = 32 - bit;
    int steps = --bit;
    unsigned int mask = 0xffffffff << (steps - bit);
    float w_angle = dir * M_2_PI / n;
    _fft_cgbody(*in, *out, w_angle, mask, n);
    while (bit-- > 0) {
        swap(in, out);
        mask = 0xffffffff << (steps - bit);
        _fft_cgbody(*in, *out, w_angle, mask, n);
    }
    bit_reverse(*out, dir, lead, n);
}