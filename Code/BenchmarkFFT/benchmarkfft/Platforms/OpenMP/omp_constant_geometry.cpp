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
    //double diffForward = (abs(in[1].y) - abs(in[n - 1].y)) / (n / 2);    
    ompConstantGeometry(FFT_INVERSE, &out, &in, n);
    double diff = diff_seq(in, ref, n);
    free(in);
    free(out);
    free(ref);
    return diff < RELATIVE_ERROR_MARGIN;// && diffForward < RELATIVE_ERROR_MARGIN;
}

int ompConstantGeometry2D_validate(const int n)
{
    cpx *in, *buf, *ref;
    setup_seq2D(&in, &buf, &ref, n);

    ompConstantGeometry2D(FFT_FORWARD, &in, &buf, n);
    write_normalized_image("OpenMP", "freq", in, n, true);
    ompConstantGeometry2D(FFT_INVERSE, &in, &buf, n);
    write_image("OpenMP", "spat", in, n);

    double diff = diff_seq(in, ref, n);
    free(in);
    free(buf);
    free(ref);
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
    cpx *in = get_seq(n * n);
    cpx *buf = get_seq(n * n);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        ompConstantGeometry2D(FFT_FORWARD, &in, &buf, n);
        measures[i] = stopTimer();
    }
    free(in);
    free(buf);
    return avg(measures, NUM_PERFORMANCE);
}

//
// Algorithm
//

_inline void ompCGBody(cpx *in, cpx *out, const cpx *W, const unsigned int mask, const int n2)
{
#pragma omp parallel for schedule(static)
    for (int l = 0; l < n2; ++l)
        cpxAddSubMulCG(in + l, in + n2 + l, out + (l << 1), W + (l & mask));
}

void ompConstantGeometry(fftDir dir, cpx **in, cpx **out, const int n)
{
    const int n2 = n / 2;
    int depth = log2_32(n);
    int steps = 0;
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    ompTwiddleFactors(W, dir, n);
    ompCGBody(*in, *out, W, 0xffffffff << steps, n2);
    while (++steps < depth) {
        swapBuffer(in, out);
        ompCGBody(*in, *out, W, 0xffffffff << steps, n2);
    }
    ompBitReverse(*out, dir, 32 - depth, n);
    free(W);
}

_inline void ompCG(fftDir dir, cpx *in, cpx *out, const cpx *W, const int n)
{
    const int n2 = n / 2;
    int depth = log2_32(n);
    int steps = 0;
    ompCGBody(in, out, W, 0xffffffff << steps, n2);
    while (++steps < depth) {
        swapBuffer(&in, &out);
        ompCGBody(in, out, W, 0xffffffff << steps, n2);
    }
    ompBitReverse(out, dir, 32 - depth, n);
}

_inline void ompCGDoRows(fftDir dir, cpx **in, cpx **out, const cpx *W, const int n)
{
#pragma omp parallel for schedule(static)
    for (int row = 0; row < n * n; row += n)
        ompCG(dir, (*in) + row, (*out) + row, W, n);
    if (log2_32(n) % 2 == 0)
        swapBuffer(in, out);
}

void ompConstantGeometry2D(fftDir dir, cpx **in, cpx **out, const int n)
{
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    ompTwiddleFactors(W, dir, n);    
    ompCGDoRows(dir, in, out, W, n);
    ompTranspose(*out, *in, n);
    ompCGDoRows(dir, in, out, W, n);
    ompTranspose(*out, *in, n);
    swapBuffer(in, out);
    free(W);
}

_inline void ompCGBody(cpx *in, cpx *outG, float w_angle, unsigned int mask, const int n)
{    
    const int n2 = n / 2;    
#pragma omp parallel
    {
        int old = -1;        
        cpx w, *out = &outG[omp_get_thread_num() * n / omp_get_num_threads()];
#pragma omp for schedule(static) private(w, old) // might remove this private decl.?
    for (int i = 0; i < n; i += 2) {
        int l = i / 2;
        int p = l & mask;
        if (old != p) {
            float ang = p * w_angle;
            w = make_cuFloatComplex(cos(ang), sin(ang));
            old = p;
        }
        cpxAddSubMulCG(in + l, in + (n / 2) + l, out, &w);
    }
    }
}

void ompConstantGeometryAlternate(fftDir dir, cpx **in, cpx **out, const int n)
{
    int bit = log2_32(n);
    const int lead = 32 - bit;
    int steps = --bit;
    unsigned int mask = 0xffffffff << (steps - bit);
    float w_angle = dir * M_2_PI / n;
    ompCGBody(*in, *out, w_angle, mask, n);
    while (bit-- > 0) {
        swapBuffer(in, out);
        mask = 0xffffffff << (steps - bit);
        ompCGBody(*in, *out, w_angle, mask, n);
    }
    ompBitReverse(*out, dir, lead, n);
}