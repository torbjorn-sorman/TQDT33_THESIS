#include "c_fft.h"

//
// Testing
//

//#define NO_TWIDDLE_TABLE

int c_validate(const int n)
{
    cpx *in = get_seq(n, 1);
    cpx *out = get_seq(n);
    cpx *ref = get_seq(n, in);
#ifdef NO_TWIDDLE_TABLE
    c_const_geom_alt(FFT_FORWARD, &in, &out, n);
    c_const_geom_alt(FFT_INVERSE, &out, &in, n);
#else
    c_const_geom(FFT_FORWARD, &in, &out, n);
    c_const_geom(FFT_INVERSE, &out, &in, n);
#endif
    double diff = diff_seq(in, ref, n);
    free(in);
    free(out);
    free(ref);
    return diff < RELATIVE_ERROR_MARGIN;
}

int c_2d_validate(const int n, bool write_img)
{
    cpx *in, *buf, *ref;
    setup_seq2D(&in, &buf, &ref, n);

    c_const_geom_2d(FFT_FORWARD, &in, &buf, n);
    if (write_img)
        write_normalized_image("C_C++", "freq", in, n, true);
    c_const_geom_2d(FFT_INVERSE, &in, &buf, n);
    if (write_img)
        write_image("C_C++", "spat", in, n);

    double diff = diff_seq(in, ref, n);
    free(in);
    free(buf);
    free(ref);
    return diff < RELATIVE_ERROR_MARGIN;
}

double c_performance(const int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in = get_seq(n, 1);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
#ifdef NO_TWIDDLE_TABLE
        c_const_geom_alt(FFT_FORWARD, &in, &in, n);
#else
        c_const_geom(FFT_FORWARD, &in, &in, n);
#endif
        measures[i] = stopTimer();
    }
    free(in);
    return average_best(measures, NUM_PERFORMANCE);
}

double c_2d_performance(const int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in = get_seq(n * n);
    cpx *out = get_seq(n * n);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        c_const_geom_2d(FFT_FORWARD, &in, &out, n);
        measures[i] = stopTimer();
    }
    free(in);
    free(out);
    return average_best(measures, NUM_PERFORMANCE);
}

//
// Algorithm
//

_inline void c_inner_body(cpx *in, cpx *out, const cpx *W, const unsigned int mask, const int n_half)
{    
    for (int l = 0; l < n_half; ++l) {
        c_add_sub_mul_cg(in + l, in + n_half + l, out, W + (l & mask));
        out += 2;
    }
}

void c_const_geom(transform_direction dir, cpx **in, cpx **out, const int n)
{
    int steps_left = log2_32(n);
    int steps = 0;
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);
    c_inner_body(*in, *out, W, 0xffffffff << steps, n / 2);
    while (++steps < steps_left) {
        swap_buffer(in, out);
        c_inner_body(*in, *out, W, 0xffffffff << steps, n / 2);
    }

    bit_reverse(*out, dir, 32 - steps_left, n);
    free(W);
}

_inline void c_const_geom_2d_helper(transform_direction dir, cpx *in, cpx *out, const cpx *W, const int n)
{
    int steps_left = log2_32(n);
    int steps = 0;
    c_inner_body(in, out, W, 0xffffffff << steps, n / 2);
    while (++steps < steps_left) {
        swap_buffer(&in, &out);
        c_inner_body(in, out, W, 0xffffffff << steps, n / 2);
    }
    bit_reverse(out, dir, 32 - steps_left, n);
}

// Result is found in the *in variable

void c_const_geom_2d(transform_direction dir, cpx **in, cpx **out, const int n)
{
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);
    for (int row = 0; row < n * n; row += n)
        c_const_geom_2d_helper(dir, (*in) + row, (*out) + row, W, n);
    if (log2_32(n) % 2 == 0) 
        swap_buffer(in, out);
    transpose(*out, *in, n);
    for (int row = 0; row < n * n; row += n)
        c_const_geom_2d_helper(dir, (*in) + row, (*out) + row, W, n);
    if (log2_32(n) % 2 == 0) 
        swap_buffer(in, out);
    transpose(*out, *in, n);
    free(W);
}

_inline void c_inner_body(cpx *in, cpx *out, float global_angle, unsigned int mask, const int n)
{
    cpx w;
    int old = -1;
    for (int i = 0; i < n; i += 2) {
        int l = i / 2;
        int p = l & mask;
        if (old != p) {
            float ang = p * global_angle;
            w = make_cuFloatComplex(cos(ang), sin(ang));
            old = p;
        }
        c_add_sub_mul_cg(in + l, in + (n / 2) + l, out, &w);
    }
}

void c_const_geom_alt(transform_direction dir, cpx **in, cpx **out, const int n)
{
    int steps_left = log2_32(n);
    int steps = 0;
    float global_angle = dir * M_2_PI / n;
    c_inner_body(*in, *out, global_angle, 0xffffffff << steps, n);
    while (steps++ < steps_left) {
        swap_buffer(in, out);
        c_inner_body(*in, *out, global_angle, 0xffffffff << steps, n);
    }
    bit_reverse(*out, dir, 32 - steps_left, n);
}