#include "omp_constant_geometry.h"

//
// Testing
//

int openmp_validate(const int n)
{
    cpx *in = get_seq(n, 1);
    cpx *out = get_seq(n);
    cpx *ref = get_seq(n, in);
    openmp_const_geom(FFT_FORWARD, &in, &out, n);
    double diffForward = diff_forward_sinus(out, n);
    openmp_const_geom(FFT_INVERSE, &out, &in, n);
    double diff = diff_seq(in, ref, n);
    free_all(in, out, ref);
    return diff < RELATIVE_ERROR_MARGIN && diffForward < RELATIVE_ERROR_MARGIN;
}

int openmp_2d_validate(const int n, bool write_img)
{
    cpx *in, *buf, *ref;
    setup_seq_2d(&in, &buf, &ref, n);

    openmp_const_geom_2d(FFT_FORWARD, &in, &buf, n);
    if (write_img)
        write_normalized_image("OpenMP", "freq", in, n, true);
    openmp_const_geom_2d(FFT_INVERSE, &in, &buf, n);
    if (write_img)
        write_image("OpenMP", "spat", in, n);

    double diff = diff_seq(in, ref, n);
    free_all(in, buf, ref);
    return diff < RELATIVE_ERROR_MARGIN;
}

double openmp_performance(const int n)
{
    double measures[64];
    cpx *in = get_seq(n, 1);
    for (int i = 0; i < number_of_tests; ++i) {
        start_timer();
        openmp_const_geom(FFT_FORWARD, &in, &in, n);
        measures[i] = stop_timer();
    }
    free(in);
    return average_best(measures, number_of_tests);
}

double openmp_2d_performance(const int n)
{
    double measures[64];
    cpx *in = get_seq(n * n);
    cpx *buf = get_seq(n * n);
    for (int i = 0; i < number_of_tests; ++i) {
        start_timer();
        openmp_const_geom_2d(FFT_FORWARD, &in, &buf, n);
        measures[i] = stop_timer();
    }
    free_all(in, buf);
    return average_best(measures, number_of_tests);
}

//
// Algorithm
//

_inline void openmp_inner_body(cpx *in, cpx *out, const cpx *w, const unsigned int mask, const int n_half)
{
#pragma omp parallel for schedule(static)
    for (int l = 0; l < n_half; ++l)
        c_add_sub_mul_cg(in + l, in + n_half + l, out + (l << 1), w + (l & mask));
}

void openmp_const_geom(transform_direction dir, cpx **in, cpx **out, const int n)
{
    const int n_half = n / 2;
    int steps_left = log2_32(n);
    int steps = 0;
    cpx *w = (cpx *)malloc(sizeof(cpx) * n);
    openmp_twiddle_factors(w, dir, n);
    openmp_inner_body(*in, *out, w, 0xffffffff << steps, n_half);
    while (++steps < steps_left) {
        swap_buffer(in, out);
        openmp_inner_body(*in, *out, w, 0xffffffff << steps, n_half);
    }
    openmp_bit_reverse(*out, dir, 32 - steps_left, n);
    free(w);
}

_inline void openmp_const_geom_2d_helper(transform_direction dir, cpx *in, cpx *out, const cpx *w, const int n)
{
    const int n_half = n / 2;
    int steps_left = log2_32(n);
    int steps = 0;
    openmp_inner_body(in, out, w, 0xffffffff << steps, n_half);
    while (++steps < steps_left) {
        swap_buffer(&in, &out);
        openmp_inner_body(in, out, w, 0xffffffff << steps, n_half);
    }
    openmp_bit_reverse(out, dir, 32 - steps_left, n);
}

_inline void openmp_rows(transform_direction dir, cpx **in, cpx **out, const cpx *w, const int n)
{
#pragma omp parallel for schedule(static)
    for (int row = 0; row < n * n; row += n)
        openmp_const_geom_2d_helper(dir, (*in) + row, (*out) + row, w, n);
    if (log2_32(n) % 2 == 0)
        swap_buffer(in, out);
}

void openmp_const_geom_2d(transform_direction dir, cpx **in, cpx **out, const int n)
{
    cpx *w = (cpx *)malloc(sizeof(cpx) * n);
    openmp_twiddle_factors(w, dir, n);    
    openmp_rows(dir, in, out, w, n);
    openmp_transpose(*out, *in, n);
    openmp_rows(dir, in, out, w, n);
    openmp_transpose(*out, *in, n);
    swap_buffer(in, out);
    free(w);
}

_inline void openmp_inner_body(cpx *in, cpx *outG, float global_angle, unsigned int mask, const int n)
{    
    const int n_half = n / 2;    
#pragma omp parallel
    {
        int old = -1;        
        cpx w, *out = &outG[omp_get_thread_num() * n / omp_get_num_threads()];
#pragma omp for schedule(static)// private(w, old) // might remove this private decl.?
    for (int i = 0; i < n; i += 2) {
        int l = i / 2;
        int p = l & mask;
        if (old != p) {
            float ang = p * global_angle;
            w = {cos(ang), sin(ang)};
            old = p;
        }
        c_add_sub_mul_cg(in + l, in + (n / 2) + l, out, &w);
    }
    }
}

void openmp_const_geom_alt(transform_direction dir, cpx **in, cpx **out, const int n)
{
    int bit = log2_32(n);
    const int leading_bits = 32 - bit;
    int steps = --bit;
    unsigned int mask = 0xffffffff << (steps - bit);
    float global_angle = dir * M_2_PI / n;
    openmp_inner_body(*in, *out, global_angle, mask, n);
    while (bit-- > 0) {
        swap_buffer(in, out);
        mask = 0xffffffff << (steps - bit);
        openmp_inner_body(*in, *out, global_angle, mask, n);
    }
    openmp_bit_reverse(*out, dir, leading_bits, n);
}