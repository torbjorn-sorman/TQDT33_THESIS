#include <omp.h> 
#include "fft_const_geom.h"

#include "tb_math.h"
#include "tb_fft_helper.h"
#include "tb_print.h"


__inline void _fft_cgbody(cpx *in, cpx *out, const cpx *W, const unsigned int mask, const int n);
__inline void _fft_cgbody(cpx *in, cpx *out, const float w_angle, const unsigned int mask, const int n);
__inline void _fft_const_geom(fft_direction dir, cpx **seq, cpx **buf, const cpx *W, const int n);

void fft_const_geom(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    int steps, depth;
    cpx *W;
    depth = log2_32(n);
    W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);

    steps = 0;
    _fft_cgbody(*in, *out, W, 0xffffffff << steps, n);
    while (++steps < depth) {
        swap(in, out);
        _fft_cgbody(*in, *out, W, 0xffffffff << steps, n);
    }

    bit_reverse(*out, dir, 32 - depth, n);
    free(W);
}

_inline void _do_rows(fft_direction dir, cpx** seq, const cpx *W, cpx **buffers, const int n_threads, const int n)
{
#pragma omp parallel
    {
        int tid = 0;
#pragma omp for schedule(static) private(tid)
        for (int row = 0; row < n; ++row) {
            _fft_const_geom(dir, &seq[row], &buffers[tid = omp_get_thread_num()], W, n);
            swap(&seq[row], &buffers[tid]);
        }
    }
}

void fft2d_const_geom(fft_direction dir, cpx** seq, const int n_threads, const int n)
{
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);
    cpx** buffers = (cpx **)malloc(sizeof(cpx *) * n_threads);
#pragma omp for schedule(static)
    for (int i = 0; i < n_threads; ++i) {
        buffers[i] = (cpx *)malloc(sizeof(cpx) * n);
    }
    _do_rows(dir, seq, W, buffers, n_threads, n);
    transpose(seq, n);
    _do_rows(dir, seq, W, buffers, n_threads, n);
    transpose(seq, n);
#pragma omp for schedule(static)
    for (int i = 0; i < n_threads; ++i) {
        free(buffers[i]);
    }
    free(buffers);
    free(W);
}

_inline void _fft_const_geom(fft_direction dir, cpx **in, cpx **out, const cpx *W, const int n)
{
    int steps, depth;
    depth = log2_32(n);
    steps = 0;
    _fft_cgbody(*in, *out, W, 0xffffffff << steps, n);
    while (++steps < depth) {
        swap(in, out);
        _fft_cgbody(*in, *out, W, 0xffffffff << steps, n);
    }
    bit_reverse(*out, dir, 32 - depth, n);
}

_inline void _fft_cgbody(cpx *in, cpx *out, const cpx *W, unsigned int mask, const int n)
{
    int l, u, p, n2;
    float tmp_r, tmp_i;
    n2 = n / 2;
#pragma omp parallel for schedule(static) private(l, u, p, tmp_r, tmp_i)
    for (int i = 0; i < n; i += 2) {
        l = i / 2;
        u = n2 + l;
        p = l & mask;
        tmp_r = in[l].r - in[u].r;
        tmp_i = in[l].i - in[u].i;
        out[i].r = in[l].r + in[u].r;
        out[i].i = in[l].i + in[u].i;
        out[i + 1].r = (W[p].r * tmp_r) - (W[p].i * tmp_i);
        out[i + 1].i = (W[p].i * tmp_r) + (W[p].r * tmp_i);
    }
}

void fft_const_geom_2(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    int bit, steps;
    unsigned int mask;
    cpx *tmp;
    float w_angle;
    bit = log2_32(n);
    const int lead = 32 - bit;
    steps = --bit;
    mask = 0xffffffff << (steps - bit);
    w_angle = dir * M_2_PI / n;
    _fft_cgbody(*in, *out, w_angle, mask, n);
    while (bit-- > 0)
    {
        tmp = *in;
        *in = *out;
        *out = tmp;
        mask = 0xffffffff << (steps - bit);
        _fft_cgbody(*in, *out, w_angle, mask, n);
    }
    bit_reverse(*out, dir, lead, n);
}

#pragma warning(disable:4700) 
_inline void _fft_cgbody(cpx *in, cpx *out, float w_angle, unsigned int mask, const int n)
{
    int l, u, p, n2, old;
    float cv, sv, ang;
    cpx tmp;
    n2 = n / 2;
    old = -1;
#pragma omp parallel for schedule(static) private(l, u, p, tmp, cv, sv, ang, old)
    for (int i = 0; i < n; i += 2) {
        l = i / 2;
        u = n2 + l;
        p = l & mask;
        if (old != p) {
            cv = cos(ang = p * w_angle);
            sv = sin(ang);
            old = p;
        }
        tmp.r = in[l].r - in[u].r;
        tmp.i = in[l].i - in[u].i;

        out[i].r = in[l].r + in[u].r;
        out[i].i = in[l].i + in[u].i;

        out[i + 1].r = (cv * tmp.r) - (sv * tmp.i);
        out[i + 1].i = (sv * tmp.r) + (cv * tmp.i);
    }
#pragma warning(default:4700)
}