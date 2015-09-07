#include "fft_const_geom.h"

#include "amp_math.h"

#include "tb_math.h"
#include "tb_fft_helper.h"
#include "tb_print.h"


__inline void _fft_body(cpx *in, cpx *out, cpx *W, unsigned int mask, const int n);
__inline void _fft_body(cpx *in, cpx *out, float w_angle, unsigned int mask, const int n);
__inline void _fft_const_geom(fft_direction dir, cpx **seq, cpx **buf, cpx *W, const int n);

__inline void swap(cpx **in, cpx **out)
{
    cpx *tmp = *in;
    *in = *out;
    *out = tmp;
}

void fft_const_geom(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    int steps, depth;
    cpx *W;
    depth = log2_32(n);
    W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);

    steps = 0;
    _fft_body(*in, *out, W, 0xffffffff << steps, n);
    while (++steps < depth) {    
        swap(in, out);
        _fft_body(*in, *out, W, 0xffffffff << steps, n);
    }

    bit_reverse(*out, dir, 32 - depth, n);
    free(W);
}

_inline void _do_rows(fft_direction dir, cpx** seq, cpx *W, const int n)
{
    
#pragma omp parallel for schedule(static)
    for (int row = 0; row < n; ++row) {
        //_fft_const_geom(dir, seq[row], seq[row], W, n);
    }
}

void fft2d_const_geom(fft_direction dir, cpx** seq, const int n)
{
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);

    // TODO: create a buffer for each thread to use

    _do_rows(dir, seq, W, n);
    transpose(seq, n);
    _do_rows(dir, seq, W, n);
    transpose(seq, n);

    free(W);
}

void _fft_const_geom(fft_direction dir, cpx **in, cpx **out, cpx *W, const int n)
{
    int steps, depth;
    cpx *W;
    depth = log2_32(n);
    steps = 0;
    _fft_body(*in, *out, W, 0xffffffff << steps, n);
    while (++steps < depth) {
        swap(in, out);
        _fft_body(*in, *out, W, 0xffffffff << steps, n);
    }
    bit_reverse(*out, dir, 32 - depth, n);
}

void _fft_body(cpx *in, cpx *out, cpx *W, unsigned int mask, const int n)
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
    _fft_body(*in, *out, w_angle, mask, n);
    while (bit-- > 0)
    {
        tmp = *in;
        *in = *out;
        *out = tmp;
        mask = 0xffffffff << (steps - bit);
        _fft_body(*in, *out, w_angle, mask, n);
    }
    bit_reverse(*out, dir, lead, n);
}

#pragma warning(disable:4700) 
void _fft_body(cpx *in, cpx *out, float w_angle, unsigned int mask, const int n)
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