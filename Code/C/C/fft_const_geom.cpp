#include "fft_const_geom.h"

#include "tb_math.h"
#include "tb_fft_helper.h"
#include "tb_print.h"

void _fft_body(cpx *in, cpx *out, cpx *W, unsigned int mask, const int n_threads, const int n);
void _fft_body(cpx *in, cpx *out, float w_angle, unsigned int mask, const int n_threads, const int n);

/* Both 'in' and 'out' will be used and the values in 'in' will not be preserved. */
void fft_const_geom(const double dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    int bit, steps;
    unsigned int mask;
    cpx *tmp, *W;
    bit = log2_32(n);
    const int lead = 32 - bit;
    steps = --bit;
    mask = 0xffffffff << (steps - bit);
    W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n_threads, n);

    _fft_body(*in, *out, W, mask, n_threads, n);
    while (bit-- > 0)
    {
        tmp = *in;
        *in = *out;
        *out = tmp;
        mask = 0xffffffff << (steps - bit);
        _fft_body(*in, *out, W, mask, n_threads, n);
    }
    bit_reverse(*out, dir, lead, n_threads, n);
}

void _fft_body(cpx *in, cpx *out, cpx *W, unsigned int mask, const int n_threads, const int n)
{
    int i, l, u, p, n2, chunk;
    cpx tmp;
    n2 = n / 2;
    chunk = n2 / n_threads;
#pragma omp parallel for schedule(static, chunk) private(i, l, u, p, tmp) shared(in, out, W, mask, n, n2)
    for (i = 0; i < n; i += 2) {
        l = i / 2;
        u = n2 + l;
        p = l & mask;
        tmp.r = in[l].r - in[u].r;
        tmp.i = in[l].i - in[u].i;
        out[i].r = in[l].r + in[u].r;
        out[i].i = in[l].i + in[u].i;
        out[i + 1].r = (W[p].r * tmp.r) - (W[p].i * tmp.i);
        out[i + 1].i = (W[p].i * tmp.r) + (W[p].r * tmp.i);
    }
}

void fft_const_geom_2(const double dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    int bit, steps;
    unsigned int mask;
    cpx *tmp;
    float w_angle;
    bit = log2_32(n);
    const int lead = 32 - bit;
    steps = --bit;
    mask = 0xffffffff << (steps - bit);
    w_angle = (float)dir * M_2_PI / n;
    _fft_body(*in, *out, w_angle, mask, n_threads, n);
    while (bit-- > 0)
    {
        tmp = *in;
        *in = *out;
        *out = tmp;
        mask = 0xffffffff << (steps - bit);
        _fft_body(*in, *out, w_angle, mask, n_threads, n);
    }
    bit_reverse(*out, dir, lead, n_threads, n);
}

#pragma warning(disable:4700) 
void _fft_body(cpx *in, cpx *out, float w_angle, unsigned int mask, const int n_threads, const int n)
{
    int i, l, u, p, n2, old, chunk;
    float cv, sv;
    cpx tmp;
    n2 = n / 2;
    old = -1;
    chunk = n2 / n_threads;
#pragma omp parallel for schedule(static, chunk) private(i, l, u, p, tmp, cv, sv, old) shared(in, out, w_angle, mask, n, n2)
    for (i = 0; i < n; i += 2) {
        l = i / 2;
        u = n2 + l;
        p = l & mask;
        if (old != p) {
            cv = cos(p * w_angle);
            sv = sin(p * w_angle);
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