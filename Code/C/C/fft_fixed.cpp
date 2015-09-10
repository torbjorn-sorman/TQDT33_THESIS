#include <memory>
#include <iostream>
#include <string>
#include <cstdio>

#include "fft_fixed.h"

#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_math.h"
#include "tb_fft_helper.h"
#include "tb_print.h"

__inline static void fft_x4(cpx *in, cpx *out);
__inline static void fft_x4inv(cpx *in, cpx *out);
__inline static void fft_x8(cpx *in, cpx *out);
__inline static void fft_x8inv(cpx *in, cpx *out);

__inline static void fft_xn(fft_direction dir, cpx **in, cpx **out, cpx *W, const int n);

void fft_fixed(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    if (n == 4) {
        if (dir == FORWARD_FFT) fft_x4(*in, *out); else fft_x4inv(*in, *out);
        bit_reverse(*out, dir, 30, 4);
    } else if (n == 8) {
        if (dir == FORWARD_FFT) fft_x8(*in, *out); else fft_x8inv(*in, *out);

        console_separator(1);
        console_print(*in, n);
        console_newline(1);
        console_print(*out, n);
        console_separator(1);

        printf("bit rev\n");
        bit_reverse(*out, dir, 29, 8);

        console_separator(1);
        console_print(*in, n);
        console_newline(1);
        console_print(*out, n);
        console_separator(1);
    } else {
        cpx *W = (cpx *)malloc(sizeof(cpx) * n);
        twiddle_factors(W, dir, n);        
        fft_xn(dir, in, out, W, n);
        bit_reverse(*out, dir, 32 - log2_32(n), n);
        free(W);
    }            
}

__inline static void fft_x4(cpx *in, cpx *out)
{
    float r0p2 = (in[0].r + in[2].r);
    float i0p2 = (in[0].i + in[2].i);
    float r0m2 = (in[0].r - in[2].r);
    float i0m2 = (in[0].i - in[2].i);
    float r1p3 = (in[1].r + in[3].r);
    float i1p3 = (in[1].i + in[3].i);
    float r1m3 = (in[1].r - in[3].r);
    float i1m3 = (in[1].i - in[3].i);
    out[0].r = r0p2 + r1p3;
    out[0].i = i0p2 + i1p3;
    out[1].r = r0p2 - r1p3;
    out[1].i = i0p2 - i1p3;
    out[3].r = r0m2 + -i1m3;
    out[3].i = i0m2 + r1m3;
    out[2].r = r0m2 - -i1m3;
    out[2].i = i0m2 - r1m3;
}

__inline static void fft_x4inv(cpx *in, cpx *out)
{
    float r0p2 = (in[0].r + in[2].r);
    float i0p2 = (in[0].i + in[2].i);
    float r1p3 = (in[1].r + in[3].r);
    float i1p3 = (in[1].i + in[3].i);
    float r0m2 = (in[0].r - in[2].r);
    float i0m2 = (in[0].i - in[2].i);
    float r1m3 = (in[1].r - in[3].r);
    float i1m3 = (in[1].i - in[3].i);
    out[0].r = (r0p2 + r1p3);
    out[0].i = (i0p2 + i1p3);
    out[1].r = (r0p2 - r1p3);
    out[1].i = (i0p2 - i1p3);
    out[3].r = (r0m2 + i1m3);
    out[3].i = (i0m2 + r1m3);
    out[2].r = (r0m2 - i1m3);
    out[2].i = (i0m2 - r1m3);
}

__inline static void fft_x8(cpx *in, cpx *out)
{
    const int val = 0.70710678118f;
    out[0].r = ((in[0].r + in[4].r) + (in[2].r + in[6].r));
    out[0].i = ((in[0].i + in[4].i) + (in[2].i + in[6].i));
    out[2].r = ((in[0].r + in[4].r) - (in[2].r + in[6].r));
    out[2].i = ((in[0].i + in[4].i) - (in[2].i + in[6].i));
    out[1].r = ((in[1].r + in[5].r) + (in[3].r + in[7].r));
    out[1].i = ((in[1].i + in[5].i) + (in[3].i + in[7].i));
    out[3].r = ( - ((in[1].i + in[5].i) - (in[3].i + in[7].i)));
    out[3].i = ((in[1].r + in[5].r) - (in[3].r + in[7].r));
    out[4].r = ((in[0].r - in[4].r) - 0 * (in[0].i - in[4].i));
    out[4].i = ((in[0].i - in[4].i) + (in[2].r - in[6].r));
    out[6].r = ((in[0].r - in[4].r) + (in[2].i - in[6].i));
    out[6].i = ((in[0].i - in[4].i) - (in[2].r - in[6].r));
    out[5].r = ((val * (in[1].r - in[5].r) - val * (in[1].i - in[5].i)) + ((-val) * (in[3].r - in[7].r) - val * (in[3].i - in[7].i)));
    out[5].i = ((val * (in[1].r - in[5].r) + val * (in[1].i - in[5].i)) + (val * (in[3].r - in[7].r) + (-val) * (in[3].i - in[7].i)));
    out[7].r = ( - (val * (in[1].r - in[5].r) + val * (in[1].i - in[5].i)) - (val * (in[3].r - in[7].r) + (-val) * (in[3].i - in[7].i)));
    out[7].i = ((val * (in[1].r - in[5].r) - val * (in[1].i - in[5].i)) - ((-val) * (in[3].r - in[7].r) - val * (in[3].i - in[7].i)));
    swap(&in, &out);
    out[0].r = (in[0].r + in[1].r);
    out[0].i = (in[0].i + in[1].i);
    out[1].r = (in[0].r - in[1].r);
    out[1].i = (in[0].i - in[1].i);
    out[2].r = (in[2].r + in[3].r);
    out[2].i = (in[2].i + in[3].i);
    out[3].r = (in[2].r - in[3].r);
    out[3].i = (in[2].i - in[3].i);
    out[4].r = (in[4].r + in[5].r);
    out[4].i = (in[4].i + in[5].i);
    out[5].r = (in[4].r - in[5].r);
    out[5].i = (in[4].i - in[5].i);
    out[6].r = (in[6].r + in[7].r);
    out[6].i = (in[6].i + in[7].i);
    out[7].r = (in[6].r - in[7].r);
    out[7].i = (in[6].i - in[7].i);
}

__inline static void fft_x8inv(cpx *in, cpx *out)
{
    const int val = 0.70710678118f;
    out[0].r = (in[0].r + in[4].r);
    out[0].i = (in[0].i + in[4].i);
    out[1].r = (in[1].r + in[5].r);
    out[1].i = (in[1].i + in[5].i);
    out[2].r = (in[2].r + in[6].r);
    out[2].i = (in[2].i + in[6].i);
    out[3].r = (in[3].r + in[7].r);
    out[3].i = (in[3].i + in[7].i);
    out[4].r = (in[0].r - in[4].r);
    out[4].i = (in[0].i - in[4].i);
    out[5].r = (val * (in[1].r - in[5].r) + (val * (in[1].i - in[5].i)));
    out[5].i = ((-val) * (in[1].r - in[5].r) + val * (in[1].i - in[5].i));
    out[6].r = (in[2].i - in[6].i);
    out[6].i = (-(in[2].r - in[6].r));
    out[7].r = ((-val) * (in[3].r - in[7].r) + (val * (in[3].i - in[7].i)));
    out[7].i = ((-val) * (in[3].r - in[7].r) - (val * (in[3].i - in[7].i)));
    swap(&in, &out);
    out[0].r = (in[0].r + in[2].r);
    out[0].i = (in[0].i + in[2].i);
    out[1].r = (in[1].r + in[3].r);
    out[1].i = (in[1].i + in[3].i);
    out[2].r = (in[0].r - in[2].r);
    out[2].i = (in[0].i - in[2].i);
    out[3].r = (in[1].i - in[3].i);
    out[3].i = (-(in[1].r - in[3].r));
    out[4].r = (in[4].r + in[6].r);
    out[4].i = (in[4].i + in[6].i);
    out[5].r = (in[5].r + in[7].r);
    out[5].i = (in[5].i + in[7].i);
    out[6].r = (in[4].r - in[6].r);
    out[6].i = (in[4].i - in[6].i);
    out[7].r = (in[5].i - in[7].i);
    out[7].i = (-(in[5].r - in[7].r));
    swap(&in, &out);
    out[0].r = (in[0].r + in[1].r);
    out[0].i = (in[0].i + in[1].i);
    out[1].r = (in[0].r - in[1].r);
    out[1].i = (in[0].i - in[1].i);
    out[2].r = (in[2].r + in[3].r);
    out[2].i = (in[2].i + in[3].i);
    out[3].r = (in[2].r - in[3].r);
    out[3].i = (in[2].i - in[3].i);
    out[4].r = (in[4].r + in[5].r);
    out[4].i = (in[4].i + in[5].i);
    out[5].r = (in[4].r - in[5].r);
    out[5].i = (in[4].i - in[5].i);
    out[6].r = (in[6].r + in[7].r);
    out[6].i = (in[6].i + in[7].i);
    out[7].r = (in[6].r - in[7].r);
    out[7].i = (in[6].i - in[7].i);
}

__inline static void fft_xn(fft_direction dir, cpx **in, cpx **out, cpx *W, const int n)
{
    int p, u, mul, upper;
    float tmp_r, tmp_i;
    const int n8 = n / 8;
    int dist2 = n;
    int dist = (n / 2);
    while ((dist2 = dist) > 1) {
        dist = dist >> 1;
        for (int lower = 0; lower < n8; lower += dist2) {
            mul = (n / dist2);
            upper = dist + lower;
            for (int l = lower; l < upper; ++l) {
                u = l + dist;
                p = (l - lower) * mul;
                tmp_r = (*in)[l].r - (*in)[u].r;
                tmp_i = (*in)[l].i - (*in)[u].i;
                (*out)[l].r = (*in)[l].r + (*in)[u].r;
                (*out)[l].i = (*in)[l].i + (*in)[u].i;
                (*out)[u].r = (W[p].r * tmp_r) - (W[p].i * tmp_i);
                (*out)[u].i = (W[p].i * tmp_r) + (W[p].r * tmp_i);
            }
        }
    }
    
    if (dir == FORWARD_FFT) {
        for (int i = 0; i < n; i += 8) {
            fft_x8(&((*out)[i]), &((*out)[i]));
        }
    }
    else {
        for (int i = 0; i < n; i += 8) {
            fft_x8inv(&((*out)[i]), &((*out)[i]));
        }
    }
    
}
