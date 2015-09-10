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

__inline static void fft_x8(fft_direction dir, cpx **in, cpx **out, cpx *W);

void fft_fixed(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    if (n == 4) {
        if (dir == FORWARD_FFT) fft_x4(*in, *out); else fft_x4inv(*in, *out);
        bit_reverse(*out, dir, 30, 4);
    } else {
        cpx *W = (cpx *)malloc(sizeof(cpx) * n);
        twiddle_factors(W, dir, n);
        if (n == 8) {
            fft_x8(dir, in, out, W);
            bit_reverse(*out, dir, 29, 8);
        } else {
            (*in)[0].i = 99999.f;
        }
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
    /*
    W[0].r = cos(0) = 1;
    W[0].i = sin(0) = 0;
    W[1].r = cos(pi / 4) = val;
    W[1].i = sin(pi / 4) = val;
    W[2].r = cos(pi / 2) = 0;
    W[2].i = sin(pi / 2) = 1;
    W[3].r = cos(3pi / 4) = -val;
    W[3].i = sin(3pi / 4) = val;
    */
    cpx *W;
    out[0].r = (in[0].r + in[4].r);
    out[0].i = (in[0].i + in[4].i);
    out[4].r = (W[0].r * (in[0].r - in[4].r) - W[0].i * (in[0].i - in[4].i));
    out[4].i = (W[0].i * (in[0].r - in[4].r) + W[0].r * (in[0].i - in[4].i));
    out[1].r = (in[1].r + in[5].r);
    out[1].i = (in[1].i + in[5].i);
    out[5].r = (W[1].r * (in[1].r - in[5].r) - W[1].i * (in[1].i - in[5].i));
    out[5].i = (W[1].i * (in[1].r - in[5].r) + W[1].r * (in[1].i - in[5].i));
    out[2].r = (in[2].r + in[6].r);
    out[2].i = (in[2].i + in[6].i);
    out[6].r = (W[2].r * (in[2].r - in[6].r) - W[2].i * (in[2].i - in[6].i));
    out[6].i = (W[2].i * (in[2].r - in[6].r) + W[2].r * (in[2].i - in[6].i));
    out[3].r = (in[3].r + in[7].r);
    out[3].i = (in[3].i + in[7].i);
    out[7].r = (W[3].r * (in[3].r - in[7].r) - W[3].i * (in[3].i - in[7].i));
    out[7].i = (W[3].i * (in[3].r - in[7].r) + W[3].r * (in[3].i - in[7].i));

    out[0].r = (out[0].r + out[2].r);
    out[0].i = (out[0].i + out[2].i);
    out[2].r = (W[0].r * (out[0].r - out[2].r) - W[0].i * (out[0].i - out[2].i));
    out[2].i = (W[0].i * (out[0].r - out[2].r) + W[0].r * (out[0].i - out[2].i));
    out[1].r = (out[1].r + out[3].r);
    out[1].i = (out[1].i + out[3].i);
    out[3].r = (W[2].r * (out[1].r - out[3].r) - W[2].i * (out[1].i - out[3].i));
    out[3].i = (W[2].i * (out[1].r - out[3].r) + W[2].r * (out[1].i - out[3].i));
    out[4].r = (out[4].r + out[6].r);
    out[4].i = (out[4].i + out[6].i);
    out[6].r = (W[0].r * (out[4].r - out[6].r) - W[0].i * (out[4].i - out[6].i));
    out[6].i = (W[0].i * (out[4].r - out[6].r) + W[0].r * (out[4].i - out[6].i));
    out[5].r = (out[5].r + out[7].r);
    out[5].i = (out[5].i + out[7].i);
    out[7].r = (W[2].r * (out[5].r - out[7].r) - W[2].i * (out[5].i - out[7].i));
    out[7].i = (W[2].i * (out[5].r - out[7].r) + W[2].r * (out[5].i - out[7].i));

    out[0].r = (out[0].r + out[1].r);
    out[0].i = (out[0].i + out[1].i);
    out[1].r = (W[0].r * (out[0].r - out[1].r) - W[0].i * (out[0].i - out[1].i));
    out[1].i = (W[0].i * (out[0].r - out[1].r) + W[0].r * (out[0].i - out[1].i));
    out[2].r = (out[2].r + out[3].r);
    out[2].i = (out[2].i + out[3].i);
    out[3].r = (W[0].r * (out[2].r - out[3].r) - W[0].i * (out[2].i - out[3].i));
    out[3].i = (W[0].i * (out[2].r - out[3].r) + W[0].r * (out[2].i - out[3].i));
    out[4].r = (out[4].r + out[5].r);
    out[4].i = (out[4].i + out[5].i);
    out[5].r = (W[0].r * (out[4].r - out[5].r) - W[0].i * (out[4].i - out[5].i));
    out[5].i = (W[0].i * (out[4].r - out[5].r) + W[0].r * (out[4].i - out[5].i));
    out[6].r = (out[6].r + out[7].r);
    out[6].i = (out[6].i + out[7].i);
    out[7].r = (W[0].r * (out[6].r - out[7].r) - W[0].i * (out[6].i - out[7].i));
    out[7].i = (W[0].i * (out[6].r - out[7].r) + W[0].r * (out[6].i - out[7].i));
}

__inline static void fft_x8inv(cpx *in, cpx *out)
{
    const int val = 0.70710678118f;
    /*
    W[0].r = cos(0) = 1;
    (-W[0].i) = sin(0) = 0;
    W[1].r = cos(pi / 4) = val;
    (-W[1].i) = sin(pi / 4) = val;
    W[2].r = cos(pi / 2) = 0;
    (-W[2].i) = sin(pi / 2) = 1;
    W[3].r = cos(3pi / 4) = -val;
    (-W[3].i) = sin(3pi / 4) = val;
    */
    cpx *W;
    out[0].r = (in[0].r + in[4].r);
    out[0].i = (in[0].i + in[4].i);
    out[4].r = (W[0].r * (in[0].r - in[4].r) - (-W[0].i) * (in[0].i - in[4].i));
    out[4].i = ((-W[0].i) * (in[0].r - in[4].r) + W[0].r * (in[0].i - in[4].i));
    out[1].r = (in[1].r + in[5].r);
    out[1].i = (in[1].i + in[5].i);
    out[5].r = (W[1].r * (in[1].r - in[5].r) - (-W[1].i) * (in[1].i - in[5].i));
    out[5].i = ((-W[1].i) * (in[1].r - in[5].r) + W[1].r * (in[1].i - in[5].i));
    out[2].r = (in[2].r + in[6].r);
    out[2].i = (in[2].i + in[6].i);
    out[6].r = (W[2].r * (in[2].r - in[6].r) - (-W[2].i) * (in[2].i - in[6].i));
    out[6].i = ((-W[2].i) * (in[2].r - in[6].r) + W[2].r * (in[2].i - in[6].i));
    out[3].r = (in[3].r + in[7].r);
    out[3].i = (in[3].i + in[7].i);
    out[7].r = (W[3].r * (in[3].r - in[7].r) - (-W[3].i) * (in[3].i - in[7].i));
    out[7].i = ((-W[3].i) * (in[3].r - in[7].r) + W[3].r * (in[3].i - in[7].i));

    out[0].r = (out[0].r + out[2].r);
    out[0].i = (out[0].i + out[2].i);
    out[2].r = (W[0].r * (out[0].r - out[2].r) - (-W[0].i) * (out[0].i - out[2].i));
    out[2].i = ((-W[0].i) * (out[0].r - out[2].r) + W[0].r * (out[0].i - out[2].i));
    out[1].r = (out[1].r + out[3].r);
    out[1].i = (out[1].i + out[3].i);
    out[3].r = (W[2].r * (out[1].r - out[3].r) - (-W[2].i) * (out[1].i - out[3].i));
    out[3].i = ((-W[2].i) * (out[1].r - out[3].r) + W[2].r * (out[1].i - out[3].i));
    out[4].r = (out[4].r + out[6].r);
    out[4].i = (out[4].i + out[6].i);
    out[6].r = (W[0].r * (out[4].r - out[6].r) - (-W[0].i) * (out[4].i - out[6].i));
    out[6].i = ((-W[0].i) * (out[4].r - out[6].r) + W[0].r * (out[4].i - out[6].i));
    out[5].r = (out[5].r + out[7].r);
    out[5].i = (out[5].i + out[7].i);
    out[7].r = (W[2].r * (out[5].r - out[7].r) - (-W[2].i) * (out[5].i - out[7].i));
    out[7].i = ((-W[2].i) * (out[5].r - out[7].r) + W[2].r * (out[5].i - out[7].i));

    out[0].r = (out[0].r + out[1].r);
    out[0].i = (out[0].i + out[1].i);
    out[1].r = (W[0].r * (out[0].r - out[1].r) - (-W[0].i) * (out[0].i - out[1].i));
    out[1].i = ((-W[0].i) * (out[0].r - out[1].r) + W[0].r * (out[0].i - out[1].i));
    out[2].r = (out[2].r + out[3].r);
    out[2].i = (out[2].i + out[3].i);
    out[3].r = (W[0].r * (out[2].r - out[3].r) - (-W[0].i) * (out[2].i - out[3].i));
    out[3].i = ((-W[0].i) * (out[2].r - out[3].r) + W[0].r * (out[2].i - out[3].i));
    out[4].r = (out[4].r + out[5].r);
    out[4].i = (out[4].i + out[5].i);
    out[5].r = (W[0].r * (out[4].r - out[5].r) - (-W[0].i) * (out[4].i - out[5].i));
    out[5].i = ((-W[0].i) * (out[4].r - out[5].r) + W[0].r * (out[4].i - out[5].i));
    out[6].r = (out[6].r + out[7].r);
    out[6].i = (out[6].i + out[7].i);
    out[7].r = (W[0].r * (out[6].r - out[7].r) - (-W[0].i) * (out[6].i - out[7].i));
    out[7].i = ((-W[0].i) * (out[6].r - out[7].r) + W[0].r * (out[6].i - out[7].i));
}

__inline static void fft_x8(fft_direction dir, cpx **in, cpx **out, cpx *W)
{
    int u, ii;
    float tmp_r, tmp_i;
    for (int l = 0; l < 4; ++l) {
            u = l + 4;
            tmp_r = (*in)[l].r - (*in)[u].r;
            tmp_i = (*in)[l].i - (*in)[u].i;
            (*out)[l].r = (*in)[l].r + (*in)[u].r;
            (*out)[l].i = (*in)[l].i + (*in)[u].i;
            (*out)[u].r = (W[l].r * tmp_r) - (W[l].i * tmp_i);
            (*out)[u].i = (W[l].i * tmp_r) + (W[l].r * tmp_i);
    }
    if (dir == FORWARD_FFT) {
        fft_x4((*out), (*out));
        fft_x4(&((*out)[4]), &((*out)[4]));
    }
    else {
        fft_x4inv((*out), (*out));
        fft_x4inv(&((*out)[4]), &((*out)[4]));
    }
}