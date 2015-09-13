#include "fft_fixed.h"

#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_math.h"
#include "tb_fft_helper.h"
#include "tb_print.h"

//#include "fft_generated_fixed_reg.h"
#include "fft_generated_fixed_const.h"

__inline static void fft_xn(fft_direction dir, cpx **in, cpx **out, cpx *W, const int n);

void fft_fixed(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n)
{
#ifdef GENERATED_FIXED_CONST
    if (n == 4) {
        if (dir == FORWARD_FFT) {
            fft_x4(*in, *out);
        }
        else  {
            fft_x4inv(*in, *out);
        }
    }
    else if (n == 8) {
        if (dir == FORWARD_FFT)
            fft_x8(*in, *out);
        else
            fft_x8inv(*in, *out);
    }
    else if (n == 16) {
        if (dir == FORWARD_FFT)
            fft_x16(*in, *out);
        else
            fft_x16inv(*in, *out);
    }
    else if (n == 32) {
        if (dir == FORWARD_FFT)
            fft_x32(*in, *out);
        else
            fft_x32inv(*in, *out);
    }
    console_separator(1);
    console_print(*in, n);
    console_newline(1);
    console_print(*out, n);
    console_separator(1);
#endif
    // else
    cpx *W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);
    fft_xn(dir, in, out, W, n);
    bit_reverse(*out, dir, 32 - log2_32(n), n);
    free(W);

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
#ifdef GENERATED_FIXED_REG
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
#endif  
}