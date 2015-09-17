#include "fft_radix4.h"

__inline void _fft_radix2(cpx *in, cpx *out, cpx *W, fft_direction dir, const int dist, const int dist2, const int n);
__inline void _fft_radix4(cpx *in, cpx *out, cpx *W, fft_direction dir, const int dist, const int dist2, const int n);


static __inline void radix4_bit_reverse(fft_direction dir, cpx *seq, const int n)
{
    cpx tmp;
    int j = 0;
    int _n;
    const int n4 = n >> 2;
    const cpx scale = make_cpx((dir == FORWARD_FFT ? 1.f : (1.f / (float) n)), 0.f);
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) {
            tmp = cpxMul(seq[i], scale);
            seq[i] = cpxMul(seq[j], scale);
            seq[j] = tmp;
        }
        _n = n4;
        while (j >= 3 * _n) {
            j -= 3 * _n;
            _n >>= 2;
        }
        j += _n;
    }
}

void fft_radix4(fft_direction dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    const int depth = log2_32(n);
    int dist, dist2, radix, radix_shift;
    void(*fn)(cpx *in, cpx *out, cpx *W, fft_direction dir, const int dist, const int dist2, const int n);
    cpx *W;
    if ((depth % 2 == 1)) {
        radix = 2;
        fn = _fft_radix2;
    }
    else {
        radix = 4;
        fn = _fft_radix4;
    }
    radix_shift = log2_32(radix);
    W = (cpx *)malloc(sizeof(cpx) * n);
    twiddle_factors(W, dir, n);

    dist = n;
    dist2 = (dist >> radix_shift);
    fn(*in, *out, W, dir, dist, dist2, n);
    for (int k = 0; k < depth; ++k) {
        dist = dist2;
        dist2 = (dist2 >> radix_shift);
        fn(*out, *out, W, dir, dist, dist2, n);
    }

    if (radix == 4)
        radix4_bit_reverse(dir, *out, n);
    else
        bit_reverse(*out, dir, 32 - depth, n);
    
    if (radix == 4 && dir == FORWARD_FFT) {
        console_separator(1);
        console_print(*out, n);
        console_separator(1);
    }
    free(W);
}

__inline void _fft_radix2(cpx *in, cpx *out, cpx *W, fft_direction dir, const int dist, const int dist2, const int n)
{
    int u, p;
    cpx inLower, inUpper;
    const int mul = n / dist;
    for (int lower = 0; lower < n; lower += dist) {
        int upper = dist2 + lower;
        for (int l = lower; l < upper; ++l) {
            u = l + dist2;
            p = (l - lower) * mul;
            inLower = in[l];
            inUpper = in[u];
            out[l] = cpxAdd(inLower, inUpper);
            out[u] = cpxMul(cpxSub(inLower, inUpper), W[p]);
        }
    }
}

static __inline void _sincosf(float a, float *y, float *x)
{
    *y = sinf(a);
    *x = cosf(a);
}

__inline void _fft_radix4(cpx *in, cpx *out, cpx *W, fft_direction dir, const int dist, const int dist2, const int n)
{
    int i1, i2, i3;
    const cpx jmag = make_cpx(0.f, 1.f);
    const cpx jmag_neg = make_cpx(0.f, -1.f);
    cpx w1, w2, w3;
    cpx in0, in1, in2, in3;
    cpx tmp1, tmp2, tmp3, tmp4;
    const float e = (dir * M_2_PI) / ((float)dist);
    for (int j = 0; j < dist2; ++j) {
        float a = j * e;
        float b = a + a;
        float c = a + b;
        _sincosf(a, &w1.i, &w1.r);
        _sincosf(b, &w2.i, &w2.r);
        _sincosf(c, &w3.i, &w3.r);
        for (int i0 = j; i0 < n; i0 += dist) {
            i1 = i0 + dist2;
            i2 = i1 + dist2;
            i3 = i2 + dist2;
            in0 = in[i0];
            in1 = in[i1];
            in2 = in[i2];
            in3 = in[i3];
            /*
            R1 = X[I] + X[I2];
            S1 = Y[I] + Y[I2];
            R2 = X[I1] + X[I3];
            S2 = Y[I1] + Y[I3];
            R3 = X[I] - X[I2];
            S3 = Y[I] - Y[I2];
            R4 = X[I1] - X[I3];
            S4 = Y[I1] - Y[I3];
            X[I] = R1 + R2;
            Y[I] = S1 + S2;
            R2 = R1 - R2;
            S2 = S1 - S2;

            R1 = R3 - S4;
            S1 = S3 + R4;
            R3 = R3 + S4;
            S3 = S3 - R4;
            X[I1] = CO1*R3 + SI1*S3;
            Y[I1] = CO1*S3 - SI1*R3;
            X[I2] = CO2*R2 + SI2*S2;
            Y[I2] = CO2*S2 - SI2*R2;
            X[I3] = CO3*R1 + SI3*S1;
            Y[I3] = CO3*S1 - SI3*R1;
            */
            tmp1 = cpxAdd(in0, in2);
            tmp2 = cpxAdd(in1, in3);
            tmp3 = cpxSub(in0, in2);
            tmp4 = cpxSub(in1, in3);
            tmp2 = cpxSub(tmp1, tmp2);
            tmp1 = cpxAdd(tmp3, cpxMul(tmp4, jmag));
            tmp3 = cpxAdd(tmp3, cpxMul(tmp4, jmag_neg));
            out[i0] = cpxAdd(cpxAdd(in0, in1), cpxAdd(in2, in3));                                    
            out[i1] = cpxMul(cpxAdd(in0, cpxAdd(cpxMul(in1, jmag_neg), cpxSub(cpxMul(in3, jmag), in2))), w1);
            out[i2] = cpxMul(cpxAdd(cpxAdd(in0, in1), cpxAdd(in2, in3)), w2);
            out[i3] = cpxMul(cpxAdd(in0, cpxAdd(cpxMul(in1, jmag), cpxSub(cpxMul(in3, jmag_neg), in2))), w3);

            // http://dsp.stackexchange.com/questions/3481/radix-4-fft-implementation
        }
    }
}
// (1 + i) * (0 + i) = -1 + i
// (1 + i) * (0 + -i) = 1 + -i