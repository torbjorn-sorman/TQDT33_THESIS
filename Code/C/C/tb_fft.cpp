#include "tb_fft.h"

#include "tb_transpose.h"

inline void fft(tb_cpx *x, tb_cpx *W, uint32_t start, uint32_t steps, uint32_t dist, uint32_t N);
inline void inner_fft(tb_cpx *x, tb_cpx *X, tb_cpx *W, int bit, uint32_t dist, uint32_t s2, uint32_t N);
inline void twiddle_factors(tb_cpx *W, double w_angle, uint32_t lead, uint32_t N);
inline void reverse_bit_order(tb_cpx *x, int dir, uint32_t n2, uint32_t N, uint32_t lead);

/* Naive Fast Fourier Transform, simple single core CPU-tests */
void tb_fft(int dir, tb_cpx *x, tb_cpx *X, uint32_t N)
{
    const uint32_t depth = log2_32(N);
    const uint32_t n2 = (N / 2);
    tb_cpx *W;
    uint32_t lead, start, dist;
    double w_angle;
    W = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    w_angle = (dir == FORWARD_FFT ? 1.0 : -1.0) * -(M_2_PI / N);
    lead = 32 - depth;

    twiddle_factors(W, w_angle, lead, N);
    start = 0;
    dist = N;
    if (x != X) {
        /* First iteration and also copy phase if not inplace */
        inner_fft(x, X, W, depth - 1, dist = n2, N, N);
        ++start;
    }
    fft(X, W, start, depth, dist, N);
    reverse_bit_order(X, dir, n2, N, lead);
    /* Free allocated resources */
    free(W);
}

/* Naive Fast Fourier Transform only Real Values */
void tb_fft_real(int dir, tb_cpx *x, tb_cpx *X, uint32_t N)
{
    const uint32_t depth = log2_32(N);
    const uint32_t n2 = (N / 2);
    tb_cpx *W;
    uint32_t lead, start, dist;
    double w_angle;
    W = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    w_angle = dir * (M_2_PI / (double)N);
    lead = 32 - depth;

    twiddle_factors(W, w_angle, lead, N);
    start = 0;
    dist = N;
    if (x != X) {
        /* First iteration and also copy phase if not inplace */
        inner_fft(x, X, W, depth - 1, dist = n2, N, N);
        ++start;
    }
    fft(X, W, start, depth, dist, N);
    reverse_bit_order(X, dir, n2, N, lead);
    /* Free allocated resources */
    free(W);
}

void tb_fft2d(int dir, void(*fn)(int, tb_cpx*, tb_cpx*, uint32_t), tb_cpx **seq2d, uint32_t N)
{
    uint32_t row, col;
    tb_cpx *seq, *out;
    seq = (tb_cpx *)malloc(N * sizeof(tb_cpx));
    out = (tb_cpx *)malloc(N * sizeof(tb_cpx));

    for (row = 0; row < N; ++row) {
        for (col = 0; col < N; ++col) {
            seq[col].r = seq2d[row][col].r;
            seq[col].i = seq2d[row][col].i;
        }
        fn(dir, seq, out, N);
        for (col = 0; col < N; ++col) {
            seq2d[row][col].r = out[col].r;
            seq2d[row][col].i = out[col].i;
        }
    }

    for (col = 0; col < N; ++col) {
        for (row = 0; row < N; ++row) {
            seq[row].r = seq2d[row][col].r;
            seq[row].i = seq2d[row][col].i;
        }
        fn(dir, seq, out, N);
        for (row = 0; row < N; ++row) {
            seq2d[row][col].r = out[row].r;
            seq2d[row][col].i = out[row].i;
        }
    }
    free(seq);
    free(out);
}

void tb_fft2d_inplace(int dir, void(*fn)(int, tb_cpx*, tb_cpx*, uint32_t), tb_cpx **seq2d, uint32_t N)
{
    uint32_t row;
    const uint32_t block_size = 16;

    for (row = 0; row < N; ++row)
        fn(dir, seq2d[row], seq2d[row], N);
    transpose_block(seq2d, N, block_size);

    for (row = 0; row < N; ++row)
        fn(dir, seq2d[row], seq2d[row], N);
    transpose_block(seq2d, N, block_size);
}

void tb_fft2d_trans(int dir, void(*fn)(int, tb_cpx*, tb_cpx*, uint32_t), tb_cpx **seq2d, uint32_t N)
{
    const uint32_t block_size = 16;
    uint32_t row, col;
    tb_cpx *seq, *out;
    seq = (tb_cpx *)malloc(N * sizeof(tb_cpx));
    out = (tb_cpx *)malloc(N * sizeof(tb_cpx));
    for (row = 0; row < N; ++row) {
        for (col = 0; col < N; ++col) {
            seq[col].r = seq2d[row][col].r;
            seq[col].i = seq2d[row][col].i;
        }
        fn(dir, seq, out, N);
        for (col = 0; col < N; ++col) {
            seq2d[row][col].r = out[col].r;
            seq2d[row][col].i = out[col].i;
        }
    }
    transpose_block(seq2d, N, block_size);
    for (row = 0; row < N; ++row) {
        for (col = 0; col < N; ++col) {
            seq[col].r = seq2d[row][col].r;
            seq[col].i = seq2d[row][col].i;
        }
        fn(dir, seq, out, N);
        for (col = 0; col < N; ++col) {
            seq2d[row][col].r = out[col].r;
            seq2d[row][col].i = out[col].i;
        }
    }
    transpose_block(seq2d, N, block_size);
    free(seq);
    free(out);
}

void tb_dft_naive(int dir, tb_cpx *x, tb_cpx *X, uint32_t N)
{
    uint32_t k, n;
    double real, img, re, im, theta, c1, c2;
    theta = 1.0;
    c1 = (-M_2_PI / N);
    c2 = 1.0;
    for (k = 0; k < N; ++k)
    {
        real = 0.0;
        img = 0.0;
        c2 = c1 * k;
        for (n = 0; n < N; ++n)
        {
            theta = c2 * n;
            re = cos(theta);
            im = sin(theta);
            real += x[n].r * re + x[n].i * im;
            img += x[n].r * im + x[n].i * re;
        }
        x[k].r = (float)real;
        x[k].i = (float)img;
    }
}

inline void fft(tb_cpx *x, tb_cpx *W, uint32_t start, uint32_t steps, uint32_t dist, uint32_t N)
{
    static int bit;
    static uint32_t s2;
    for (bit = steps - 1 - start; bit >= 0; --bit) {
        s2 = dist;
        dist = dist >> 1;
        inner_fft(x, x, W, bit, dist, s2, N);
    }
}

inline void inner_fft(tb_cpx *x, tb_cpx *X, tb_cpx *W, int bit, uint32_t dist, uint32_t s2, uint32_t N)
{
    static uint32_t m, n, l, u, p;
    static tb_cpx tl, tu;
    for (m = 0; m < N; m += s2) {
        n = dist + m;
        for (l = m; l < n; ++l) {
            u = l + dist;
            tl = x[l];
            tu = x[u];
            p = (l >> bit);
            X[l].r = tl.r + W[p].r * tu.r - W[p].i * tu.i;
            X[l].i = tl.i + W[p].r * tu.i + W[p].i * tu.r;
            p = (u >> bit);
            X[u].r = tl.r + W[p].r * tu.r - W[p].i * tu.i;
            X[u].i = tl.i + W[p].r * tu.i + W[p].i * tu.r;
        }
    }
}

inline void twiddle_factors(tb_cpx *W, double w_angle, uint32_t lead, uint32_t N)
{
    static uint32_t n;
    static double ang;
    for (n = 0; n < N; ++n) {
        ang = w_angle * reverseBits(n, lead);
        W[n].r = cos(ang);
        W[n].i = sin(ang);
    }
}

inline void reverse_bit_order(tb_cpx *x, int dir, uint32_t n2, uint32_t N, uint32_t lead)
{
    static uint32_t n, p;
    static tb_cpx tmp_cpx;
    for (n = 0; n <= n2; ++n) {
        p = reverseBits(n, lead);
        if (n < p) {
            tmp_cpx = x[n];
            x[n] = x[p];
            x[p] = tmp_cpx;
        }
    }
    if (dir == INVERSE_FFT && 0) {
        for (n = 0; n < N; ++n) {
            x[n].r = x[n].r / (float)N;
            x[n].i = x[n].i / (float)N;
        }
    }
}