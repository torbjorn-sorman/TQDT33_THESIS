#include "tb_fft.h"

#include <omp.h>

#include "tb_math.h"
#include "tb_transpose.h"
#include "tb_print.h"
#include "tb_fft_helper.h"

void inner_fft(tb_cpx *x, tb_cpx *X, tb_cpx *W, int bit, int dist, int s2, const int n);
void inner_fft_omp(tb_cpx *x, tb_cpx *X, tb_cpx *W, int bit, int dist, int dist2, const int n);
void inner_fft2(tb_cpx *x, tb_cpx *X, tb_cpx *W, int bit, int dist, int s2, const int N2);

void tb_fft(const double dir, tb_cpx *x, tb_cpx *X, tb_cpx *W, const int n)
{
    int bit = log2_32(n);
    const int n2 = (n / 2);
    int lead, dist, dist2;
    lead = 32 - bit;
    dist2 = n;
    dist = n2;
    --bit;
    inner_fft(x, X, W, bit, dist, dist2, n);
    while (bit-- > 0)
    {
        dist2 = dist;
        dist = dist >> 1;
        inner_fft(X, X, W, bit, dist, dist2, n);
    }
    bit_reverse(X, dir, n, lead);
}

void tb_fft_alt(const double dir, tb_cpx *x, tb_cpx *X, tb_cpx *W, const int n)
{
    int bit = log2_32(n);
    const int n2 = (n / 2);
    int lead, dist, dist_2;
    lead = 32 - bit;
    dist_2 = n;
    dist = n2;
    inner_fft2(x, X, W, --bit, dist, dist_2, n2);
    while (bit-- > 0)
    {
        dist_2 = dist;
        dist = dist >> 1;
        inner_fft2(X, X, W, bit, dist, dist_2, n2);
    }
    bit_reverse(X, dir, n, lead);
}

void tb_fft_real(const double dir, tb_cpx *x, tb_cpx *X, tb_cpx *W, const int n)
{
    int bit = log2_32(n);
    const int n2 = (n / 2);
    int lead, dist, dist2;
    lead = 32 - bit;
    dist2 = n;
    dist = n2;
    inner_fft(x, X, W, --bit, dist, dist2, n);
    while (bit-- > 0)
    {
        dist2 = dist;
        dist = dist >> 1;
        inner_fft(X, X, W, bit, dist, dist2, n);
    }
    bit_reverse(X, dir, n, lead);
}

void tb_fft_omp(const double dir, tb_cpx *x, tb_cpx *X, tb_cpx *W, const int n)
{
    const int n2 = (n / 2);
    int bit, lead, dist, dist2;
    bit = log2_32(n);
    lead = 32 - bit;
    dist2 = n;
    dist = n2;
    --bit;
    inner_fft_omp(x, X, W, bit, dist, dist2, n2);
    while (bit-- > 0)
    {
        dist2 = dist;
        dist = dist >> 1;
        inner_fft_omp(X, X, W, bit, dist, dist2, n);
    }
    bit_reverse_omp(X, dir, n, lead);
}


void tb_fft2d(const double dir, fft_function fn, tb_cpx **seq2d, const int n)
{
    int row;
    const int block_size = n < 16 ? n : n / 2;
    tb_cpx *W;
    W = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    twiddle_factors(W, 32 - log2_32(n), n);
    if (dir == INVERSE_FFT)
        twiddle_factors_inverse_omp(W, n);

    for (row = 0; row < n; ++row)
        fn(dir, seq2d[row], seq2d[row], W, n);
    transpose_block(seq2d, block_size, n);    

    for (row = 0; row < n; ++row)
        fn(dir, seq2d[row], seq2d[row], W, n);
    transpose_block(seq2d, block_size, n);

    if (W != NULL)
        free(W);
}

void tb_fft2d_omp(const double dir, fft_function fn, tb_cpx **seq2d, const int n)
{
    int row, col;
    tb_cpx tmp, *W;
    W = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    twiddle_factors_omp(W, 32 - log2_32(n), n);
    if (dir == INVERSE_FFT)
        twiddle_factors_inverse_omp(W, n);

#pragma omp parallel for private(row) shared(W, dir, fn, seq2d, n)
    for (row = 0; row < n; ++row) {
        fn(dir, seq2d[row], seq2d[row], W, n);
    }
#pragma omp parallel for private(row, col, tmp) shared(n, seq2d)
    for (row = 0; row < n; ++row) {
        for (col = row + 1; col < n; ++col) {
            tmp = seq2d[row][col];
            seq2d[row][col] = seq2d[col][row];
            seq2d[col][row] = tmp;
        }
    }
#pragma omp parallel for private(row) shared(W, dir, fn, seq2d, n)
    for (row = 0; row < n; ++row) {
        fn(dir, seq2d[row], seq2d[row], W, n);
    }
#pragma omp parallel for private(row, col, tmp) shared(n, seq2d)
    for (row = 0; row < n; ++row) {
        for (col = row + 1; col < n; ++col) {
            tmp = seq2d[row][col];
            seq2d[row][col] = seq2d[col][row];
            seq2d[col][row] = tmp;
        }
    }
    if (W != NULL)
        free(W);
}

void inner_fft(tb_cpx *x, tb_cpx *X, tb_cpx *W, int bit, int dist, int s2, const int n)
{
    int start, end, l, u, p;
    tb_cpx tl, tu;
    for (start = 0; start < n; start += s2) {
        end = dist + start;
        for (l = start; l < end; ++l) {
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

void inner_fft_omp(tb_cpx *x, tb_cpx *X, tb_cpx *W, int bit, int dist, int dist2, const int n)
{
    int start, end, l, u, p;
    tb_cpx tl, tu;
#pragma omp parallel for private(start, end, l, u, p, tl, tu) shared(x, X, W, bit, dist, dist2, n)
    for (start = 0; start < n; start += dist2) {
        end = dist + start;
        for (l = start; l < end; ++l) {
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

void inner_fft2(tb_cpx *x, tb_cpx *X, tb_cpx *W, int bit, int dist, int s2, const int n2)
{
    int n, l, u, p, offset, tmp;
    tb_cpx tl, tu;
    offset = 0;
    tmp = dist;
    for (n = 0; n < n2; ++n)
    {
        if (n >= tmp) {
            offset += s2;
            tmp += s2;
        }
        //offset += (n >= (dist + offset)) * dist_2;
        l = (n & ~(1 << bit)) + offset;
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