#include "tb_fft.h"

#include <omp.h>

#include "tb_math.h"
#include "tb_transpose.h"
#include "tb_print.h"
#include "tb_fft_helper.h"

void fft(tb_cpx *x, tb_cpx *W, int start, int steps, int dist, const int n);
void inner_fft(tb_cpx *x, tb_cpx *X, tb_cpx *W, int bit, int dist, int s2, const int n);
void inner_fft2(tb_cpx *x, tb_cpx *X, tb_cpx *W, int bit, int dist, int s2, const int N2);

void tb_fft(const double dir, tb_cpx *x, tb_cpx *X, const int n)
{
    int bit = log2_32(n);
    const int n2 = (n / 2);
    tb_cpx *W;
    int lead, dist, dist2;
    W = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    lead = 32 - bit;
    twiddle_factors(W, dir, lead, n);

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
    free(W);
}

void tb_fft_old(const double dir, tb_cpx *x, tb_cpx *X, const int n)
{
    int bit = log2_32(n);
    const int n2 = (n / 2);
    int lead, dist, dist_2;
    tb_cpx *W;
    W = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    lead = 32 - bit;
    twiddle_factors(W, dir, lead, n);

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
    free(W);
}

void tb_fft_real(const double dir, tb_cpx *x, tb_cpx *X, const int n)
{
    int bit = log2_32(n);
    const int n2 = (n / 2);
    tb_cpx *W;
    int lead, dist, dist2;
    W = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    lead = 32 - bit;
    twiddle_factors(W, dir, lead, n);

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
    free(W);
}

void tb_fft_openmp(const double dir, tb_cpx *x, tb_cpx *X, const int n)
{
    int bit, i, l, start;
    const int n2 = (n / 2);
    tb_cpx *W, tl, tu;
    int lead, dist, dist2, end, u, p;
    double ang, wAngle;

    bit = log2_32(n);
    W = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    lead = 32 - bit;
    twiddle_factors_omp(W, dir, lead, n);
    dist2 = n;
    dist = n2;
    --bit;
#pragma omp parallel for private(i, l, u, p, tl, tu) shared(x, X, W, n2, dist, bit)
    for (i = 0; i < n2; ++i)
    {
        l = (i & ~(1 << bit));
        u = l + dist;
        tl = x[l];
        tu = x[u];
        p = (l >> bit);
        X[l].r = tl.r + W[p].r * tu.r - W[p].i * tu.i;
        X[l].i = tl.i + W[p].r * tu.i + W[p].i * tu.r;
        p = (u >> bit);
        X[u].r = tl.r + W[p].r * tu.r - W[p].i * tu.i;
        X[u].i = tl.i + W[p].r * tu.i + W[p].i * tu.r;
    } // ----- parallel section  END  -----
    while (bit-- > 0)
    {
        dist2 = dist;
        dist = dist >> 1;    
#pragma omp parallel for private(start, l, u, p, tl, tu, end) shared(X, W, dist, dist2, bit, n)
        for (start = 0; start < n; start += dist2) {
            end = dist + start;
            for (l = start; l < end; ++l) {
                u = l + dist;
                tl = X[l];
                tu = X[u];
                p = (l >> bit);
                X[l].r = tl.r + W[p].r * tu.r - W[p].i * tu.i;
                X[l].i = tl.i + W[p].r * tu.i + W[p].i * tu.r;
                p = (u >> bit);
                X[u].r = tl.r + W[p].r * tu.r - W[p].i * tu.i;
                X[u].i = tl.i + W[p].r * tu.i + W[p].i * tu.r;
            }
        } // ----- parallel section  END  -----
    }
    bit_reverse_omp(X, dir, n, lead);
    free(W);
}


void tb_fft2d(const double dir, fft_function fn, tb_cpx **seq2d, const int n)
{
    int row;
    const int block_size = 16;

    for (row = 0; row < n; ++row)
        fn(dir, seq2d[row], seq2d[row], n);
    transpose_block(seq2d, n, block_size);

    for (row = 0; row < n; ++row)
        fn(dir, seq2d[row], seq2d[row], n);
    transpose_block(seq2d, n, block_size);
}

void tb_fft2d_openmp(const double dir, fft_function fn, tb_cpx **seq2d, const int n)
{
    const int block_size = 8;
    int row, col;
    tb_cpx tmp;

#pragma omp parallel private(row, col, tmp) shared(block_size, dir, fn, seq2d, n) 
    {
#pragma omp for
        for (row = 0; row < n; ++row) {
            fn(dir, seq2d[row], seq2d[row], n);
        }
#pragma omp barrier
#pragma omp for
        for (row = 0; row < n; ++row) {
            for (col = row + 1; col < n; ++col) {
                tmp = seq2d[row][col];
                seq2d[row][col] = seq2d[col][row];
                seq2d[col][row] = tmp;
            }
        }
#pragma omp barrier
#pragma omp for
        for (row = 0; row < n; ++row) {
            fn(dir, seq2d[row], seq2d[row], n);
        }
#pragma omp barrier
#pragma omp for
        for (row = 0; row < n; ++row) {
            for (col = row + 1; col < n; ++col) {
                tmp = seq2d[row][col];
                seq2d[row][col] = seq2d[col][row];
                seq2d[col][row] = tmp;
            }
        }
    }
}

void tb_fft2d_openmp_alt(const double dir, fft_function fn, tb_cpx **seq2d, const int n)
{
    const int block_size = 8;
    int row, col;
    tb_cpx tmp;
#pragma omp parallel for private(row) shared(dir, fn, seq2d, n)
    for (row = 0; row < n; ++row) {
        fn(dir, seq2d[row], seq2d[row], n);
    }
#pragma omp parallel for private(row, col, tmp) shared(n, seq2d)
    for (row = 0; row < n; ++row) {
        for (col = row + 1; col < n; ++col) {
            tmp = seq2d[row][col];
            seq2d[row][col] = seq2d[col][row];
            seq2d[col][row] = tmp;
        }
    }
#pragma omp parallel for private(row) shared(dir, fn, seq2d, n)
    for (row = 0; row < n; ++row) {
        fn(dir, seq2d[row], seq2d[row], n);
    }
#pragma omp parallel for private(row, col, tmp) shared(n, seq2d)
    for (row = 0; row < n; ++row) {
        for (col = row + 1; col < n; ++col) {
            tmp = seq2d[row][col];
            seq2d[row][col] = seq2d[col][row];
            seq2d[col][row] = tmp;
        }
    }
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