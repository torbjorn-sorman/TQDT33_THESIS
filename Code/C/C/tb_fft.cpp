#include "tb_fft.h"

#include <omp.h>

#include "tb_math.h"
#include "tb_transpose.h"
#include "tb_print.h"

void fft(tb_cpx *x, tb_cpx *W, int start, int steps, int dist, const int n);
void inner_fft(tb_cpx *x, tb_cpx *X, tb_cpx *W, int bit, int dist, int s2, const int n);
void inner_fft2(tb_cpx *x, tb_cpx *X, tb_cpx *W, int bit, int dist, int s2, const int N2);
void twiddle_factors(tb_cpx *W, double w_angle, int lead, const int n);
void bit_reverse(tb_cpx *x, double dir, const int n, int lead);

void tb_fft_old(double dir, tb_cpx *x, tb_cpx *X, const int n)
{
    int bit = log2_32(n);
    const int n2 = (n / 2);
    int lead, dist, dist_2;
    tb_cpx *W;
    W = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    lead = 32 - bit;
    twiddle_factors(W, dir * (M_2_PI / n), lead, n);

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

/* Naive Fast Fourier Transform, simple single core CPU-tests */
void tb_fft(double dir, tb_cpx *x, tb_cpx *X, const int n)
{
    int bit = log2_32(n);
    const int n2 = (n / 2);
    tb_cpx *W;
    int lead, dist, dist2;
    W = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    lead = 32 - bit;
    twiddle_factors(W, dir * M_2_PI / (double)n, lead, n);

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

/* Naive Fast Fourier Transform only Real Values */
void tb_fft_real(double dir, tb_cpx *x, tb_cpx *X, const int n)
{
    int bit = log2_32(n);
    const int n2 = (n / 2);
    tb_cpx *W;
    int lead, dist, dist2;
    W = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    lead = 32 - bit;
    twiddle_factors(W, dir * (M_2_PI / (double)n), lead, n);

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

void tb_fft2d(double dir, fft_function fn, tb_cpx **seq2d, const int n)
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

void tb_fft2d_openmp(double dir, fft_function fn, tb_cpx **seq2d, const int n)
{
    int row;
    const int block_size = 16;

#pragma omp parallel for private(row) shared(n, fn, dir, seq2d)
    for (row = 0; row < n; ++row) {
        fn(dir, seq2d[row], seq2d[row], n);
    }
    transpose_openmp(seq2d, n, block_size);

#pragma omp parallel for private(row) shared(n, fn, dir, seq2d)
    for (row = 0; row < n; ++row) {
        fn(dir, seq2d[row], seq2d[row], n);
    }
    transpose_openmp(seq2d, n, block_size);
}

void tb_dft_naive(double dir, tb_cpx *x, tb_cpx *X, const int n)
{
    int k, i;
    double real, img, re, im, theta, c1, c2;
    theta = 1.0;
    c1 = (-M_2_PI / n);
    c2 = 1.0;
    for (k = 0; k < n; ++k)
    {
        real = 0.0;
        img = 0.0;
        c2 = c1 * k;
        for (i = 0; i < n; ++i)
        {
            theta = c2 * i;
            re = cos(theta);
            im = sin(theta);
            real += x[i].r * re + x[i].i * im;
            img += x[i].r * im + x[i].i * re;
        }
        x[k].r = (float)real;
        x[k].i = (float)img;
    }
}

void tb_fft_openmp(double dir, tb_cpx *x, tb_cpx *X, const int n)
{
    int bit, i, l, num_of_threads, start, *offset, *tmp;
    const int n2 = (n / 2);
    tb_cpx *W, tl, tu;
    int lead, dist, dist2, end, u, p;
    double ang, wAngle;

    num_of_threads = omp_get_max_threads();
    num_of_threads = num_of_threads > n2 ? n2 : num_of_threads;
    tmp = (int *)malloc(sizeof(int) * 4);
    offset = (int *)malloc(sizeof(int) * 4);
    bit = log2_32(n);
    W = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    lead = 32 - bit;

    wAngle = dir * (M_2_PI / (double)n);
    // ----- parallel section START -----
#pragma omp parallel for private(i, ang) shared(wAngle, lead, W, n)
    for (i = 0; i < n; ++i) {
        ang = wAngle * reverseBits(i, lead);
        W[i].r = (float)cos(ang);
        W[i].i = (float)sin(ang);
    }
    // ----- parallel section  END  -----
    dist2 = n;
    dist = n2;
    --bit;
    
    // ----- parallel section START -----
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
    }
    // ----- parallel section  END  -----
    while (bit-- > 0)
    {
        dist2 = dist;
        dist = dist >> 1;
        // ----- parallel section START -----        
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
        }
        //*/
        /*
        for (n = 0; n < num_of_threads; ++n) {                         
            offset[n] = ((n * (n2 / num_of_threads)) / dist2) * dist2;
            tmp[n] = dist + offset[n];
        }
#pragma omp parallel for private(n, l, u, p, tl, tu, end, tid) shared(X, W, dist, dist2, bit, tmp, offset, num_of_threads)
        for (n = 0; n < n2; ++n)
        {
            tid = omp_get_thread_num();
            if (n >= tmp[tid]) {
                offset[tid] += dist2;
                tmp[tid] += dist2;
            }            
            //offset[tid] += (n >= (dist + offset[tid])) * dist2;
            l = (n & ~(1 << bit)) + offset[tid];
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
        //*/
        // ----- parallel section  END  -----
    }
    // ----- parallel section START -----
#pragma omp parallel for private(i, p, tl) shared(lead, X, n)
    for (i = 0; i <= n; ++i) {
        p = reverseBits(i, lead);
        if (i < p) {
            tl = X[i];
            X[i] = X[p];
            X[p] = tl;
        }
    }
    // ----- parallel section  END  -----
    if (dir == INVERSE_FFT) {
        // ----- parallel section START -----
#pragma omp parallel for private(i) shared(X, n)
        for (i = 0; i < n; ++i) {
            X[i].r = X[i].r / (float)n;
            X[i].i = X[i].i / (float)n;
        }
        // ----- parallel section  END  -----
    }
    free(tmp);
    free(offset);
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

void twiddle_factors(tb_cpx *W, double w_angle, int lead, const int n)
{
    int i;
    double ang;
    for (i = 0; i < n; ++i) {
        ang = w_angle * reverseBits(i, lead);
        W[i].r = (float)cos(ang);
        W[i].i = (float)sin(ang);
    }
}

void bit_reverse(tb_cpx *x, double dir, const int n, int lead)
{
    int i, p;
    tb_cpx tmp_cpx;
    for (i = 0; i <= n; ++i) {
        p = reverseBits(i, lead);
        if (i < p) {
            tmp_cpx = x[i];
            x[i] = x[p];
            x[p] = tmp_cpx;
        }
    }
    if (dir == INVERSE_FFT) {
        for (i = 0; i < n; ++i) {
            x[i].r = x[i].r / (float)n;
            x[i].i = x[i].i / (float)n;
        }
    }
}