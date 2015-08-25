#include "tb_fft.h"

// Saves space... for readability
/*              
    u = l + dist;
    tmp_l = X[l];
    tmp_u = X[u];
    p = (l >> bit);
    X[l].r = tmp_l.r + W[p].r * tmp_u.r - W[p].i * tmp_u.i;
    X[l].i = tmp_l.i + W[p].r * tmp_u.i + W[p].i * tmp_u.r;
    p = (u >> bit);
    X[u].r = tmp_l.r + W[p].r * tmp_u.r - W[p].i * tmp_u.i;
    X[u].i = tmp_l.i + W[p].r * tmp_u.i + W[p].i * tmp_u.r;
*/

#define INNER_FFT(A,B) u = l + dist; tmp_l = A[l]; tmp_u = A[u]; p = (l >> bit); B[l].r = tmp_l.r + W[p].r * tmp_u.r - W[p].i * tmp_u.i; B[l].i = tmp_l.i + W[p].r * tmp_u.i + W[p].i * tmp_u.r; p = (u >> bit); B[u].r = tmp_l.r + W[p].r * tmp_u.r - W[p].i * tmp_u.i; B[u].i = tmp_l.i + W[p].r * tmp_u.i + W[p].i * tmp_u.r

void reverse_bit_order(my_complex *x, int dir, uint32_t N, uint32_t lead)
{
    uint32_t n, p;
    float tmp;
    my_complex tmp_cpx;
    if (dir == INVERSE_FFT) {
        for (n = 0; n < N / 2; ++n) {
            p = reverseBits(n, lead);
            if (n < p) {
                tmp = x[p].r / N;
                x[p].r = x[n].r / N;
                x[n].r = tmp;
                tmp = x[p].i / N;
                x[p].i = x[n].i / N;
                x[n].i = tmp;
            }
        }
    }
    else {
        for (n = 0; n < N / 2; ++n) {
            p = reverseBits(n, lead);
            if (n < p) {
                tmp_cpx = x[n];
                x[n] = x[p];
                x[p] = tmp_cpx;
            }
        }
    }
}

/* Naive Fast Fourier Transform, simple single core CPU-tests */
void tb_fft(int dir, my_complex *x, my_complex *X, uint32_t N)
{
    const uint32_t depth = log2_32(N);
    const uint32_t n2 = (N / 2);
    my_complex *W = (my_complex *)malloc(sizeof(my_complex) * N);
    my_complex tmp_u, tmp_l;
    uint32_t lead, dist, dist_2, l, m, n, p, u;
    int bit;
    float ang;
    double w_angle;

    w_angle = (dir == FORWARD_FFT ? 1.0 : -1.0) * -(M_2_PI / N);
    lead = 32 - depth;
    for (uint32_t n = 0; n < N; ++n)
    {
        ang = w_angle * reverseBits(n, lead); // Potential source of errors...
        W[n].r = cos(ang);
        W[n].i = sin(ang);
    }
    /* First iteration and also copy phase */
    bit = depth - 1;
    dist_2 = N;
    dist = n2;
    for (m = 0; m < N; m += dist_2)
    {
        n = dist + m;
        for (l = m; l < n; ++l)
        {
            INNER_FFT(x, X);
        }
    }
    /* Iteration 2 and further */
    for (bit = depth - 2; bit >= 0; --bit)
    {
        dist_2 = dist;
        dist = dist >> 1;
        for (m = 0; m < N; m += dist_2)
        {
            n = dist + m;
            for (l = m; l < n; ++l)
            {
                INNER_FFT(X, X);
            }
        }
    }
    reverse_bit_order(X, dir, N, lead);
    free(W);
}

/* Naive Fast Fourier Transform, simple single core CPU-tests */
void tb_fft_inplace(int dir, my_complex *x, my_complex *X, uint32_t N)
{
    const uint32_t depth = log2_32(N);
    const uint32_t n2 = (N / 2);
    my_complex *W = (my_complex *)malloc(sizeof(my_complex) * N);
    my_complex tmp_u, tmp_l;
    uint32_t lead, bit, u, l, p, dist, dist_2, k, m, n;
    float ang, w_angle;
    w_angle = (dir == FORWARD_FFT ? 1.0 : -1.0) * -(M_2_PI / N);
    lead = 32 - depth;
    bit = 0;
    for (uint32_t n = 0; n < N; ++n)
    {
        ang = w_angle * reverseBits(n, lead);
        W[n].r = cos(ang);
        W[n].i = sin(ang);
    }
    dist = N;
    for (k = 0; k < depth; ++k)
    {
        bit = depth - 1 - k;
        dist_2 = dist;
        dist = dist >> 1;
        for (m = 0; m < N; m += dist_2)
        {
            n = dist + m;
            for (l = m; l < n; ++l)
            {
                INNER_FFT(x, x);
            }
        }
    }
    reverse_bit_order(x, dir, N, lead);
    free(W);
}



/* Naive Fast Fourier Transform only Real Values */
void tb_fft_test(int dir, my_complex *x, my_complex *X, uint32_t N)
{
    const uint32_t depth = log2_32(N);
    const uint32_t n2 = (N / 2);
    my_complex *W = (my_complex *)malloc(sizeof(my_complex) * N);
    my_complex tmp_u, tmp_l;
    uint32_t lead, bit, u, l, p, dist, dist_2, k, m, n;
    float ang, w_angle;

    w_angle = (dir == FORWARD_FFT ? 1.f : -1.f) * -(M_2_PI / N);
    lead = 32 - depth;
    for (uint32_t n = 0; n < N; ++n)
    {
        ang = w_angle * reverseBits(n, lead);
        W[n].r = cos(ang);
        W[n].i = sin(ang);
    }
    dist = N;
    for (k = 0; k < depth; ++k)
    {
        bit = depth - 1 - k;
        dist_2 = dist;
        dist = dist >> 1;
        for (m = 0; m < N; m += dist_2)
        {
            n = dist + m;
            for (l = m; l < n; ++l)
            {
                INNER_FFT(x, x);
                /*
                u = l + dist;
                tmp_l = x[l];
                tmp_u = x[u];
                p = (l >> bit);
                x[l].r = tmp_l.r + W[p].r * tmp_u.r - W[p].i * tmp_u.i;
                x[l].i = tmp_l.i + W[p].r * tmp_u.i + W[p].i * tmp_u.r;
                p = (u >> bit);
                x[u].r = tmp_l.r + W[p].r * tmp_u.r - W[p].i * tmp_u.i;
                x[u].i = tmp_l.i + W[p].r * tmp_u.i + W[p].i * tmp_u.r;
                */
            }
        }
    }
    reverse_bit_order(x, dir, N, lead);
    free(W);
}

void transpose_maybe(my_complex **seq, uint32_t N)
{
    uint32_t x, y;
    for (y = 0; y < N; ++y)
        for (x = y + 1; x < N; ++x)
            seq[y][x] = seq[x][y];
}

void tb_fft2d(int dir, void(*fn)(int, my_complex*, my_complex*, uint32_t), my_complex **seq2d, uint32_t N)
{
    const uint32_t depth = log2_32(N);
    uint32_t row, col;
    my_complex *seq, *out;

    seq = (my_complex *)malloc(N * sizeof(my_complex));
    out = (my_complex *)malloc(N * sizeof(my_complex));

    for (row = 0; row < N; ++row)
    {
        for (col = 0; col < N; ++col)
        {
            seq[col].r = seq2d[row][col].r;
            seq[col].i = seq2d[row][col].i;
        }

        fn(dir, seq, out, N);

        for (col = 0; col < N; ++col)
        {
            seq2d[row][col].r = out[col].r;
            seq2d[row][col].i = out[col].i;
        }
    }
    // TODO: Do transpose!    
    for (col = 0; col < N; ++col)
    {
        for (row = 0; row < N; ++row)
        {
            seq[row].r = seq2d[row][col].r;
            seq[row].i = seq2d[row][col].i;
        }

        fn(dir, seq, out, N);

        for (row = 0; row < N; ++row)
        {
            seq2d[row][col].r = out[row].r;
            seq2d[row][col].i = out[row].i;
        }
    }
    // TODO: Transpose back!
    free(seq);
    free(out);
}

void tb_fft2d_inplace(int dir, void(*fn)(int, my_complex*, my_complex*, uint32_t), my_complex **seq2d, uint32_t N)
{
    const uint32_t depth = log2_32(N);
    uint32_t row, col;
    my_complex *seq, *out;

    seq = (my_complex *)malloc(N * sizeof(my_complex));
    out = (my_complex *)malloc(N * sizeof(my_complex));

    for (row = 0; row < N; ++row)
    {
        for (col = 0; col < N; ++col)
        {
            seq[col].r = seq2d[row][col].r;
            seq[col].i = seq2d[row][col].i;
        }

        fn(dir, seq, out, N);

        for (col = 0; col < N; ++col)
        {
            seq2d[row][col].r = out[col].r;
            seq2d[row][col].i = out[col].i;
        }
    }
    // TODO: Do transpose!    
    for (col = 0; col < N; ++col)
    {
        for (row = 0; row < N; ++row)
        {
            seq[row].r = seq2d[row][col].r;
            seq[row].i = seq2d[row][col].i;
        }

        fn(dir, seq, out, N);

        for (row = 0; row < N; ++row)
        {
            seq2d[row][col].r = out[row].r;
            seq2d[row][col].i = out[row].i;
        }
    }
    // TODO: Transpose back!
    free(seq);
    free(out);
}

/* Naive Discrete Fourier Transform, essentially as per definition */
void tb_dft_naive(my_complex *x, my_complex *X, uint32_t N)
{
    float real, img;
    my_complex y = { 0.0, 0.0 };
    float re, im;
    my_complex tmp = { 0.0, 0.0 };
    float theta = 1.0;
    float c1 = -M_2_PI / N;
    float c2 = 1.0;
    for (uint32_t k = 0; k < N; ++k)
    {
        real = 0.0;
        img = 0.0;
        c2 = c1 * k;
        for (uint32_t n = 0; n < N; ++n)
        {
            theta = c2 * n;
            re = cos(theta);
            im = sin(theta);
            real += x[n].r * re + x[n].i * im;
            img += x[n].r * im + x[n].i * re;
        }
        x[k].r = real;
        x[k].i = img;
    }
}