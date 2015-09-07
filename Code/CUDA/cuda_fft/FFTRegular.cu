#include <stdio.h>
#include <device_launch_parameters.h>

#include "math.h"
#include "FFTRegular.cuh"
#include "fft_helper.cuh"

__global__ void _FFTBody(cuFloatComplex *in, cuFloatComplex *out, cuFloatComplex *W, const int dist, const int dist2, const int n2);
__host__ __inline void _setBlocksAndThreads(int *numBlocks, int *threadsPerBlock, const int size);

__host__ void FFTRegular(fftDirection dir, cuFloatComplex *dev_in, cuFloatComplex *dev_out, cuFloatComplex *dev_W, unsigned int *buf, const int n)
{
    int dist, dist2, threadsPerBlock, numBlocks;
    const int n2 = (n / 2);
    const float w_angle = dir * (M_2_PI / n);
    const float scale = dir == FFT_FORWARD ? 1.f : 1.f / n;
    dist2 = n;
    dist = n2;

    _setBlocksAndThreads(&numBlocks, &threadsPerBlock, n2);
    twiddle_factors << < numBlocks, threadsPerBlock >> >(dev_W, w_angle, n);
    cudaDeviceSynchronize();
    
    _FFTBody << < numBlocks, threadsPerBlock >> >(dev_in, dev_out, dev_W, dist, dist2, n);
    cudaDeviceSynchronize();
    while ((dist2 = dist) > 1) {
        dist = dist >> 1;
        _FFTBody << < numBlocks, threadsPerBlock >> >(dev_out, dev_out, dev_W, dist, dist2, n);
        cudaDeviceSynchronize();
    }

    _setBlocksAndThreads(&numBlocks, &threadsPerBlock, n);
    bit_reverse << < numBlocks, threadsPerBlock >> >(dev_out, dev_in, scale, 32 - log2_32(n));
    cudaDeviceSynchronize();

    *buf = 0;
}

__global__ void _FFTBody(cuFloatComplex *in, cuFloatComplex *out, cuFloatComplex *W, const int dist, const int dist2, const int n2)
{
}

/*

__inline void _fft_inner_body(cuFloatComplex *in, cuFloatComplex *out, const cuFloatComplex *W, const int lower, const int upper, const int dist, const int mul)
{
    int u, p;
    float tmp_r, tmp_i;
    for (int l = lower; l < upper; ++l) {
        u = l + dist;
        p = (l - lower) * mul;
        tmp_r = in[l].r - in[u].r;
        tmp_i = in[l].i - in[u].i;
        out[l].r = in[l].r + in[u].r;
        out[l].i = in[l].i + in[u].i;
        out[u].r = (W[p].r * tmp_r) - (W[p].i * tmp_i);
        out[u].i = (W[p].i * tmp_r) + (W[p].r * tmp_i);
    }
}

__inline void _fft_body(cuFloatComplex *in, cuFloatComplex *out, cuFloatComplex *W, int dist, int dist2, const int n_threads, const int n)
{
    const int count = n / dist2;
#ifdef _OPENMP        
    if (count >= n_threads) {
#pragma omp parallel for schedule(static)              
        for (int lower = 0; lower < n; lower += dist2) {
            _fft_inner_body(in, out, W, lower, dist + lower, dist, count);
        }
    }
    else
    {
        int u, p, upper;
        float tmp_r, tmp_i;
        for (int lower = 0; lower < n; lower += dist2) {
            upper = dist + lower;
#pragma omp parallel for schedule(static) private(u, p, tmp_r, tmp_i)
            for (int l = lower; l < upper; ++l) {
                u = l + dist;
                p = (l - lower) * count;
                tmp_r = in[l].r - in[u].r;
                tmp_i = in[l].i - in[u].i;
                out[l].r = in[l].r + in[u].r;
                out[l].i = in[l].i + in[u].i;
                out[u].r = (W[p].r * tmp_r) - (W[p].i * tmp_i);
                out[u].i = (W[p].i * tmp_r) + (W[p].r * tmp_i);
            }
        }
    }
#else
    for (int lower = 0; lower < n; lower += dist2) {
        fft_inner_body(in, out, W, lower, dist + lower, dist, count);
    }
#endif
}

*/

__host__ __inline void _setBlocksAndThreads(int *numBlocks, int *threadsPerBlock, const int size)
{
    int v1 = 256;
    if (size > v1) {
        *numBlocks = size / v1;
        *threadsPerBlock = v1;
    }
    else {
        *numBlocks = 1;
        *threadsPerBlock = size;
    }
}

__host__ __inline void _swap(cuFloatComplex **in, cuFloatComplex **out)
{
    cuFloatComplex *tmp = *in;
    *in = *out;
    *out = tmp;
}