#include <stdio.h>
#include <device_launch_parameters.h>

#include "math.h"
#include "FFTTobb.cuh"
#include "fft_helper.cuh"

__global__ void _FFTBody(cuFloatComplex *in, cuFloatComplex *out, cuFloatComplex *W, const unsigned int lmask, const unsigned int pmask, const int steps, const int dist, const int dist2, const int n2);
__host__ __inline void _setBlocksAndThreads(int *numBlocks, int *threadsPerBlock, const int size);
__host__ __inline void _swap(cuFloatComplex **in, cuFloatComplex **out);

__host__ void FFTTobb(fftDirection dir, cuFloatComplex *dev_in, cuFloatComplex *dev_out, cuFloatComplex *dev_W, unsigned int *buf, const int n)
{
    int bit, dist, dist2;
    int steps, threadsPerBlock, numBlocks;
    const float w_angle = dir * (M_2_PI / n);
    const float scale = dir == FFT_FORWARD ? 1.f : 1.f / n;
    const int n2 = (n / 2);
    bit = log2_32(n);
    const int lead = 32 - bit;

    _setBlocksAndThreads(&numBlocks, &threadsPerBlock, n2);
    twiddle_factors << < numBlocks, threadsPerBlock >> >(dev_W, w_angle, n);
    cudaDeviceSynchronize();

    steps = 0;
    dist2 = n;
    dist = n2;
    --bit;
    _FFTBody << < numBlocks, threadsPerBlock >> >(dev_in, dev_out, dev_W, 0xFFFFFFFF << bit, (dist - 1) << steps, steps, dist, dist2, n2);
    cudaDeviceSynchronize();
    while (bit-- > 0) {
        dist2 = dist;
        dist = dist >> 1;
        _FFTBody << < numBlocks, threadsPerBlock >> >(dev_in, dev_out, dev_W, 0xFFFFFFFF << bit, (dist - 1) << steps, steps, dist, dist2, n2);
        cudaDeviceSynchronize();
    }

    _setBlocksAndThreads(&numBlocks, &threadsPerBlock, n);
    bit_reverse << < numBlocks, threadsPerBlock >> >(dev_out, dev_in, scale, lead);
    cudaDeviceSynchronize();

    *buf = 0;
}

__global__ void _FFTBody(cuFloatComplex *in, cuFloatComplex *out, cuFloatComplex *W, const unsigned int lmask, const unsigned int pmask, const int steps, const int dist, const int dist2, const int n2)
{
    int threadId = (blockIdx.x * blockDim.x + threadIdx.x);
    int l = threadId + (threadId & lmask);
    int u = u = l + dist;
    int p = (threadId << steps) & pmask;
    cuFloatComplex tmp = cuCsubf(in[l], in[u]);
    out[l] = cuCaddf(in[l], in[u]);
    out[u] = cuCmulf(W[p], tmp);
}

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