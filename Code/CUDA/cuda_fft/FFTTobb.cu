#include <stdio.h>
#include <device_launch_parameters.h>

#include "math.h"
#include "FFTTobb.cuh"
#include "fft_helper.cuh"
#include "fft_test.cuh"

__global__ void _FFTBody(cpx *in, cpx *out, cpx *W, const unsigned int lmask, const unsigned int pmask, const int steps, const int dist, const int dist2, const int n2);
__host__ __inline void _setBlocksAndThreads(int *numBlocks, int *threadsPerBlock, const int size);

__host__ int FFTTobb_Validate(const size_t n)
{
    int result;
    cpx *in, *ref, *out, *dev_in, *dev_out, *dev_W;

    _fftTestSeq(n, &in, &ref, &out);
    _cudaMalloc(n, &dev_in, &dev_out, &dev_W);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);

    FFTTobb(FFT_FORWARD, &dev_in, &dev_out, dev_W, n);
    FFTTobb(FFT_INVERSE, &dev_out, &dev_in, dev_W, n);

    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);
    _cudaFree(&dev_in, &dev_out, &dev_W);
    cudaDeviceSynchronize();

    result = checkError(in, ref, n);
    _fftFreeSeq(&in, &out, &ref);
    return result != 1;
}

__host__ double FFTTobb_Performance(const size_t n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *out, *dev_in, *dev_out, *dev_W;

    _fftTestSeq(n, &in, &ref, &out);
    _cudaMalloc(n, &dev_in, &dev_out, &dev_W);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);

    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        FFTTobb(FFT_FORWARD, &dev_in, &dev_out, dev_W, n);
        measures[i] = stopTimer();
    }

    _cudaFree(&dev_in, &dev_out, &dev_W);
    cudaDeviceSynchronize();
    _fftFreeSeq(&in, &out, &ref);
    return avg(measures, NUM_PERFORMANCE);
}

__host__ void FFTTobb(fftDirection dir, cpx **dev_in, cpx **dev_out, cpx *dev_W, const int n)
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

    _FFTBody << < numBlocks, threadsPerBlock >> >(*dev_in, *dev_out, dev_W, 0xFFFFFFFF << bit, (dist - 1) << steps, steps, dist, dist2, n2);
    cudaDeviceSynchronize();
    while (bit-- > 0) {
        dist2 = dist;
        dist = dist >> 1;
        ++steps;
        _FFTBody << < numBlocks, threadsPerBlock >> >(*dev_out, *dev_out, dev_W, 0xFFFFFFFF << bit, (dist - 1) << steps, steps, dist, dist2, n2);
        cudaDeviceSynchronize();
    }

    _setBlocksAndThreads(&numBlocks, &threadsPerBlock, n);
    bit_reverse << < numBlocks, threadsPerBlock >> >(*dev_out, *dev_in, scale, lead);
    swap(dev_in, dev_out);
    cudaDeviceSynchronize();
}

__global__ void _FFTBody(cpx *in, cpx *out, cpx *W, const unsigned int lmask, const unsigned int pmask, const int steps, const int dist, const int dist2, const int n2)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    int l = i + (i & lmask);
    int u = l + dist;
    int p = (i << steps) & pmask;
    cpx tmp = cuCsubf(in[l], in[u]);
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

__host__ __inline void _swap(cpx **in, cpx **out)
{
    cpx *tmp = *in;
    *in = *out;
    *out = tmp;
}