#include <stdio.h>
#include <device_launch_parameters.h>
#include "math.h"

#include "tsTobb.cuh"
#include "tsHelper.cuh"
#include "tsTest.cuh"

__global__ void _tsTobb_body(cpx *in, cpx *out, cpx *W, const unsigned int lmask, const unsigned int pmask, const int steps, const int dist, const int dist2, const int n2);

__host__ int tsTobb_Validate(const size_t n)
{
    int result;
    cpx *in, *ref, *out, *dev_in, *dev_out, *dev_W;
    fftMalloc(n, &dev_in, &dev_out, &dev_W, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsTobb(FFT_FORWARD, &dev_in, &dev_out, dev_W, n);
    tsTobb(FFT_INVERSE, &dev_out, &dev_in, dev_W, n);
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);

    return fftResultAndFree(n, &dev_in, &dev_out, &dev_W, &in, &ref, &out) != 1;
}

__host__ double tsTobb_Performance(const size_t n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *out, *dev_in, *dev_out, *dev_W;
    fftMalloc(n, &dev_in, &dev_out, &dev_W, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        tsTobb(FFT_FORWARD, &dev_in, &dev_out, dev_W, n);
        measures[i] = stopTimer();
    }

    fftResultAndFree(n, &dev_in, &dev_out, &dev_W, &in, &ref, &out);
    return avg(measures, NUM_PERFORMANCE);
}

__host__ void tsTobb(fftDirection dir, cpx **dev_in, cpx **dev_out, cpx *dev_W, const int n)
{
    int bit, dist, dist2;
    int steps, threadsPerBlock, numBlocks;
    const float w_angle = dir * (M_2_PI / n);
    const float scale = dir == FFT_FORWARD ? 1.f : 1.f / n;
    const int n2 = (n / 2);
    bit = log2_32(n);
    const int lead = 32 - bit;

    _setBlocksAndThreads(&numBlocks, &threadsPerBlock, n2);
    twiddle_factors KERNEL_ARGS2(numBlocks, threadsPerBlock)(dev_W, w_angle, n);
    cudaDeviceSynchronize();
    
    steps = 0;
    dist2 = n;
    dist = n2;
    --bit;

    _tsTobb_body KERNEL_ARGS2(numBlocks, threadsPerBlock)(*dev_in, *dev_out, dev_W, 0xFFFFFFFF << bit, (dist - 1) << steps, steps, dist, dist2, n2);
    cudaDeviceSynchronize();
    while (bit-- > 0) {
        dist2 = dist;
        dist = dist >> 1;
        ++steps;
        _tsTobb_body KERNEL_ARGS2(numBlocks, threadsPerBlock)(*dev_out, *dev_out, dev_W, 0xFFFFFFFF << bit, (dist - 1) << steps, steps, dist, dist2, n2);
        cudaDeviceSynchronize();
    }

    _setBlocksAndThreads(&numBlocks, &threadsPerBlock, n);
    bit_reverse KERNEL_ARGS2(numBlocks, threadsPerBlock)(*dev_out, *dev_in, scale, lead);
    swap(dev_in, dev_out);
    cudaDeviceSynchronize();
}

__global__ void _tsTobb_body(cpx *in, cpx *out, cpx *W, const unsigned int lmask, const unsigned int pmask, const int steps, const int dist, const int dist2, const int n2)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    int l = i + (i & lmask);
    int u = l + dist;
    int p = (i << steps) & pmask;
    cpx tmp = cuCsubf(in[l], in[u]);
    out[l] = cuCaddf(in[l], in[u]);
    out[u] = cuCmulf(W[p], tmp);
}