#include <stdio.h>
#include <device_launch_parameters.h>
#include "math.h"

#include "tsTobb.cuh"
#include "tsHelper.cuh"
#include "tsTest.cuh"

__global__ void _tsTobb_body(cpx *in, cpx *out, cpx *W, cUInt lmask, cUInt pmask, cInt steps, cInt dist);

__host__ int tsTobb_Validate(cInt n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out, *dev_W;
    fftMalloc(n, &dev_in, &dev_out, &dev_W, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsTobb(FFT_FORWARD, &dev_in, &dev_out, dev_W, n);
    tsTobb(FFT_INVERSE, &dev_out, &dev_in, dev_W, n);
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);

    return fftResultAndFree(n, &dev_in, &dev_out, &dev_W, &in, &ref, &out) != 1;
}

__host__ double tsTobb_Performance(cInt n)
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

__host__ void tsTobb(fftDir dir, cpx **dev_in, cpx **dev_out, cpx *dev_W, cInt n)
{
    int bit, dist;
    int steps, threadsPerBlock, numBlocks;
    cFloat w_angle = dir * (M_2_PI / n);
    cFloat scale = dir == FFT_FORWARD ? 1.f : 1.f / n;
    cInt n2 = (n / 2);
    bit = log2_32(n);
    cInt lead = 32 - bit;

    set_block_and_threads(&numBlocks, &threadsPerBlock, n2);
    twiddle_factors KERNEL_ARGS2(numBlocks, threadsPerBlock)(dev_W, w_angle, n);
    cudaDeviceSynchronize();
    
    steps = 0;
    dist = n2;
    --bit;

    _tsTobb_body KERNEL_ARGS2(numBlocks, threadsPerBlock)(*dev_in, *dev_out, dev_W, 0xFFFFFFFF << bit, (dist - 1) << steps, steps, dist);
    cudaDeviceSynchronize();

    while (bit-- > 0) {
        dist = dist >> 1;
        ++steps;
        _tsTobb_body KERNEL_ARGS2(numBlocks, threadsPerBlock)(*dev_out, *dev_out, dev_W, 0xFFFFFFFF << bit, (dist - 1) << steps, steps, dist);
        cudaDeviceSynchronize();
    }

    set_block_and_threads(&numBlocks, &threadsPerBlock, n);
    bit_reverse KERNEL_ARGS2(numBlocks, threadsPerBlock)(*dev_out, *dev_in, scale, lead);
    swap(dev_in, dev_out);
    cudaDeviceSynchronize();
}

__global__ void _tsTobb_body(cpx *in, cpx *out, cpx *W, cUInt lmask, cUInt pmask, cInt steps, cInt dist)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    int l = i + (i & lmask);
    int u = l + dist;
    int p = (i << steps) & pmask;
    cpx in_lower = in[l];
    cpx in_upper = in[u];
    cpx w = W[p];    
    cpx_add_sub_mul(&(out[l]), &(out[u]), in_lower, in_upper, w);
}