#include <stdio.h>
#include <device_launch_parameters.h>

#include "math.h"
#include "tsConstantGeometry.cuh"
#include "tsHelper.cuh"
#include "tsTest.cuh"

__global__ void _tsConstantGeometry_Body(cpx *in, cpx *out, cpx *W, unsigned int mask, cInt n2);

__host__ int tsConstantGeometry_Validate(cInt n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out, *dev_W;
    fftMalloc(n, &dev_in, &dev_out, &dev_W, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsConstantGeometry(FFT_FORWARD, &dev_in, &dev_out, dev_W, n);
    tsConstantGeometry(FFT_INVERSE, &dev_out, &dev_in, dev_W, n);    
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);

    return fftResultAndFree(n, &dev_in, &dev_out, &dev_W, &in, &ref, &out) != 1;
}

__host__ double tsConstantGeometry_Performance(cInt n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *out, *dev_in, *dev_out, *dev_W;
    fftMalloc(n, &dev_in, &dev_out, &dev_W, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        tsConstantGeometry(FFT_FORWARD, &dev_in, &dev_out, dev_W, n);
        measures[i] = stopTimer();
    }

    fftResultAndFree(n, &dev_in, &dev_out, &dev_W, &in, &ref, &out);
    return avg(measures, NUM_PERFORMANCE);
}

__host__ void tsConstantGeometry(const fftDir dir, cpx **dev_in, cpx **dev_out, cpx *dev_W, cInt n)
{
    int steps, depth, threadsPerBlock, numBlocks;
    cFloat w_angle = dir * (M_2_PI / n);
    cFloat scale = dir == FFT_FORWARD ? 1.f : 1.f / n;
    cInt n2 = (n / 2);

    depth = log2_32(n);

    set_block_and_threads(&numBlocks, &threadsPerBlock, n2);
    twiddle_factors KERNEL_ARGS2(numBlocks, threadsPerBlock)(dev_W, w_angle, n);
    cudaDeviceSynchronize();
    
    steps = 0;
    _tsConstantGeometry_Body KERNEL_ARGS2(numBlocks, threadsPerBlock)(*dev_in, *dev_out, dev_W, 0xffffffff << steps, n2);
    cudaDeviceSynchronize();
    while (++steps < depth) {
        swap(dev_in, dev_out);        
        _tsConstantGeometry_Body KERNEL_ARGS2(numBlocks, threadsPerBlock)(*dev_in, *dev_out, dev_W, 0xffffffff << steps, n2);
        cudaDeviceSynchronize();
    }

    set_block_and_threads(&numBlocks, &threadsPerBlock, n);    
    bit_reverse KERNEL_ARGS2(numBlocks, threadsPerBlock)(*dev_out, *dev_in, scale, 32 - depth);
    swap(dev_in, dev_out);
    cudaDeviceSynchronize();
}

__global__ void _tsConstantGeometry_Body(cpx *in, cpx *out, cpx *W, unsigned int mask, cInt n2)
{
    int l = (blockIdx.x * blockDim.x + threadIdx.x);
    int i = l * 2;
    cpx in_lower = in[l];
    cpx in_upper = in[n2 + l];
    cpx w = W[l & mask];
    cpx_add_sub_mul(&(out[i]), &(out[i + 1]), in_lower, in_upper, w);
}