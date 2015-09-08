#include <stdio.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "math.h"
#include "FFTConstGeom.cuh"
#include "fft_helper.cuh"
#include "fft_test.cuh"

__global__ void _FFTBody(cpx *in, cpx *out, cpx *W, unsigned int mask, const int n2);
__global__ void _FFTBody(cpx *in, cpx *out, const float w_angle, unsigned int mask, const int n2);
__host__ __inline void _setBlocksAndThreads(int *numBlocks, int *threadsPerBlock, const int size);

__host__ int FFTConstGeom_Validate(const size_t n)
{
    int result;
    cpx *in, *ref, *out, *dev_in, *dev_out, *dev_W;
    
    _fftTestSeq(n, &in, &ref, &out);
    _cudaMalloc(n, &dev_in, &dev_out, &dev_W);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);

    FFTConstGeom(FFT_FORWARD, &dev_in, &dev_out, dev_W, n);
    FFTConstGeom(FFT_INVERSE, &dev_out, &dev_in, dev_W, n);
    
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);    
    _cudaFree(&dev_in, &dev_out, &dev_W);
    cudaDeviceSynchronize();

    result = checkError(in, ref, n);
    _fftFreeSeq(&in, &out, &ref);
    return result != 1;
}

__host__ double FFTConstGeom_Performance(const size_t n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *out, *dev_in, *dev_out, *dev_W;

    _fftTestSeq(n, &in, &ref, &out);
    _cudaMalloc(n, &dev_in, &dev_out, &dev_W);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);

    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        FFTConstGeom(FFT_FORWARD, &dev_in, &dev_out, dev_W, n);
        measures[i] = stopTimer();
    }

    _cudaFree(&dev_in, &dev_out, &dev_W);
    cudaDeviceSynchronize();
    _fftFreeSeq(&in, &out, &ref);
    return avg(measures, NUM_PERFORMANCE);
}

__host__ void FFTConstGeom(fftDirection dir, cpx **dev_in, cpx **dev_out, cpx *dev_W, const int n)
{
    int steps, depth, threadsPerBlock, numBlocks;
    const float w_angle = dir * (M_2_PI / n);
    const float scale = dir == FFT_FORWARD ? 1.f : 1.f / n;
    const int n2 = (n / 2);
    cpx *W;

    depth = log2_32(n);

    _FFTBody<<<numBlocks, threadsPerBlock, 

    _setBlocksAndThreads(&numBlocks, &threadsPerBlock, n2);
    twiddle_factors << < numBlocks, threadsPerBlock >> >(dev_W, w_angle, n);
    cudaDeviceSynchronize();
    
    steps = 0;
    _FFTBody << < numBlocks, threadsPerBlock >> >(*dev_in, *dev_out, dev_W, 0xffffffff << steps, n2);
    cudaDeviceSynchronize();
    while (++steps < depth) {
        swap(dev_in, dev_out);        
        _FFTBody << < numBlocks, threadsPerBlock >> >(*dev_in, *dev_out, dev_W, 0xffffffff << steps, n2);
        cudaDeviceSynchronize();
    }

    _setBlocksAndThreads(&numBlocks, &threadsPerBlock, n);    
    bit_reverse << < numBlocks, threadsPerBlock >> >(*dev_out, *dev_in, scale, 32 - depth);
    swap(dev_in, dev_out);
    cudaDeviceSynchronize();
}

__global__ void _Body(cpx *in, cpx *out, cpx *W, unsigned int mask, const int n2)
{
    extern __shared__ cpx input[];
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);

    //__syncthreads()
}

__global__ void _FFTBody(cpx *in, cpx *out, cpx *W, unsigned int mask, const int n2)
{
    //__shared__ cpx input[256];
    int l = (blockIdx.x * blockDim.x + threadIdx.x);
    int i = l * 2;
    cpx in_lower = in[l];
    cpx in_upper = in[n2 + l];
    cpx tmp = cuCsubf(in_lower, in_upper);
    cpx twiddle = W[l & mask];
    out[i] = cuCaddf(in_lower, in_upper);
    out[i + 1] = cuCmulf(twiddle, tmp);
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