#include <stdio.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "math.h"
#include "FFTConstantGeom.cuh"
#include "fft_helper.cuh"

__global__ void _FFTBody(cuFloatComplex *in, cuFloatComplex *out, cuFloatComplex *W, unsigned int mask, const int n2);
__global__ void _FFTBody(cuFloatComplex *in, cuFloatComplex *out, const float w_angle, unsigned int mask, const int n2);
__host__ __inline void _setBlocksAndThreads(int *numBlocks, int *threadsPerBlock, const int size);
__host__ __inline void _swap(cuFloatComplex **in, cuFloatComplex **out);

__host__ void FFTConstGeom(fftDirection dir, cuFloatComplex *dev_in, cuFloatComplex *dev_out, cuFloatComplex *dev_W, unsigned int *buf, const int n)
{
    int steps, depth, threadsPerBlock, numBlocks;
    const float w_angle = dir * (M_2_PI / n);
    const float scale = dir == FFT_FORWARD ? 1.f : 1.f / n;
    const int n2 = (n / 2);
    cuFloatComplex *W;

    depth = log2_32(n);

    _setBlocksAndThreads(&numBlocks, &threadsPerBlock, n2);
    twiddle_factors << < numBlocks, threadsPerBlock >> >(dev_W, w_angle, n);
    cudaDeviceSynchronize();

    // Assign values from dev_W to texture memory
    cudaTextureObject_t texDev_W = specifyTexture(dev_W);

    steps = 0;
    //_FFTBody << < numBlocks, threadsPerBlock >> >(dev_in, dev_out, dev_W, 0xffffffff << steps, n2);
    _FFTBody << < numBlocks, threadsPerBlock >> >(dev_in, dev_out, texDev_W, 0xffffffff << steps, n2);
    cudaDeviceSynchronize();
    while (++steps < depth) {
        _swap(&dev_in, &dev_out);
        //_FFTBody << < numBlocks, threadsPerBlock >> >(dev_in, dev_out, dev_W, 0xffffffff << steps, n2);
        _FFTBody << < numBlocks, threadsPerBlock >> >(dev_in, dev_out, texDev_W, 0xffffffff << steps, n2);
        cudaDeviceSynchronize();
    }

    _setBlocksAndThreads(&numBlocks, &threadsPerBlock, n);
    bit_reverse << < numBlocks, threadsPerBlock >> >(dev_out, dev_in, scale, 32 - depth);
    cudaDeviceSynchronize();

    *buf = (depth - 1) % 2;
}

__host__ void FFTConstGeom2(fftDirection dir, cuFloatComplex *dev_in, cuFloatComplex *dev_out, unsigned int *buf, const int n)
{
    int steps, depth, threadsPerBlock, numBlocks;
    const float w_angle = dir * (M_2_PI / n);
    const float scale = dir == FFT_FORWARD ? 1.f : 1.f / n;
    const int n2 = (n / 2);

    depth = log2_32(n);
    steps = 0;

    _setBlocksAndThreads(&numBlocks, &threadsPerBlock, n2);
    _FFTBody << < numBlocks, threadsPerBlock >> >(dev_in, dev_out, w_angle, 0xffffffff << steps, n2);
    cudaDeviceSynchronize();
    while (++steps < depth) {
        _swap(&dev_in, &dev_out);
        _FFTBody << < numBlocks, threadsPerBlock >> >(dev_in, dev_out, w_angle, 0xffffffff << steps, n2);
        cudaDeviceSynchronize();
    }

    _setBlocksAndThreads(&numBlocks, &threadsPerBlock, n);
    bit_reverse << < numBlocks, threadsPerBlock >> >(dev_out, dev_in, scale, 32 - depth);
    cudaDeviceSynchronize();

    *buf = (depth - 1) % 2;
}

__host__ void FFTConstGeom3(fftDirection dir, cuFloatComplex *dev_in, cuFloatComplex *dev_out, cuFloatComplex *dev_W, unsigned int *buf, const int n)
{
    int steps, depth, threadsPerBlock, numBlocks;
    const float w_angle = dir * (M_2_PI / n);
    const float scale = dir == FFT_FORWARD ? 1.f : 1.f / n;
    const int n2 = (n / 2);
    cuFloatComplex *W;

    depth = log2_32(n);

    _setBlocksAndThreads(&numBlocks, &threadsPerBlock, n2);
    twiddle_factors << < numBlocks, threadsPerBlock >> >(dev_W, w_angle, n);
    cudaDeviceSynchronize();

    // Assign values from dev_W to texture memory
    cudaTextureObject_t texDev_W = specifyTexture(dev_W);

    steps = 0;
    //_FFTBody << < numBlocks, threadsPerBlock >> >(dev_in, dev_out, dev_W, 0xffffffff << steps, n2);
    _FFTBody << < numBlocks, threadsPerBlock >> >(dev_in, dev_out, texDev_W, 0xffffffff << steps, n2);
    cudaDeviceSynchronize();
    while (++steps < depth) {
        _swap(&dev_in, &dev_out);
        //_FFTBody << < numBlocks, threadsPerBlock >> >(dev_in, dev_out, dev_W, 0xffffffff << steps, n2);
        _FFTBody << < numBlocks, threadsPerBlock >> >(dev_in, dev_out, texDev_W, 0xffffffff << steps, n2);
        cudaDeviceSynchronize();
    }

    _setBlocksAndThreads(&numBlocks, &threadsPerBlock, n);
    bit_reverse << < numBlocks, threadsPerBlock >> >(dev_out, dev_in, scale, 32 - depth);
    cudaDeviceSynchronize();

    *buf = (depth - 1) % 2;
}

__global__ void _FFTBody(cuFloatComplex *in, cuFloatComplex *out, cuFloatComplex *W, unsigned int mask, const int n2)
{
    //__shared__ cuFloatComplex input[256];
    int l = (blockIdx.x * blockDim.x + threadIdx.x);
    int i = l * 2;
    cuFloatComplex in_lower = in[l];
    __syncthreads(); // to be tested...
    cuFloatComplex in_upper = in[n2 + l];
    cuFloatComplex tmp = cuCsubf(in_lower, in_upper);
    cuFloatComplex twiddle = W[l & mask];
    out[i] = cuCaddf(in_lower, in_upper);
    out[i + 1] = cuCmulf(twiddle, tmp);
}

__global__ void _FFTBody(cuFloatComplex *in, cuFloatComplex *out, const float w_angle, unsigned int mask, const int n2)
{
    int threadId = (blockIdx.x * blockDim.x + threadIdx.x);
    int i = threadId * 2;
    int u = n2 + threadId;
    cuFloatComplex in_lower = in[threadId];
    cuFloatComplex in_upper = in[n2 + threadId];
    cuFloatComplex tmp = cuCsubf(in_lower, in_upper);
    cuFloatComplex twiddle;
    sincosf(w_angle * (threadId & mask), &twiddle.y, &twiddle.x);
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

__host__ __inline void _swap(cuFloatComplex **in, cuFloatComplex **out)
{
    cuFloatComplex *tmp = *in;
    *in = *out;
    *out = tmp;
}