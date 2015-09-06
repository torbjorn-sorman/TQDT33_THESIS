#include "fft_const_geom.cuh"
#include <stdio.h>

#include "cuda_runtime.h"
#include "cuComplex.h"
#include "math.h"

#include "fft_helper.cuh"

__global__ void fft_body(cuComplex *in, cuComplex *out, cuComplex *W, unsigned int mask, const int n);
__global__ void fft_body(cuComplex *in, cuComplex *out, float w_angle, unsigned int mask, const int n2);

__host__ void set_blocks_and_threads(int *numBlocks, int *threadsPerBlock, const int size)
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

__host__ void fft_const_geom(const float dir, cpx *dev_in, cpx *dev_out, cuComplex *dev_W, unsigned int *buf, const int n)
{
    int bit, steps, n2, threadsPerBlock, numBlocks;
    unsigned int mask;
    cpx *tmp;
    float w_angle = dir * (M_2_PI / n);
    bit = log2_32(n);
    const int lead = 32 - bit;
    n2 = n / 2;
    steps = --bit;
    mask = 0xffffffff << (steps - bit);
    *buf = 1;
    set_blocks_and_threads(&numBlocks, &threadsPerBlock, n2);    
    //twiddle_factors << <numBlocks, threadsPerBlock >> >(dev_W, dir, n);
    //cudaDeviceSynchronize();

    //fft_body << <numBlocks, threadsPerBlock >> >(dev_in, dev_out, dev_W, mask, n2);
    fft_body << <numBlocks, threadsPerBlock >> >(dev_in, dev_out, w_angle, mask, n2);
    while (bit-- > 0)
    {
        *buf ^= 1;
        tmp = dev_in;
        dev_in = dev_out;
        dev_out = tmp;
        mask = 0xffffffff << (steps - bit);
        cudaDeviceSynchronize();        
        //fft_body << <numBlocks, threadsPerBlock >> >(dev_in, dev_out, dev_W, mask, n2);
        fft_body << <numBlocks, threadsPerBlock >> >(dev_in, dev_out, w_angle, mask, n2);
    }
    cudaDeviceSynchronize();
    
    set_blocks_and_threads(&numBlocks, &threadsPerBlock, n);
    *buf ^= 1;
    bit_reverse << <numBlocks, threadsPerBlock >> >(dev_out, dev_in, dir, lead, n);            
    
}

__global__ void fft_body(cuComplex *in, cuComplex *out, float w_angle, unsigned int mask, const int n2)
{
    cuComplex tmp, trig;
    int l = (blockIdx.x * blockDim.x + threadIdx.x);
    int i = l * 2;
    int u = n2 + l;
    sincosf(w_angle * (l & mask), &trig.y, &trig.x);
    tmp = cuCsubf(in[l], in[u]);
    out[i] = cuCaddf(in[l], in[u]);
    out[i + 1] = cuCmulf(trig, tmp);
}

__global__ void fft_body(cuComplex *in, cuComplex *out, cuComplex *W, unsigned int mask, const int n2)
{
    cuComplex tmp;
    int l = (blockIdx.x * blockDim.x + threadIdx.x);
    int i = l * 2;
    int u = n2 + l;
    tmp = cuCsubf(in[l], in[u]);
    out[i] = cuCaddf(in[l], in[u]);
    out[i + 1] = cuCmulf(W[l & mask], tmp);
}