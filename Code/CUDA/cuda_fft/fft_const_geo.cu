#include "fft_const_geom.cuh"
#include <stdio.h>

#include "cuda_runtime.h"
#include "cuComplex.h"
#include "math.h"

#include "fft_helper.cuh"

__global__ void fft_body(cuComplex *in, cuComplex *out, cuComplex *W, const float w_angle, int steps, const int n2);
__host__ void blocks_and_threads(int *numBlocks, int *threadsPerBlock, const int size);

__host__ void fft_const_geo(const float dir, cpx *dev_in, cpx *dev_out, cuComplex *dev_W, unsigned int *buf, const int n)
{
    int bit, n2, threadsPerBlock, numBlocks;
    const float w_angle = dir * (M_2_PI / n);        
    const int lg = log2_32(n);    
    n2 = n / 2;
    blocks_and_threads(&numBlocks, &threadsPerBlock, n2);
    fft_body << <numBlocks, threadsPerBlock >> >(dev_in, dev_out, dev_W, w_angle, lg - 1, n2);
    
    const int lead = 32 - lg;
    *buf = (lg - 1) % 2;
    blocks_and_threads(&numBlocks, &threadsPerBlock, n);    
    cudaDeviceSynchronize();
    bit_reverse << <numBlocks, threadsPerBlock >> >(dev_out, dev_in, dir, lead, n);

}

__global__ void fft_body(cuComplex *in, cuComplex *out, cuComplex *W, float w_angle, int steps, const int n2)
{
    int l = (blockIdx.x * blockDim.x + threadIdx.x);
    int i = l * 2;
    int ii = i + 1;
    int u = n2 + l;
    int bit = steps + 1;
    cuComplex tmp, trig, *swap_buf;

    while (bit-- > 0) {
        sincosf(w_angle * (l & (0xffffffff << (steps - bit))), &trig.y, &trig.x);
        tmp = cuCsubf(in[l], in[u]);
        out[i] = cuCaddf(in[l], in[u]);
        out[ii] = cuCmulf(trig, tmp);

        swap_buf = in;
        in = out;
        out = swap_buf;
        __syncthreads();
    }    
}

__host__ void blocks_and_threads(int *numBlocks, int *threadsPerBlock, const int size)
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