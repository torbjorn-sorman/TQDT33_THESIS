#include "fft_const_geom.cuh"
#include <stdio.h>

#include "cuda_runtime.h"
#include "math.h"

#include "fft_helper.cuh"

__global__ void fft_body(cpx *in, cpx *out, cpx *W, unsigned int mask, const int n);

__host__ void fft_const_geom(const double dir, cpx *dev_in, cpx *dev_out, cpx *dev_W, const int n)
{
    cudaError_t cudaStatus;
    int bit, steps, n2;
    unsigned int mask;
    cpx *tmp;
    bit = log2_32(n);
    const int lead = 32 - bit;
    n2 = n / 2;
    steps = --bit;
    mask = 0xffffffff << (steps - bit);
    int threadsPerBlock, numBlocks;
    if (n > 256) {        
        threadsPerBlock = 256;
        numBlocks = n / 256;
    }
    else {
        threadsPerBlock = n;
        numBlocks = 1;        
    }
    twiddle_factors << <numBlocks, threadsPerBlock >> >(dev_W, dir, n);  
    

             
    cudaDeviceSynchronize();
    fft_body << <numBlocks, threadsPerBlock >> >(dev_in, dev_out, dev_W, mask, n2);    
    while (bit-- > 0)
    {
        tmp = *(&dev_in);
        *(&dev_in) = *(&dev_out);
        *(&dev_out) = tmp;
        mask = 0xffffffff << (steps - bit);
        cudaDeviceSynchronize();        
        fft_body << <numBlocks, threadsPerBlock >> >(dev_in, dev_out, dev_W, mask, n2);
    }
    //cudaDeviceSynchronize();
    //bit_reverse << <numBlocks, threadsPerBlock >> >(dev_out, dir, lead, n);            
    //cudaDeviceSynchronize();
}

__global__ void fft_body(cpx *in, cpx *out, cpx *W, unsigned int mask, const int n2)
{
    int l, u, p;
    cpx tmp;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    l = i / 2;
    u = n2 + l;
    p = l & mask;
    tmp.x = in[l].x - in[u].x;
    tmp.y = in[l].y - in[u].y;
    out[i].x = in[l].x + in[u].x;
    out[i].y = in[l].y + in[u].y;
    out[i + 1].x = (W[p].x * tmp.x) - (W[p].y * tmp.y);
    out[i + 1].y = (W[p].y * tmp.x) + (W[p].x * tmp.y);
}