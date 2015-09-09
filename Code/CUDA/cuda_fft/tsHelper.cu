#include <stdio.h>
#include <device_launch_parameters.h>

#include "tsHelper.cuh"
#include "math.h"

int log2_32(int value)
{
    value |= value >> 1; value |= value >> 2; value |= value >> 4; value |= value >> 8; value |= value >> 16;
    return tab32[(unsigned int)(value * 0x07C4ACDD) >> 27];
}

/* Doubtful this works... */
__host__ cudaTextureObject_t specifyTexture(cpx *dev_W)
{
    // Specify texture
    struct cudaResourceDesc resDesc;
    cudaMemset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.linear.devPtr = dev_W;
    //resDesc.res.array.array = cuArray; 

    // Specify texture object parameters 
    struct cudaTextureDesc texDesc;
    cudaMemset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // Create texture object 
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    return texObj;
}

__host__ void swap(cpx **in, cpx **out)
{
    cpx *tmp = *in;
    *in = *out;
    *out = tmp;
}

__host__ void setBlocksAndThreads(int *numBlocks, int *threadsPerBlock, const int size)
{
    if (size > MAX_BLOCK_SIZE) {
        *numBlocks = size / MAX_BLOCK_SIZE;
        *threadsPerBlock = MAX_BLOCK_SIZE;
    }
    else {
        *numBlocks = 1;
        *threadsPerBlock = size;
    }
}

__global__ void twiddle_factors(cpx *W, const float angle, const int n)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    SIN_COS_F(angle * i, &W[i].y, &W[i].x);
}

__global__ void bit_reverse(cpx *in, cpx *out, const float scale, const int lead)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    //unsigned int p = bitReverse32(i, lead);
    //unsigned int p = __brev(i) >> lead;
    unsigned int p = BIT_REVERSE(i, lead);

    out[p].x = in[i].x * scale;
    out[p].y = in[i].y * scale;
}

__global__ void bit_reverse(cpx *x, const float dir, const int lead, const int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    //unsigned int p = bitReverse32(i, lead);
    //unsigned int p = __brev(i) >> lead;
    unsigned int p = BIT_REVERSE(i, lead);
    cpx tmp;
    if (i < p) {
        tmp = x[i];
        x[i] = x[p];
        x[p] = tmp;
    }
    if (dir > 0) {
        x[i].x = x[i].x / (float)n;
        x[i].y = x[i].y / (float)n;
    }
}

__device__ unsigned int bitReverse32(unsigned int x, const int l)
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return((x >> 16) | (x << 16)) >> l;
}