#ifndef TSHELPER_CUH
#define TSHELPER_CUH

#include <Windows.h>
#include <cmath>
#include <limits>
#include <stdio.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cuda_runtime.h>
#include "tsDefinitions.cuh"
#include "imglib.h"

// Fast bit-reversal
static int tab32[32] = {
    0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30, 8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
};

__host__ static __inline__ int log2_32(int value)
{
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return tab32[(unsigned int)(value * 0x07C4ACDD) >> 27];
}

#define SYNC_BLOCKS(g, i) (__gpu_sync_lock_free((g) + (i)))

__device__ volatile int _sync_array_in[MAX_BLOCK_SIZE];
__device__ volatile int _sync_array_out[MAX_BLOCK_SIZE];

__device__ static __inline__ void init_sync(cInt tid, cInt blocks)
{
    if (tid < blocks) {
        _sync_array_in[tid] = 0;
        _sync_array_out[tid] = 0;
    }
}

__device__ static __inline__ void __gpu_sync_lock_free(cInt goal)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int nBlocks = gridDim.x;
    if (tid == 0) { _sync_array_in[bid] = goal; }    
    if (bid == 1) { // Use bid == 1, if only one block this part will not run.
        if (tid < nBlocks) { while (_sync_array_in[tid] != goal){} }
        SYNC_THREADS;
        if (tid < nBlocks) { _sync_array_out[tid] = goal; }
    }
    if (tid == 0) { while (_sync_array_out[bid] != goal) {} }
    SYNC_THREADS;
}

__device__ static inline int log2(int v)
{
    return FIND_FIRST_BIT(v) - 1;
}

__host__ __device__ static __inline__ void swap(cpx **in, cpx **out)
{
    cpx *tmp = *in;
    *in = *out;
    *out = tmp;
}

__host__ __device__ static __inline__ unsigned int bitReverse32(unsigned int x, cInt l)
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return((x >> 16) | (x << 16)) >> l;
}

__device__ static __inline__ void cpx_add_sub_mul(cpx *out_lower, cpx *out_upper, cpx in_lower, cpx in_upper, cpx w)
{
    (*out_lower) = cuCaddf(in_lower, in_upper);
    (*out_upper) = cuCmulf(cuCsubf(in_lower, in_upper), w);
}

__device__ static __inline__ void mem_gtos(int low, int high, int offset, cpx *shared, cpx *global)
{
    shared[low] = global[low + offset];
    shared[high] = global[high + offset];
}

__device__ static __inline__ void mem_gtos_db(int low, int high, int offset, unsigned int lead, cpx *shared, cpx *global)
{
    shared[low] = global[BIT_REVERSE(low + offset, lead)];
    shared[high] = global[BIT_REVERSE(high + offset, lead)];
}

__device__ static __inline__ void mem_gtos_tb(int low, int high, int offset, unsigned int lead, cpx *shared, cpx *global)
{
    shared[BIT_REVERSE(low, lead)] = global[low + offset];
    shared[BIT_REVERSE(high, lead)] = global[high + offset];
}

__device__ static __inline__ void mem_stog(int low, int high, int offset, cpx scale, cpx *shared, cpx *global)
{
    global[low + offset] = cuCmulf(shared[low], scale);
    global[high + offset] = cuCmulf(shared[high], scale);
}

__device__ static __inline__ void mem_stog_db(int low, int high, int offset, unsigned int lead, cpx scale, cpx *shared, cpx *global)
{
    global[BIT_REVERSE(low + offset, lead)] = cuCmulf(shared[low], scale);
    global[BIT_REVERSE(high + offset, lead)] = cuCmulf(shared[high], scale);
}

__device__ static __inline__ void mem_stog_tb(int low, int high, int offset, unsigned int lead, cpx scale, cpx *shared, cpx *global)
{
    global[low + offset] = cuCmulf(shared[BIT_REVERSE(low, lead)], scale);
    global[high + offset] = cuCmulf(shared[BIT_REVERSE(high, lead)], scale);
}

__host__ static __inline void set_block_and_threads(int *numBlocks, int *threadsPerBlock, cInt size)
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

__host__ void set2DBlocksNThreads(dim3 *bFFT, dim3 *tFFT, dim3 *bTrans, dim3 *tTrans, cInt n);

// Texture memory, find use?
__host__ cudaTextureObject_t specifyTexture(cpx *dev_W);
__host__ cpx* read_image(char *name, int *n);
__host__ void write_image(char *name, cpx* seq, cInt n);
__host__ void clear_image(cpx* seq, cInt n);

// Kernel functions
__global__ void twiddle_factors(cpx *W, cFloat angle, cInt n);
__global__ void bit_reverse(cpx *in, cpx *out, cFloat scale, cInt lead);
__global__ void bit_reverse(cpx *seq, fftDir dir, cInt lead, cInt n);

#endif