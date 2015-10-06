#ifndef TSHELPER_CUH
#define TSHELPER_CUH

#include <Windows.h>
#include <cmath>
#include <limits>
#include <stdio.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cuda_runtime.h>
#include "definitions.h"
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

__device__ volatile int _sync_array_in[MAX_BLOCK_SIZE];
__device__ volatile int _sync_array_out[MAX_BLOCK_SIZE];
__device__ volatile int _sync_array_2din[8192][32]; // assume block dim >= 256 AND rows <= 8192
__device__ volatile int _sync_array_2dout[8192][32];

__device__ static __inline__ void init_sync(int tid, int blocks)
{
    if (tid < blocks) {
        _sync_array_in[tid] = 0;
        _sync_array_out[tid] = 0;
    }
}

__device__ static __inline__ void __gpu_sync(int goal)
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

__host__ __device__ static __inline__ void swap(cuSurf *in, cuSurf *out)
{
    cuSurf tmp = *in;
    *in = *out;
    *out = tmp;
}

__host__ __device__ static __inline__ unsigned int bitReverse32(unsigned int x, int l)
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

__device__ static __inline__ void mem_gtos_col(int low, int high, int global_low, int offsetHigh, cpx *shared, cpx *global)
{
    shared[low] = global[global_low];
    shared[high] = global[global_low + offsetHigh];
}

__device__ static __inline__ void mem_stog_db_col(int shared_low, int shared_high, int offset, unsigned int lead, cpx scale, cpx *shared, cpx *global, int n)
{
    int row_low = BIT_REVERSE(shared_low + offset, lead);
    int row_high = BIT_REVERSE(shared_high + offset, lead);
    global[row_low * n + blockIdx.x] = cuCmulf(shared[shared_low], scale);
    global[row_high * n + blockIdx.x] = cuCmulf(shared[shared_high], scale);
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

__device__ static __inline__ void mem_stog_dbt(int low, int high, int offset, unsigned int lead, cpx scale, cpx *shared, cpx *global)
{
    int xl = BIT_REVERSE(low + offset, lead);
    int xh = BIT_REVERSE(high + offset, lead);
    global[blockIdx.x + xl * gridDim.x] = cuCmulf(shared[low], scale);
    global[blockIdx.x + xh * gridDim.x] = cuCmulf(shared[high], scale);
}

__device__ static __inline__ void mem_stog_tb(int low, int high, int offset, unsigned int lead, cpx scale, cpx *shared, cpx *global)
{
    global[low + offset] = cuCmulf(shared[BIT_REVERSE(low, lead)], scale);
    global[high + offset] = cuCmulf(shared[BIT_REVERSE(high, lead)], scale);
}


__device__ static __inline__ void mem_gtos_row(int low, int high, int offset, cpx *shared, cuSurf surf)
{
    SURF2D_READ(&(shared[low]), surf, low + offset, blockIdx.x);
    SURF2D_READ(&(shared[high]), surf, high + offset, blockIdx.x);
}

__device__ static __inline__ void mem_gtos_col(int low, int high, int offset, cpx *shared, cuSurf surf)
{
    SURF2D_READ(&(shared[low]), surf, blockIdx.x, low + offset);
    SURF2D_READ(&(shared[high]), surf, blockIdx.x, high + offset);
}

__device__ static __inline__ void mem_stog_db_row(int low, int high, int offset, unsigned int lead, cpx scale, cpx *shared, cuSurf surf)
{
    int row_low = BIT_REVERSE(low + offset, lead);
    int row_high = BIT_REVERSE(high + offset, lead);
    SURF2D_WRITE(cuCmulf(shared[low], scale), surf, row_low, blockIdx.x);
    SURF2D_WRITE(cuCmulf(shared[high], scale), surf, row_high, blockIdx.x);
}

__device__ static __inline__ void mem_stog_dbt_row(int low, int high, int offset, unsigned int lead, cpx scale, cpx *shared, cuSurf surf)
{
    int row_low = BIT_REVERSE(low + offset, lead);
    int row_high = BIT_REVERSE(high + offset, lead);
    SURF2D_WRITE(cuCmulf(shared[low], scale), surf, blockIdx.x, row_low);
    SURF2D_WRITE(cuCmulf(shared[high], scale), surf, blockIdx.x, row_high);
}

__device__ static __inline__ void mem_stog_db_col(int low, int high, int offset, unsigned int lead, cpx scale, cpx *shared, cuSurf surf)
{
    int col_low = BIT_REVERSE(low + offset, lead);
    int col_high = BIT_REVERSE(high + offset, lead);
    SURF2D_WRITE(cuCmulf(shared[low], scale), surf, blockIdx.x, col_low);
    SURF2D_WRITE(cuCmulf(shared[high], scale), surf, blockIdx.x, col_high);
}

__host__ static __inline void set_block_and_threads(int *numBlocks, int *threadsPerBlock, int size)
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

__host__ static __inline void set_block_and_threads2D(dim3 *numBlocks, int *threadsPerBlock, int n)
{
    numBlocks->x = n;
    int n2 = n >> 1;
    if (n2 > MAX_BLOCK_SIZE) {        
        numBlocks->y = n2 / MAX_BLOCK_SIZE;
        *threadsPerBlock = MAX_BLOCK_SIZE;
    }
    else {
        numBlocks->y = 1;
        *threadsPerBlock = n2;
    }  
}

__host__ static __inline void set_block_and_threads_transpose(dim3 *bTrans, dim3 *tTrans, int n)
{
    bTrans->z = tTrans->z = 1;
    bTrans->x = bTrans->y = (n / TILE_DIM);
    tTrans->x = tTrans->y = THREAD_TILE_DIM;
}

// Likely room for optimizations!
// In place swapping is problematic over several blocks and is not the task of this thesis work (solving block sync)

// Transpose data set with dimensions n x n
__global__ void _kernelTranspose(cpx *in, cpx *out, int n);
__global__ void _kernelTranspose(cuSurf in, cuSurf out, int n);

#define NO_STREAMING_MULTIPROCESSORS 7

// The limit is number of SMs but some configuration can run one step beyond that limit when running with release config.
__host__ static __inline int checkValidConfig(int blocks, int n)
{
    if (blocks > NO_STREAMING_MULTIPROCESSORS) {
        switch (MAX_BLOCK_SIZE)
        {
        case 256:   return blocks <= 32;    // 2^14
        case 512:   return blocks <= 16;    // 2^14
        case 1024:  return blocks <= 4;     // 2^13
            // Default is a configurable limit, essentially blocksize limits the number of treads that can perform the synchronization.
        default:    return n <= MAX_BLOCK_SIZE * MAX_BLOCK_SIZE;
        }
    }
    return 1;
}

__host__ void checkCudaError();
__host__ void checkCudaError(char *msg);
__host__ void set2DBlocksNThreads(dim3 *bFFT, dim3 *tFFT, dim3 *bTrans, dim3 *tTrans, int n);
__host__ cpx* read_image(char *name, int *n);
__host__ void write_image(char *name, char *type, cpx* seq, int n);
__host__ void write_normalized_image(char *name, cpx* seq, int n);
__host__ void normalized_image(cpx* seq, int n);
__host__ cpx *fftShift(cpx *seq, int n);
__host__ void clear_image(cpx* seq, int n);

// Kernel functions
__global__ void twiddle_factors(cpx *W, float angle, int n);
__global__ void bit_reverse(cpx *in, cpx *out, float scale, int lead);
__global__ void bit_reverse(cpx *seq, fftDir dir, int lead, int n);

#endif