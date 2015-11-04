#pragma once
#ifndef MYHELPERCUDA_CUH
#define MYHELPERCUDA_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cuda_runtime.h>
#include "../../Definitions.h"
#include "../../Common/imglib.h"
#include "../../Common/mycomplex.h"

//
// CUDA compiler nvcc intrisics related defines.
//
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#define SYNC_THREADS __syncthreads()
#define BIT_REVERSE(x, l) ((__brev((x))) >> (l))
#define SIN_COS_F(a, x, y) __sincosf(a, x, y)
#define FIND_FIRST_BIT(v) (__ffs(v))
#define ATOMIC_CAS(a,c,v) (atomicCAS((int *)(a),(int)(c),(int)(v)))
#define THREAD_FENCE __threadfence()
#define ATOMIC_ADD(a, b) (atomicAdd((int *)(a), (int)(b)))
#define SURF2D_READ(d,s,x,y) (surf2Dread((d), (s), (x) * sizeof(cpx), (y)))
#define SURF2D_WRITE(d,s,x,y) (surf2Dwrite((d), (s), (x) * sizeof(cpx), (y)));
#else
#define SURF2D_READ(d,s,x,y) 1
#define SURF2D_WRITE(d,s,x,y) 1
#define ATOMIC_ADD(a, b) 1
#define ATOMIC_CAS(a, c, v) 1
#define THREAD_FENCE
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#define SYNC_THREADS
#define BIT_REVERSE(x, l) 0
#define SIN_COS_F(a, x, y)
#define FIND_FIRST_BIT(v)
#endif

__host__ __device__ static __inline__ void cuda_surface_swap(cudaSurfaceObject_t *in, cudaSurfaceObject_t *out)
{
    cudaSurfaceObject_t tmp = *in;
    *in = *out;
    *out = tmp;
}
/*
__device__ static __inline__ void cuda_block_sync_init(volatile int sync_in[], volatile int sync_out[], int tid, int blocks)
{
    if (tid < blocks) {
        sync_in[tid] = 0;
        sync_out[tid] = 0;
    }
}

__device__ static __inline__ void cuda_block_sync(volatile int sync_in[], volatile int sync_out[], int goal)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int number_of_blocks = gridDim.x;
    if (tid == 0) { sync_in[bid] = goal; }
    if (bid == 1) { // Use bid == 1, if only one block this part will not run.
        if (tid < number_of_blocks) { while (sync_in[tid] != goal){} }
        SYNC_THREADS;
        if (tid < number_of_blocks) { sync_out[tid] = goal; }
    }
    if (tid == 0) { while (sync_out[bid] != goal) {} }
    SYNC_THREADS;
}
*/

__device__ static inline int log2(int v)
{
    return FIND_FIRST_BIT(v) - 1;
}

__host__ __device__ static __inline__ void cpx_add_sub_mul(cpx *inL, cpx *inU, cpx *outL, cpx *outU, const cpx *W)
{
    float x = inL->x - inU->x;
    float y = inL->y - inU->y;
    outL->x = inL->x + inU->x;
    outL->y = inL->y + inU->y;
    outU->x = (W->x * x) - (W->y * y);
    outU->y = (W->y * x) + (W->x * y);
}

__host__ __device__ static __inline__ void cpx_add_sub_mul(cpx *low, cpx *high, const cpx *W)
{
    float x = low->x - high->x;
    float y = low->y - high->y;
    low->x = low->x + high->x;
    low->y = low->y + high->y;
    high->x = (W->x * x) - (W->y * y);
    high->y = (W->y * x) + (W->x * y);
}

__device__ static __inline__ void mem_gtos_row(int low, int high, int offset, cpx *shared, cudaSurfaceObject_t surf)
{
    SURF2D_READ(&(shared[low]), surf, low + offset, blockIdx.x);
    SURF2D_READ(&(shared[high]), surf, high + offset, blockIdx.x);
}

__device__ static __inline__ void mem_gtos_col(int low, int high, int offset, cpx *shared, cudaSurfaceObject_t surf)
{
    SURF2D_READ(&(shared[low]), surf, blockIdx.x, low + offset);
    SURF2D_READ(&(shared[high]), surf, blockIdx.x, high + offset);
}

__device__ static __inline__ void mem_stog_dbt_row(int low, int high, int offset, unsigned int leading_bits, cpx scalar, cpx *shared, cudaSurfaceObject_t surf)
{
    int row_low = BIT_REVERSE(low + offset, leading_bits);
    int row_high = BIT_REVERSE(high + offset, leading_bits);
    SURF2D_WRITE(cuCmulf(shared[low], scalar), surf, blockIdx.x, row_low);
    SURF2D_WRITE(cuCmulf(shared[high], scalar), surf, blockIdx.x, row_high);
}

__device__ static __inline__ void mem_stog_db_col(int low, int high, int offset, unsigned int leading_bits, cpx scalar, cpx *shared, cudaSurfaceObject_t surf)
{
    int col_low = BIT_REVERSE(low + offset, leading_bits);
    int col_high = BIT_REVERSE(high + offset, leading_bits);
    SURF2D_WRITE(cuCmulf(shared[low], scalar), surf, blockIdx.x, col_low);
    SURF2D_WRITE(cuCmulf(shared[high], scalar), surf, blockIdx.x, col_high);
}

__host__ static __inline void set_block_and_threads(int *number_of_blocks, int *threads_per_block, int block_size, int size)
{
    if (size > block_size) {
        *number_of_blocks = size / block_size;
        *threads_per_block = block_size;
    }
    else {
        *number_of_blocks = 1;
        *threads_per_block = size;
    }
}

__host__ static __inline void set_block_and_threads2D(dim3 *number_of_blocks, int *threads_per_block, int block_size, int n)
{
    number_of_blocks->x = n;
    int n_half = n >> 1;
    if (n_half > block_size) {
        number_of_blocks->y = n_half / block_size;
        *threads_per_block = block_size;
    }
    else {
        number_of_blocks->y = 1;
        *threads_per_block = n_half;
    }
}

__host__ static __inline void set_block_and_threads_transpose(dim3 *bTrans, dim3 *tTrans, int tile_dim, int block_dim, int n)
{
    int minDim = n > tile_dim ? (n / tile_dim) : 1;
    bTrans->z = tTrans->z = 1;
    bTrans->x = bTrans->y = minDim;
    tTrans->x = tTrans->y = block_dim;
}
void checkCudaError();
void checkCudaError(char *msg);

// OLD TEST

cpx *get_sin_img(int n);
void cudaCheckError(cudaError_t err);
void cudaCheckError();

void cuda_setup_buffers     (int n, cpx **dev_in, cpx **dev_out, cpx **in, cpx **ref, cpx **out);
int cuda_shakedown          (int n, cpx **dev_in, cpx **dev_out, cpx **in, cpx **ref, cpx **out);
void cuda_setup_buffers_2d  (cpx **in, cpx **ref, cpx **dev_i, cpx **dev_o, size_t *size, int n);
void cuda_shakedown_2d      (cpx **in, cpx **ref, cpx **dev_i, cpx **dev_o);
int cuda_compare_result     (cpx *in, cpx *ref, cpx *dev, size_t size, int len);
int cuda_compare_result     (cpx *in, cpx *ref, cpx *dev, size_t size, int len, double *diff);

#endif