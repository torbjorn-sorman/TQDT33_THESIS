#ifndef MYHELPERCUDA_CUH
#define MYHELPERCUDA_CUH

#include <device_launch_parameters.h>

#include <cuda_runtime.h>
#include "../../Definitions.h"
#include "../../Common/imglib.h"
#include "../../Common/mycomplex.h"

/*
__device__ static __inline__ void devSwap(cpx **in, cpx **out)
{
    cpx *tmp = *in;
    *in = *out;
    *out = tmp;
}
*/

__host__ __device__ static __inline__ void devSwap(cuSurf *in, cuSurf *out)
{
    cuSurf tmp = *in;
    *in = *out;
    *out = tmp;
}

/*
__device__ static __inline__ unsigned int devBitReverse32(unsigned int x, int l)
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return((x >> 16) | (x << 16)) >> l;
}
*/

__device__ static __inline__ void init_sync(volatile int sync_in[], volatile int sync_out[], int tid, int blocks)
{
    if (tid < blocks) {
        sync_in[tid] = 0;
        sync_out[tid] = 0;
    }
}

__device__ static __inline__ void __gpu_sync(volatile int sync_in[], volatile int sync_out[], int goal)
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

__device__ static inline int log2(int v)
{
    return FIND_FIRST_BIT(v) - 1;
}

__device__ static __inline__ void cpx_add_sub_mul(cpx *inL, cpx *inU, cpx *outL, cpx *outU, const cpx *W)
{
    float x = inL->x - inU->x;
    float y = inL->y - inU->y;
    outL->x = inL->x + inU->x;
    outL->y = inL->y + inU->y;
    outU->x = (W->x * x) - (W->y * y);
    outU->y = (W->y * x) + (W->x * y);
}

__device__ static __inline__ void mem_gtos(int low, int high, int offset, cpx *shared, cpx *global)
{
    shared[low] = global[low + offset];
    shared[high] = global[high + offset];
}
/*
__device__ static __inline__ void mem_gtos_db(int low, int high, int offset, unsigned int leading_bits, cpx *shared, cpx *global)
{
    shared[low] = global[BIT_REVERSE(low + offset, leading_bits)];
    shared[high] = global[BIT_REVERSE(high + offset, leading_bits)];
}
*/
/*
__device__ static __inline__ void mem_gtos_tb(int low, int high, int offset, unsigned int leading_bits, cpx *shared, cpx *global)
{
    shared[BIT_REVERSE(low, leading_bits)] = global[low + offset];
    shared[BIT_REVERSE(high, leading_bits)] = global[high + offset];
}
*/
__device__ static __inline__ void mem_gtos_col(int low, int high, int global_low, int offsetHigh, cpx *shared, cpx *global)
{
    shared[low] = global[global_low];
    shared[high] = global[global_low + offsetHigh];
}

__device__ static __inline__ void mem_stog_db_col(int shared_low, int shared_high, int offset, unsigned int leading_bits, cpx scale, cpx *shared, cpx *global, int n)
{
    int row_low = BIT_REVERSE(shared_low + offset, leading_bits);
    int row_high = BIT_REVERSE(shared_high + offset, leading_bits);
    global[row_low * n + blockIdx.x] = cuCmulf(shared[shared_low], scale);
    global[row_high * n + blockIdx.x] = cuCmulf(shared[shared_high], scale);
}
/*
__device__ static __inline__ void mem_stog(int low, int high, int offset, cpx scale, cpx *shared, cpx *global)
{
    global[low + offset] = cuCmulf(shared[low], scale);
    global[high + offset] = cuCmulf(shared[high], scale);
}
*/
__device__ static __inline__ void mem_stog_db(int low, int high, int offset, unsigned int leading_bits, cpx scale, cpx *shared, cpx *global)
{
    global[BIT_REVERSE(low + offset, leading_bits)] = cuCmulf(shared[low], scale);
    global[BIT_REVERSE(high + offset, leading_bits)] = cuCmulf(shared[high], scale);
}
/*
__device__ static __inline__ void mem_stog_dbt(int low, int high, int offset, unsigned int leading_bits, cpx scale, cpx *shared, cpx *global)
{
    int xl = BIT_REVERSE(low + offset, leading_bits);
    int xh = BIT_REVERSE(high + offset, leading_bits);
    global[blockIdx.x + xl * gridDim.x] = cuCmulf(shared[low], scale);
    global[blockIdx.x + xh * gridDim.x] = cuCmulf(shared[high], scale);
}
*/
/*
__device__ static __inline__ void mem_stog_tb(int low, int high, int offset, unsigned int leading_bits, cpx scale, cpx *shared, cpx *global)
{
    global[low + offset] = cuCmulf(shared[BIT_REVERSE(low, leading_bits)], scale);
    global[high + offset] = cuCmulf(shared[BIT_REVERSE(high, leading_bits)], scale);
}
*/

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
/*
__device__ static __inline__ void mem_stog_db_row(int low, int high, int offset, unsigned int leading_bits, cpx scale, cpx *shared, cuSurf surf)
{
    int row_low = BIT_REVERSE(low + offset, leading_bits);
    int row_high = BIT_REVERSE(high + offset, leading_bits);
    SURF2D_WRITE(cuCmulf(shared[low], scale), surf, row_low, blockIdx.x);
    SURF2D_WRITE(cuCmulf(shared[high], scale), surf, row_high, blockIdx.x);
}
*/
__device__ static __inline__ void mem_stog_dbt_row(int low, int high, int offset, unsigned int leading_bits, cpx scale, cpx *shared, cuSurf surf)
{
    int row_low = BIT_REVERSE(low + offset, leading_bits);
    int row_high = BIT_REVERSE(high + offset, leading_bits);
    SURF2D_WRITE(cuCmulf(shared[low], scale), surf, blockIdx.x, row_low);
    SURF2D_WRITE(cuCmulf(shared[high], scale), surf, blockIdx.x, row_high);
}

__device__ static __inline__ void mem_stog_db_col(int low, int high, int offset, unsigned int leading_bits, cpx scale, cpx *shared, cuSurf surf)
{
    int col_low = BIT_REVERSE(low + offset, leading_bits);
    int col_high = BIT_REVERSE(high + offset, leading_bits);
    SURF2D_WRITE(cuCmulf(shared[low], scale), surf, blockIdx.x, col_low);
    SURF2D_WRITE(cuCmulf(shared[high], scale), surf, blockIdx.x, col_high);
}

__global__ void kernelTranspose(cpx *in, cpx *out, int n);
__global__ void kernelTranspose(cuSurf in, cuSurf out, int n);

void set_block_and_threads(int *number_of_blocks, int *threadsPerBlock, int size);
void set_block_and_threads2D(dim3 *number_of_blocks, int *threadsPerBlock, int n);
void set_block_and_threads_transpose(dim3 *bTrans, dim3 *tTrans, int n);
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
void fft2DSurfSetup         (cpx **in, cpx **ref, size_t *size, int sinus, int n, cudaArray **cuInputArray, cudaArray **cuOutputArray, cuSurf *inputSurfObj, cuSurf *outputSurfObj);

#endif