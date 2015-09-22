#ifndef TSHELPER_CUH
#define TSHELPER_CUH

#include "cuda_runtime.h"
#include "tsDefinitions.cuh"

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

__host__ __device__ static __inline__ unsigned int bitReverse32(unsigned int x, const int l)
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

#ifdef PRECALC_TWIDDLE
#define GLOBAL_LOW low - offset
#define GLOBAL_HIGH high - offset
#define W_OFFSET(A, O) ((A) + (O))
#else
#define GLOBAL_LOW low
#define GLOBAL_HIGH high
#define W_OFFSET(A, O) (A)
#endif

__device__ static __inline__ void globalToShared(int low, int high, int offset, cpx *shared, cpx *global)
{
    shared[low] = global[low + offset];
    shared[high] = global[high + offset];
}

__device__ static __inline__ void sharedToGlobal(int low, int high, int offset, cpx *shared, cpx *global)
{
    global[low + offset] = shared[low];
    global[high + offset] = shared[high];
}

__device__ static __inline__ void globalToShared(int low, int high, int offset, unsigned int lead, cpx *shared, cpx *global)
{
#ifdef BIT_REVERSED_OUTPUT
    shared[low] = global[GLOBAL_LOW];
    shared[high] = global[GLOBAL_HIGH];
#else
    shared[W_OFFSET(BIT_REVERSE(GLOBAL_LOW, lead), offset)] = global[GLOBAL_LOW];
    shared[W_OFFSET(BIT_REVERSE(GLOBAL_HIGH, lead), offset)] = global[GLOBAL_HIGH];
#endif
}

__device__ static __inline__ void globalToShared(int low, int high, int offset, unsigned int lead, cpx *shared_low, cpx *shared_high, cpx *global)
{
#ifdef BIT_REVERSED_OUTPUT
    shared_low[low] = global[GLOBAL_LOW];
    shared_high[low] = global[GLOBAL_HIGH];
#else
    shared_low[W_OFFSET(BIT_REVERSE(GLOBAL_LOW, lead), offset)] = global[GLOBAL_LOW];
    shared_high[W_OFFSET(BIT_REVERSE(GLOBAL_LOW, lead), offset)] = global[GLOBAL_HIGH];
#endif
}

__device__ static __inline__ void sharedToGlobal(int low, int high, int offset, cpx scale, unsigned int lead, cpx *shared, cpx *global)
{
#ifdef BIT_REVERSED_OUTPUT
    global[GLOBAL_LOW] = cuCmulf(shared[W_OFFSET(BIT_REVERSE(GLOBAL_LOW, lead), offset)], scale);
    global[GLOBAL_HIGH] = cuCmulf(shared[W_OFFSET(BIT_REVERSE(GLOBAL_HIGH, lead), offset)], scale);
#else
    global[GLOBAL_LOW] = cuCmulf(shared[low], scale);
    global[GLOBAL_HIGH] = cuCmulf(shared[high], scale);
#endif
}

__device__ static __inline__ void sharedToGlobal(int low, int high, int offset, cpx scale, unsigned int lead, cpx *shared_low, cpx *shared_high, cpx *global)
{
#ifdef BIT_REVERSED_OUTPUT
    global[GLOBAL_LOW] = cuCmulf(shared_low[W_OFFSET(BIT_REVERSE(GLOBAL_LOW, lead), offset)], scale);
    global[GLOBAL_HIGH] = cuCmulf(shared_high[W_OFFSET(BIT_REVERSE(GLOBAL_LOW, lead), offset)], scale);
#else
    global[GLOBAL_LOW] = cuCmulf(shared[low], scale);
    global[GLOBAL_HIGH] = cuCmulf(shared[high], scale);
#endif
}

//GPU lock-free synchronization function
__device__ static __inline__ void DONOTUSE__gpu_sync(int val, volatile int *a, volatile int *b, const int blocks)
{
    // blockIdx.x * blockDim.x + threadIdx.x
    int tid = threadIdx.x;
    int nBlock = gridDim.x;
    int bid = blockIdx.x;
    if (tid == 0) {
        a[bid] = val;
    }
    if (bid == 0) {
        // Dangerous?
        if (tid < nBlock) {
            while (a[tid] != val){ /* Do nothing here */ }
        }
        SYNC_THREADS;
        if (tid < nBlock) {
            b[tid] = val;
        }
    }
    // Dangerous?
    if (tid == 0) {
        while (b[bid] != val) { /* Do nothing here */ }
    }
    SYNC_THREADS;
}

//
__host__ void setBlocksAndThreads(int *numBlocks, int *threadsPerBlock, const int size);

// Texture memory?
__host__ cudaTextureObject_t specifyTexture(cpx *dev_W);

// Kernel functions
__global__ void twiddle_factors(cpx *W, const float angle, const int n);
__global__ void bit_reverse(cpx *in, cpx *out, const float scale, const int lead);
__global__ void bit_reverse(cpx *seq, fftDirection dir, const int lead, const int n);

#endif