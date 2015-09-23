#include "tsCombine.cuh"

typedef int syncVal;

__global__ void _kernelAll(cpx *in, cpx *out, const float angle, const unsigned int lmask, const unsigned int pmask, const int steps, const int dist);
__global__ void _kernelBlock(cpx *in, cpx *out, const float angle, const cpx scale, const int depth, const unsigned int lead, const int n2);
__global__ void _kernelB(cpx *in, cpx *out, const float angle, const cpx scale, const int depth, const unsigned int lead, const int n2);
__global__ void _kernelNoLockSynch(cpx *in, cpx *out, syncVal *a, syncVal *b, const float angle, const float bAngle, const int depth, const int breakSize, cpx scale, const int blocks, const int n);

#define EXPERIMENTAL

__host__ int tsCombine_Validate(const int n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
#ifdef EXPERIMENTAL
    tsCombine2(FFT_FORWARD, &dev_in, &dev_out, n);
    //tsCombine2(FFT_INVERSE, &dev_out, &dev_in, n);
#else
    tsCombine(FFT_FORWARD, &dev_in, &dev_out, n);
    tsCombine(FFT_INVERSE, &dev_out, &dev_in, n);
#endif
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);

    return fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out) != 1;
}

__host__ double tsCombine_Performance(const int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
#ifdef EXPERIMENTAL
        //tsCombine2(FFT_FORWARD, &dev_in, &dev_out, n);
#else
        tsCombine(FFT_FORWARD, &dev_in, &dev_out, n);
#endif
        measures[i] = stopTimer();
    }

    fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    return avg(measures, NUM_PERFORMANCE);
}

#define REVERSE_ON_OUT

__host__ void tsCombine(fftDirection dir, cpx **dev_in, cpx **dev_out, const int n)
{
    int threads, blocks;
    float w_angle = dir * (M_2_PI / n);
    const int depth = log2_32(n);
    const int lead = 32 - depth;
    const int n2 = (n / 2);
    const int breakSize = log2_32(MAX_BLOCK_SIZE);
    const cpx scaleCpx = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
    int steps = 0;
    int bit = depth - 1;
    int dist = n2;

    // Set number of blocks and threads
    setBlocksAndThreads(&blocks, &threads, n2);

    if (blocks > 1) {
        // Sync at device level until 
        _kernelAll KERNEL_ARGS2(blocks, threads)(*dev_in, *dev_out, w_angle, 0xFFFFFFFF << bit, (dist - 1) << steps, steps, dist);
        cudaDeviceSynchronize();
        while (--bit > breakSize) {
            dist = dist >> 1;
            ++steps;
            _kernelAll KERNEL_ARGS2(blocks, threads)(*dev_out, *dev_out, w_angle, 0xFFFFFFFF << bit, (dist - 1) << steps, steps, dist);
            cudaDeviceSynchronize();
        }
        const int nBlock = n / blocks;
        w_angle = dir * (M_2_PI / nBlock);
        _kernelBlock KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock)(*dev_out, *dev_in, w_angle, scaleCpx, bit + 1, lead, nBlock / 2);
        swap(dev_in, dev_out);
    }
    else {
        _kernelB KERNEL_ARGS3(1, threads, sizeof(cpx) * n)(*dev_in, *dev_out, w_angle, scaleCpx, depth, lead, n2);
    }
    cudaDeviceSynchronize();
}

__host__ void prepSync(syncVal **dev_a, syncVal **dev_b, int blocks)
{
    *dev_a = 0;
    *dev_b = 0;
    if (blocks == 1)
        return;
    cudaMalloc((void**)dev_a, sizeof(int) * (blocks + 1));
    cudaMalloc((void**)dev_b, sizeof(int) * (blocks + 1));
}

#include <time.h>
#include <stdlib.h>

__host__ void tsCombine2(fftDirection dir, cpx **dev_in, cpx **dev_out, const int n)
{
    int threads, blocks;
    syncVal *dev_a, *dev_b;
    int n2 = n / 2;
    float w_angle = dir * (M_2_PI / n);
    setBlocksAndThreads(&blocks, &threads, n2);
    prepSync(&dev_a, &dev_b, blocks);
    int nBlock = n / blocks;
    float w_bangle = dir * (M_2_PI / nBlock);
    cpx scale = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);

    srand((unsigned int)time(NULL));
    _kernelNoLockSynch KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (*dev_in, *dev_out, dev_a, dev_b, w_angle, w_bangle, log2_32(n), log2_32(MAX_BLOCK_SIZE), scale, blocks, n);
    cudaDeviceSynchronize();
    if (blocks > 1) {
        cudaError_t e = cudaGetLastError();
        if (e) printf("\nError: %s\n", cudaGetErrorString(e));
        cudaFree((void *)dev_a);
        cudaFree((void *)dev_b);
    }
}

__device__ static __inline__ void valuesIn(int *in_low, int *in_high, cpx *in_lower, cpx *in_upper, cpx *in, int tid, int lmask, int dist)
{
    *in_low = tid + (tid & lmask);
    *in_high = *in_low + dist;
    *in_lower = in[*in_low];
    *in_upper = in[*in_high];
}

#ifdef __CUDACC__
#define ATOMIC_CAS(a,c,v) (atomicCAS((int *)(a),(int)(c),(int)(v)))
#define THREAD_FENCE __threadfence()
#else
#define ATOMIC_CAS(a,c,v) 1
#define THREAD_FENCE
#endif

#define CRITICAL_BLOCK 8
/*
//blocks shared variable
__device__ int g_mutex;

//centralized barrier function
__device__ void __gpu_sync2(int goalVal){
    // in each block, only thread 0 synchronizes
    // with the other blocks
    if (threadIdx.x == 0) {
        atomicAdd(&g_mutex, 1);
        // wait for the other blocks
        //for (;;) { if (g_mutex == goalVal) break; }
        printf("\nblock: %d", blockIdx.x);
        while (g_mutex < goalVal) {}
    }
    SYNC_THREADS;}*/
//GPU lock-free synchronization function
__device__ void __gpu_sync(const int goal, syncVal *a, syncVal *b)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int nBlocks = gridDim.x;

    if (bid * blockDim.x + tid == 0)
        if (nBlocks == CRITICAL_BLOCK) printf("\nGoal: %d GridDim.x: %d  blockDim.x: %d ", goal, gridDim.x, blockDim.x);

    if (tid == 0) {
        a[bid] = goal;
        if (nBlocks == CRITICAL_BLOCK) printf("\n\tIN %d\t%d", goal, bid);
    }
    if (bid == 1) {
        if (tid < nBlocks) {
            while (goal != ATOMIC_CAS(&(a[tid]), goal, goal)){}
            //while (a[tid] != goal){ /* ATOMIC_CAS(&(a[tid]), goal, goal); */ }
            if (nBlocks == CRITICAL_BLOCK) printf("\n\t\tIN %d\t%d", goal, tid);
        }
        SYNC_THREADS;
        if (tid < nBlocks) {
            if (nBlocks == CRITICAL_BLOCK) printf("\n\t\tUT %d\t%d", goal, tid);
            b[tid] = goal;
        }

    }
    if (tid == 0) {
        while (b[bid] != goal) { /* ATOMIC_CAS(&(b[tid]), goal, goal); */ }
        if (nBlocks == CRITICAL_BLOCK) printf("\n\tUT %d\t%d", goal, bid);
    }
    SYNC_THREADS;
}

__global__ void _kernelNoLockSynch(cpx *in, cpx *out, syncVal *a, syncVal *b, const float angle, const float bAngle, const int depth, const int breakSize, cpx scale, const int blocks, const int n)
{
    extern __shared__ cpx shared[];    
    int bit = depth - 1;
    int in_low, in_high;
    cpx w, in_lower, in_upper;
    if (blocks > 1) {
        int tid = (blockIdx.x * blockDim.x + threadIdx.x);
        if (blockIdx.x == 0 && threadIdx.x < blocks) {
            //g_mutex = 0;
            a[threadIdx.x] = 0;
            b[threadIdx.x] = 0;
        }
        int dist = n / 2;
        int steps = 0;
        unsigned int pmask = (dist - 1) << steps;
        valuesIn(&in_low, &in_high, &in_lower, &in_upper, in, tid, 0xFFFFFFFF << bit, dist);
        SIN_COS_F(angle * ((tid << steps) & pmask), &w.y, &w.x);
        cpx_add_sub_mul(&(out[in_low]), &(out[in_high]), in_lower, in_upper, w);
        // TODO DO NOT USE STEP
        __gpu_sync(blocks + steps + (angle > 1), a, b);
        //__gpu_sync2(blocks);
        while (--bit > breakSize) {
            dist = dist >> 1;
            ++steps;
            pmask = (dist - 1) << steps;
            valuesIn(&in_low, &in_high, &in_lower, &in_upper, out, tid, 0xFFFFFFFF << bit, dist);
            SIN_COS_F(angle * ((tid << steps) & pmask), &w.y, &w.x);
            cpx_add_sub_mul(&(out[in_low]), &(out[in_high]), in_lower, in_upper, w);
            // TODO DO NOT USE STEP
            __gpu_sync(blocks + steps + (angle > 1), a, b);
            //__gpu_sync2(blocks);
        }
    }

    /* Next step, run all in shared mem, copy of _kernelB(...) */
    in_low = threadIdx.x;
    in_high = ((n / blocks) / 2) + in_low;
    const int global_low = in_low + blockIdx.x * blockDim.x * 2;
    const int global_high = in_high + blockIdx.x * blockDim.x * 2;
    const int i = (in_low << 1);
    const int ii = i + 1;

    /* Move Global to Shared */
    if (blocks > 1) {
        shared[in_low] = out[global_low];
        shared[in_high] = out[global_high];
    }
    else {
        shared[in_low] = in[global_low];
        shared[in_high] = in[global_high];
    }

    /* Run FFT algorithm */
    ++bit;
    for (int steps = 0; steps < bit; ++steps) {
        SYNC_THREADS;
        in_lower = shared[in_low];
        in_upper = shared[in_high];
        SIN_COS_F(bAngle * ((in_low & (0xFFFFFFFF << steps))), &w.y, &w.x);
        SYNC_THREADS;
        cpx_add_sub_mul(&(shared[i]), &(shared[ii]), in_lower, in_upper, w);
    }

    /* Move Shared to Global */
    SYNC_THREADS;
    out[BIT_REVERSE(global_low, 32 - depth)] = cuCmulf(shared[in_low], scale);
    out[BIT_REVERSE(global_high, 32 - depth)] = cuCmulf(shared[in_high], scale);
}

// Take no usage of shared mem yet...
__global__ void _kernelAll(cpx *in, cpx *out, const float angle, const unsigned int lmask, const unsigned int pmask, const int steps, const int dist)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    int l = tid + (tid & lmask);
    int u = l + dist;
    cpx in_lower = in[l];
    cpx in_upper = in[u];
    cpx w;
    SIN_COS_F(angle * ((tid << steps) & pmask), &w.y, &w.x);
    cpx_add_sub_mul(&(out[l]), &(out[u]), in_lower, in_upper, w);
}

// Full usage of shared mem!
__global__ void _kernelBlock(cpx *in, cpx *out, const float angle, const cpx scale, const int depth, const unsigned int lead, const int n2)
{
    extern __shared__ cpx shared[];
    cpx w, in_lower, in_upper;
    const int offset = blockIdx.x * blockDim.x * 2;
    const int in_low = threadIdx.x;
    const int in_high = n2 + in_low;
    const int global_low = in_low + offset;
    const int global_high = in_high + offset;
    const int i = (in_low << 1);
    const int ii = i + 1;

    /* Move Global to Shared */
    shared[in_low] = in[global_low];
    shared[in_high] = in[global_high];

    /* Run FFT algorithm */
    for (int steps = 0; steps < depth; ++steps) {
        SYNC_THREADS;
        in_lower = shared[in_low];
        in_upper = shared[in_high];
        SIN_COS_F(angle * ((in_low & (0xffffffff << steps))), &w.y, &w.x);
        SYNC_THREADS;
        cpx_add_sub_mul(&(shared[i]), &(shared[ii]), in_lower, in_upper, w);
    }

    /* Move Shared to Global */
    SYNC_THREADS;
    out[BIT_REVERSE(global_low, lead)] = cuCmulf(shared[in_low], scale);
    out[BIT_REVERSE(global_high, lead)] = cuCmulf(shared[in_high], scale);
}

__global__ void _kernelB(cpx *in, cpx *out, const float angle, const cpx scale, const int depth, const unsigned int lead, const int n2)
{
    extern __shared__ cpx shared[];
    cpx w, in_lower, in_upper;
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    const int in_high = n2 + tid;
    const int i = (tid << 1);
    const int ii = i + 1;

    /* Move (bit-reversed?) Global to Shared */
    globalToShared(tid, in_high, 0, lead, shared, in);

    /* Run FFT algorithm */
    for (int steps = 0; steps < depth; ++steps) {
        SYNC_THREADS;
        in_lower = shared[tid];
        in_upper = shared[in_high];
        SIN_COS_F(angle * ((tid & (0xffffffff << steps))), &w.y, &w.x);
        SYNC_THREADS;
        cpx_add_sub_mul(&(shared[i]), &(shared[ii]), in_lower, in_upper, w);
    }

    /* Move Shared to Global (index bit-reversed) */
    SYNC_THREADS;
    sharedToGlobal(tid, in_high, 0, scale, lead, shared, out);
}