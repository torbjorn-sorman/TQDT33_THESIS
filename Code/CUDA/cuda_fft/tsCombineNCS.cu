#include "tsCombineNCS.cuh"

#define CRITICAL_BLOCK 8

typedef volatile int sync_buffer;

__global__ void _kernelNCS(cpx *in, cpx *out, const float angle, const float bAngle, const int depth, const int breakSize, cpx scale, const int blocks, const int n);

//#define EXPERIMENTAL
//#define SYNC_BARRIER

#ifdef SYNC_BARRIER
#define SYNC_BLOCKS(g, i, a, b) (__gpu_sync_mutex((g)));
__device__ int g_mutex;
__device__ static __inline__ void init_sync(const int tid, const int blocks)
{
    if (tid < blocks) g_mutex = 0;
}
__device__ static __inline__ void __gpu_sync_mutex(const int goalVal)
{
    if (threadIdx.x == 0) {
        ATOMIC_ADD(&g_mutex, 1);
        while (g_mutex != goalVal) {}
    }
    SYNC_THREADS;
}
#else
#define SYNC_BLOCKS(g, i) (__gpu_sync_lock_free((g) + (i)))
__device__ sync_buffer syncIn[MAX_BLOCK_SIZE];
__device__ sync_buffer syncOut[MAX_BLOCK_SIZE];
__device__ static __inline__ void init_sync(const int tid, const int blocks)
{
    if (tid < blocks) {
        syncIn[tid] = 0;
        syncOut[tid] = 0;
    }
}
__device__ static __inline__ void __gpu_sync_lock_free(const int goal)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int nBlocks = gridDim.x;

    //int cnt = 0; // Usually three cycles on 4 blocks

    if (tid == 0) {
        syncIn[bid] = goal;
    }
    if (bid == 1) {
        if (tid < nBlocks) {
            while (syncIn[tid] != goal){ /*++cnt;*/ }
        }
        SYNC_THREADS;
        if (tid < nBlocks) {
            syncOut[tid] = goal;
        }
    }
    if (tid == 0) {
        while (syncOut[bid] != goal) { /*++cnt;*/ }
    }
    SYNC_THREADS;
    //if (cnt > 2)
        //printf("\nTID: %d, BID: %d, CNT: %d\n", tid, bid, cnt);
}
#endif

__host__ int tsCombineNCS_Validate(const int n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsCombineNCS(FFT_FORWARD, &dev_in, &dev_out, n);
    tsCombineNCS(FFT_INVERSE, &dev_out, &dev_in, n);
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);

    return fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out) != 1;
}

__host__ double tsCombineNCS_Performance(const int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        tsCombineNCS(FFT_FORWARD, &dev_in, &dev_out, n);
        measures[i] = stopTimer();
    }

    fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    return avg(measures, NUM_PERFORMANCE);
}

__host__ void tsCombineNCS(fftDirection dir, cpx **dev_in, cpx **dev_out, const int n)
{
    int threads, blocks;
    int n2 = n / 2;
    float w_angle = dir * (M_2_PI / n);
    set_block_and_threads(&blocks, &threads, n2);
    int nBlock = n / blocks;
    float w_bangle = dir * (M_2_PI / nBlock);
    cpx scale = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);

    _kernelNCS KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (*dev_in, *dev_out, w_angle, w_bangle, log2_32(n), log2_32(MAX_BLOCK_SIZE), scale, blocks, n);
    
    cudaDeviceSynchronize();
    if (blocks > 1) {
        cudaError_t e = cudaGetLastError();
        if (e) printf("\nError: %s\n", cudaGetErrorString(e));        
    }
}

__device__ static __inline__ void get_input(int *in_low, int *in_high, cpx *in_lower, cpx *in_upper, cpx *in, const int tid, const unsigned int lmask, const int dist)
{
    *in_low = tid + (tid & lmask);
    *in_high = *in_low + dist;
    *in_lower = in[*in_low];
    *in_upper = in[*in_high];
}

__global__ void _kernelNCS(cpx *in, cpx *out, const float angle, const float bAngle, const int depth, const int breakSize, const cpx scale, const int blocks, const int n)
{
    extern __shared__ cpx shared[];
    int bit = depth - 1;
    int in_low, in_high;
    cpx w, in_lower, in_upper;
    in_high = n >> 1;
    if (blocks > 1) {
        int tid = (blockIdx.x * blockDim.x + threadIdx.x);
        int dist = in_high;
        int steps = 0;
        unsigned int pmask = (dist - 1) << steps;
        init_sync(tid, blocks);        
        get_input(&in_low, &in_high, &in_lower, &in_upper, in, tid, 0xFFFFFFFF << bit, dist);        
        SIN_COS_F(angle * ((tid << steps) & pmask), &w.y, &w.x);
        cpx_add_sub_mul(&(out[in_low]), &(out[in_high]), in_lower, in_upper, w);
        SYNC_BLOCKS(blocks, steps);
        while (--bit > breakSize) {
            dist = dist >> 1;
            ++steps;
            pmask = (dist - 1) << steps;
            get_input(&in_low, &in_high, &in_lower, &in_upper, out, tid, 0xFFFFFFFF << bit, dist);
            SIN_COS_F(angle * ((tid << steps) & pmask), &w.y, &w.x);
            cpx_add_sub_mul(&(out[in_low]), &(out[in_high]), in_lower, in_upper, w);
            SYNC_BLOCKS(blocks, steps);
        }
        in_high = ((n / blocks) >> 1);//(n >> (depth - 10)); // 10 = log2(size_fit_in_shared) - 1
    }

    /* Next step, run all in shared mem, copy of _kernelB(...) */
    in_low = threadIdx.x;
    in_high += in_low;
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
        SYNC_THREADS;
        SIN_COS_F(bAngle * ((in_low & (0xFFFFFFFF << steps))), &w.y, &w.x);
        cpx_add_sub_mul(&(shared[i]), &(shared[ii]), in_lower, in_upper, w);
    }

    /* Move Shared to Global */
    SYNC_THREADS;
    out[BIT_REVERSE(global_low, 32 - depth)] = cuCmulf(shared[in_low], scale);
    out[BIT_REVERSE(global_high, 32 - depth)] = cuCmulf(shared[in_high], scale);
}