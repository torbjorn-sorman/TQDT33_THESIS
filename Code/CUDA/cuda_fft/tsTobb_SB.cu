#include <stdio.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "tsTobb_SB.cuh"
#include "tsHelper.cuh"
#include "tsTest.cuh"

__global__ void _kernel(cpx *in, cpx *out, const int depth, const float angle, const cpx scale, const int n);
__global__ void _kernel48K(cpx *in, cpx *out, const int depth, const float angle, const cpx scale, const int n);

__host__ int tsTobb_SB_Validate(const size_t n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsTobb_SB(FFT_FORWARD, &dev_in, &dev_out, n);
    tsTobb_SB(FFT_INVERSE, &dev_out, &dev_in, n);
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);

    return fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out) != 1;
}

__host__ double tsTobb_SB_Performance(const size_t n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        tsTobb_SB(FFT_FORWARD, &dev_in, &dev_out, n);
        measures[i] = stopTimer();
    }

    fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    return avg(measures, NUM_PERFORMANCE);
}

__host__ void tsTobb_SB(fftDirection dir, cpx **dev_in, cpx **dev_out, const int n)
{
    int threadsPerBlock, numBlocks;
    const float w_angle = dir * (M_2_PI / n);
    const cpx scale = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    setBlocksAndThreads(&numBlocks, &threadsPerBlock, n);
#ifdef PRECALC_TWIDDLE
    const int sharedMem = min(sizeof(cpx) * (n + n / 2), SHARED_MEM_SIZE);
    _kernel KERNEL_ARGS3(numBlocks, threadsPerBlock, sharedMem)(*dev_in, *dev_out, log2_32(n), w_angle, scale, n);
#else
    const int sharedMem = min(sizeof(cpx) * n, SHARED_MEM_SIZE);
    _kernel48K KERNEL_ARGS3(numBlocks, threadsPerBlock, sharedMem)(*dev_in, *dev_out, log2_32(n), w_angle, scale, n);
#endif
    cudaDeviceSynchronize();
}

__host__ __device__ static __inline__ void globalToShared(int n, int tid, int offset, cpx *shared, cpx *global)
{
    shared[offset + tid] = global[tid];
    shared[offset + n / 2 + tid] = global[n / 2 + tid];
}

__host__ __device__ static __inline__ void globalToShared(int n, int tid, cpx *shared, cpx *global)
{
    shared[tid] = global[tid];
    shared[n / 2 + tid] = global[n / 2 + tid];
}

__host__ __device__ static __inline__ void globalToSharedBOR(int n, int tid, int offset, unsigned int lead, cpx *shared, cpx *global)
{
    shared[offset + BIT_REVERSE(tid, lead)] = global[tid];
    shared[offset + BIT_REVERSE(tid + n / 2, lead)] = global[n / 2 + tid];
}

__host__ __device__ static __inline__ void globalToSharedBOR(int n, int tid, unsigned int lead, cpx *shared, cpx *global)
{
    shared[BIT_REVERSE(tid, lead)] = global[tid];
    shared[BIT_REVERSE(tid + n / 2, lead)] = global[n / 2 + tid];
}

__host__ __device__ static __inline__ void sharedToGlobal(int n, int tid, int offset, cpx *shared, cpx *global)
{
    global[tid] = shared[offset + tid];
    global[n / 2 + tid] = shared[offset + n / 2 + tid];
}

__host__ __device__ static __inline__ void sharedToGlobal(int n, int tid, cpx *shared, cpx *global)
{
    global[tid] = shared[tid];
    global[n / 2 + tid] = shared[n / 2 + tid];
}

__host__ __device__ static __inline__ void sharedToGlobalBOR(int n, int tid, int offset, unsigned int lead, cpx *shared, cpx *global)
{
    global[tid] = shared[offset + BIT_REVERSE(tid, lead)];
    global[n / 2 + tid] = shared[offset + BIT_REVERSE(tid + n / 2, lead)];
}

__host__ __device__ static __inline__ void sharedToGlobalBOR(int n, int tid, unsigned int lead, cpx *shared, cpx *global)
{
    global[tid] = shared[BIT_REVERSE(tid, lead)];
    global[n / 2 + tid] = shared[BIT_REVERSE(tid + n / 2, lead)];
}

__global__ void _kernel(cpx *in, cpx *out, const int depth, const float angle, const cpx scale, const int n)
{
    extern __shared__ cpx mem[];
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    int offset = n / 2;
    int bit = depth;
    int dist = n;
    int lower;
    cpx in_lower, in_upper;

    /* Twiddle factors */
    SIN_COS_F(angle * tid, &mem[tid].y, &mem[tid].x);

    /* Move (bit-reversed?) Global to Shared */    
    globalToShared(n, tid, offset, mem, in);

    // Sync, as long as one block, not needed(?)
    SYNC_THREADS;

    /* Run FFT algorithm */
    for (int steps = 0; steps < depth; ++steps) {
        --bit;
        dist /= 2;
        lower = offset + tid + (tid & (0xFFFFFFFF << bit));
        in_lower = mem[lower];
        in_upper = mem[lower + dist];
        mem[lower] = cuCaddf(in_lower, in_upper);
        mem[lower + dist] = cuCmulf(mem[(tid << steps) & ((dist - 1) << steps)], cuCsubf(in_lower, in_upper));
        // Sync, as long as one block, not needed(?)
        SYNC_THREADS;
    }

    /* Move (bit-reversed?) Shared to Global */
    out[tid * 2] = cuCmulf(mem[offset + BIT_REVERSE(tid * 2, 32 - depth)], scale);
    out[tid * 2 + 1] = cuCmulf(mem[offset + BIT_REVERSE(tid * 2 + 1, 32 - depth)], scale);
}

__global__ void _kernel48K(cpx *in, cpx *out, const int depth, const float angle, const cpx scale, const int n)
{
    extern __shared__ cpx mem[];
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    int bit = depth;
    int dist = n;
    int lower;
    cpx w, in_lower, in_upper;

    /* Move (bit-reversed?) Global to Shared */
    globalToShared(n, tid, mem, in);

    // Sync, as long as one block, not needed(?)
    SYNC_THREADS;

    /* Run FFT algorithm */
    for (int steps = 0; steps < depth; ++steps) {
        --bit;
        dist /= 2;
        lower = tid + (tid & (0xFFFFFFFF << bit));
        in_lower = mem[lower];
        in_upper = mem[lower + dist];
        SIN_COS_F(angle * ((tid << steps) & ((dist - 1) << steps)), &w.y, &w.x);
        mem[lower] = cuCaddf(in_lower, in_upper);
        mem[lower + dist] = cuCmulf(w, cuCsubf(in_lower, in_upper));
        // Sync, as long as one block, not needed(?)
        SYNC_THREADS;
    }

    /* Move (bit-reversed?) Shared to Global */
    out[tid * 2] = cuCmulf(mem[BIT_REVERSE(tid * 2, 32 - depth)], scale);
    out[tid * 2 + 1] = cuCmulf(mem[BIT_REVERSE(tid * 2 + 1, 32 - depth)], scale);
}