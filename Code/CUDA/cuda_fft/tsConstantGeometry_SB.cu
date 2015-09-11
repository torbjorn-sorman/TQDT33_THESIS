#include <stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "math.h"
#include "tsConstantGeometry_SB.cuh"
#include "tsHelper.cuh"
#include "tsTest.cuh"

__global__ void _kernelCGSB(cpx *in, cpx *out, const float angle, const cpx scale, int depth, unsigned int lead, const int n2);
__global__ void _kernelCGSB48K(cpx *in, cpx *out, const float angle, const cpx scale, int depth, unsigned int lead, const int n2);

__host__ int tsConstantGeometry_SB_Validate(const int n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsConstantGeometry_SB(FFT_FORWARD, &dev_in, &dev_out, n);
    //cudaMemcpy(out, dev_out, n * sizeof(cpx), cudaMemcpyDeviceToHost);
    tsConstantGeometry_SB(FFT_INVERSE, &dev_out, &dev_in, n);    
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);
    /*
    printf("\n");
    console_print(out, n);
    printf("\n");
    console_print(in, n);
    printf("\n");
    */
    return fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out) != 1;
}

__host__ double tsConstantGeometry_SB_Performance(const int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        tsConstantGeometry_SB(FFT_FORWARD, &dev_in, &dev_out, n);
        measures[i] = stopTimer();
    }

    fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    return avg(measures, NUM_PERFORMANCE);
}

__host__ void tsConstantGeometry_SB(fftDirection dir, cpx **dev_in, cpx **dev_out, const int n)
{
    
    int threadsPerBlock, numBlocks;
    const float w_angle = dir * (M_2_PI / n);
    const cpx scale = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);   
    const int depth = log2_32(n);
    setBlocksAndThreads(&numBlocks, &threadsPerBlock, n);
#ifdef PRECALC_TWIDDLE
    _kernelCGSB KERNEL_ARGS2(numBlocks, threadsPerBlock)(*dev_in, *dev_out, w_angle, scale, depth, 32 - depth, n);
#else
    _kernelCGSB48K KERNEL_ARGS2(numBlocks, threadsPerBlock)(*dev_in, *dev_out, w_angle, scale, depth, 32 - depth, n);
#endif
    cudaDeviceSynchronize();
}

__global__ void _kernelCGSB(cpx *in, cpx *out, const float angle, const cpx scale, int depth, const unsigned int lead, const int n)
{
    __shared__ cpx shared[6144];
    cpx in_lower, in_upper;
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);  
    const int n2 = (n >> 1);
    int in_low = n2 + tid;  
    int in_high = n + tid;
    int i = n2 + (tid << 1);
    int ii = i + 1;

    /* Twiddle factors */
    SIN_COS_F(angle * tid, &shared[tid].y, &shared[tid].x);

    /* Move Global to Shared */
    globalToShared(in_low, in_high, n2, lead, shared, in);
       

    /* Run FFT algorithm */
    for (int steps = 0; steps < depth; ++steps) {
        SYNC_THREADS;
        in_lower = shared[in_low];
        in_upper = shared[in_high];
        SYNC_THREADS;
        cpx tw = shared[(tid & (0xffffffff << steps))];
        shared[i] = cuCaddf(in_lower, in_upper);
        shared[ii] = cuCmulf(tw, cuCsubf(in_lower, in_upper));        
    }

    /* Move Shared to Global (index bit-reversed) */
    SYNC_THREADS;
    sharedToGlobal(in_low, in_high, n2, scale, lead, shared, out);
}

__global__ void _kernelCGSB48K(cpx *in, cpx *out, const float angle, const cpx scale, const int depth, const unsigned int lead, const int n)
{
    __shared__ cpx shared_low[3071];
    __shared__ cpx shared_mid[1];
    __shared__ cpx shared_high[3071];
    cpx w, in_lower, in_upper;    
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    cpx *output = tid < (n >> 1) ? shared_low : shared_high;
    int in_high = (n >> 1) + tid;
    int i = (tid << 1);
    int ii = i + 1;

    /* Move (bit-reversed?) Global to Shared */
    globalToShared(tid, in_high, 0, lead, shared_low, shared_high, in);

    /* Run FFT algorithm */
    for (int steps = 0; steps < depth; ++steps) {
        SYNC_THREADS;
        in_lower = shared_low[tid];
        in_upper = shared_high[tid];
        SIN_COS_F(angle * ((tid & (0xffffffff << steps))), &w.y, &w.x);
        SYNC_THREADS;
        output[i] = cuCaddf(in_lower, in_upper);
        output[ii] = cuCmulf(w, cuCsubf(in_lower, in_upper));
    }

    /* Move Shared to Global (index bit-reversed) */
    SYNC_THREADS;
    sharedToGlobal(tid, in_high, 0, scale, lead, shared_low, shared_high, out);
}

__global__ void _kernelCGSB48K_back(cpx *in, cpx *out, const float angle, const cpx scale, const int depth, const unsigned int lead, const int n)
{
    __shared__ cpx shared[6144];
    cpx w, in_lower, in_upper;
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    int in_high = (n >> 1) + tid;
    int i = (tid << 1);
    int ii = i + 1;

    /* Move (bit-reversed?) Global to Shared */
    globalToShared(tid, in_high, 0, lead, shared, in);

    /* Run FFT algorithm */
    for (int steps = 0; steps < depth; ++steps) {
        SYNC_THREADS;
        in_lower = shared[tid];
        in_upper = shared[in_high];
        SIN_COS_F(angle * ((tid & (0xffffffff << steps))), &w.y, &w.x);
        SYNC_THREADS;
        shared[i] = cuCaddf(in_lower, in_upper);
        shared[ii] = cuCmulf(w, cuCsubf(in_lower, in_upper));
    }

    /* Move Shared to Global (index bit-reversed) */
    //SYNC_THREADS;
    sharedToGlobal(tid, in_high, 0, scale, lead, shared, out);
}