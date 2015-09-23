#include <stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "math.h"
#include "tsConstantGeometry_DP.cuh"
#include "tsHelper.cuh"
#include "tsTest.cuh"

__global__ void _kernelCGDP(cpx *in, cpx *out, const float angle, const cpx scale, int depth, unsigned int lead, const int n2);
__global__ void _kernelCGDP48K(cpx *in, cpx *out, const float angle, const cpx scale, int depth, unsigned int lead, const int n2);

__host__ int tsConstantGeometry_DP_Validate(const int n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsConstantGeometry_DP(FFT_FORWARD, &dev_in, &dev_out, n);
    tsConstantGeometry_DP(FFT_INVERSE, &dev_out, &dev_in, n);    
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);

    return fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out) != 1;
}

__host__ double tsConstantGeometry_DP_Performance(const int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        tsConstantGeometry_DP(FFT_FORWARD, &dev_in, &dev_out, n);
        measures[i] = stopTimer();
    }

    fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    return avg(measures, NUM_PERFORMANCE);
}

__host__ void tsConstantGeometry_DP(fftDirection dir, cpx **dev_in, cpx **dev_out, const int n)
{
    int threadsPerBlock, numBlocks;
    const float w_angle = dir * (M_2_PI / n);
    const cpx scale = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);   
    const int depth = log2_32(n);
    set_block_and_threads(&numBlocks, &threadsPerBlock, n);
    int sharedMem = sizeof(cpx) * n;
    sharedMem = sharedMem > SHARED_MEM_SIZE ? SHARED_MEM_SIZE : sharedMem;
    _kernelCGDP48K KERNEL_ARGS3(numBlocks, threadsPerBlock, sharedMem)(*dev_in, *dev_out, w_angle, scale, depth, 32 - depth, n / 2);
    cudaDeviceSynchronize();
}

__global__ void _kernelCGDP(cpx *in, cpx *out, const float angle, const cpx scale, int depth, unsigned int lead, const int n2)
{
    extern __shared__ cpx mem[]; // sizeof(cpx) * (n + n + n/n)  
    cpx in_lower, in_upper;
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    int in_low = n2 + tid;
    int in_high = n2 + in_low;
    int i = n2 + tid * 2;
    int ii = i + 1;

    /* Twiddle factors */
    SIN_COS_F(angle * tid, &mem[tid].y, &mem[tid].x);

    /* Move (bit-reversed?) Global to Shared */
    mem[n2 + tid * 2] = in[tid * 2];
    mem[n2 + tid * 2 + 1] = in[tid * 2 + 1];
    // Sync, as long as one block, not needed(?)
    SYNC_THREADS;

    /* Run FFT algorithm */
    for (int steps = 0; steps < depth; ++steps) {
        in_lower = mem[in_low];
        in_upper = mem[in_high];
        // Sync, as long as one block, not needed(?)
        SYNC_THREADS;
        mem[i] = cuCaddf(in_lower, in_upper);
        mem[ii] = cuCmulf(mem[(tid & (0xffffffff << steps))], cuCsubf(in_lower, in_upper));
        // A = B*(C-D) = B*C - B*D
        // Look for some single precision intrisics ;-)
        
        // Sync, as long as one block, not needed(?)
        SYNC_THREADS;
    }
    /* Move Shared to Global (index bit-reversed) */
    out[tid * 2] = cuCmulf(mem[n2 + BIT_REVERSE(tid * 2, lead)], scale);
    out[tid * 2 + 1] = cuCmulf(mem[n2 + BIT_REVERSE(tid * 2 + 1, lead)], scale);
}

__global__ void _kernelCGDP48K(cpx *in, cpx *out, const float angle, const cpx scale, int depth, unsigned int lead, const int n2)
{
    extern __shared__ cpx mem[]; // sizeof(cpx) * (n + n + n/n)  
    cpx w, in_lower, in_upper;
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    int in_low = n2 + tid;
    int in_high = n2 + in_low;
    int i = tid * 2;
    int ii = i + 1;

    /* Move (bit-reversed?) Global to Shared */
    mem[tid * 2] = in[tid * 2];
    mem[tid * 2 + 1] = in[tid * 2 + 1];
    // Sync, as long as one block, not needed(?)
    SYNC_THREADS;

    /* Run FFT algorithm */
    for (int steps = 0; steps < depth; ++steps) {
        in_lower = mem[in_low];
        in_upper = mem[in_high];
        // Sync, as long as one block, not needed(?)
        SIN_COS_F(angle * ((tid & (0xffffffff << steps))), &w.y, &w.x);
        SYNC_THREADS;
        mem[i] = cuCaddf(in_lower, in_upper);
        mem[ii] = cuCmulf(w, cuCsubf(in_lower, in_upper));
        // A = B*(C-D) = B*C - B*D
        // Look for some single precision intrisics ;-)

        // Sync, as long as one block, not needed(?)
        SYNC_THREADS;
    }

    /* Move Shared to Global (index bit-reversed) */
    out[tid * 2] = cuCmulf(mem[BIT_REVERSE(tid * 2, lead)], scale);
    out[tid * 2 + 1] = cuCmulf(mem[BIT_REVERSE(tid * 2 + 1, lead)], scale);
}