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
    setBlocksAndThreads(&numBlocks, &threadsPerBlock, n / 2);
#ifdef PRECALC_TWIDDLE
    _kernelCGSB KERNEL_ARGS3(numBlocks, threadsPerBlock, sizeof(cpx) * (n + n / 2))(*dev_in, *dev_out, w_angle, scale, depth, 32 - depth, n);
#else
    _kernelCGSB48K KERNEL_ARGS3(numBlocks, threadsPerBlock, sizeof(cpx) * (n + n / 32))(*dev_in, *dev_out, w_angle, scale, depth, 32 - depth, n);
#endif
    //printf("sz: %d\n", sizeof(cpx) * n);
    if (cudaError_t error = cudaGetLastError())
        printf("Error: %s Memsize: %d n: %d\n", cudaGetErrorString(error), sizeof(cpx) * n / numBlocks, n);
    cudaDeviceSynchronize();
}

__global__ void _kernelCGSB48K(cpx *in, cpx *out, const float angle, const cpx scale, const int depth, const unsigned int lead, const int n)
{
    extern __shared__ cpx shared[];
    cpx w, tmp, in_lower, in_upper;
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
        shared[i].x = in_lower.x + in_upper.x;
        shared[i].y = in_lower.y + in_upper.y;
        tmp.x = in_lower.x - in_upper.x;
        tmp.y = in_lower.y - in_upper.y;
        shared[ii].x = tmp.x * w.x - tmp.y * w.y;
        shared[ii].y = tmp.x * w.y + tmp.y * w.x;
    }

    /* Move Shared to Global (index bit-reversed) */
    SYNC_THREADS;
    sharedToGlobal(tid, in_high, 0, scale, lead, shared, out);
}

__global__ void _kernelCGSB(cpx *in, cpx *out, const float angle, const cpx scale, int depth, const unsigned int lead, const int n)
{
    extern __shared__ cpx shared[];
    cpx w, tmp, in_lower, in_upper;
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
        w = shared[(tid & (0xffffffff << steps))];
        shared[i].x = in_lower.x + in_upper.x;
        shared[i].y = in_lower.y + in_upper.y;
        tmp.x = in_lower.x - in_upper.x;
        tmp.y = in_lower.y - in_upper.y;
        shared[ii].x = tmp.x * w.x - tmp.y * w.y;
        shared[ii].y = tmp.x * w.y + tmp.y * w.x;
    }

    /* Move Shared to Global (index bit-reversed) */
    SYNC_THREADS;
    sharedToGlobal(in_low, in_high, n2, scale, lead, shared, out);
}