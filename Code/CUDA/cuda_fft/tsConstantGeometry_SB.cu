#include <stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "math.h"
#include "tsConstantGeometry_SB.cuh"
#include "tsHelper.cuh"
#include "tsTest.cuh"

__global__ void _kernelCGSB48K(cpx *in, cpx *out, const float angle, const cpx scale, const int depth, unsigned int lead, const int n);

__host__ int tsConstantGeometry_SB_Validate(const int n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsConstantGeometry_SB(FFT_FORWARD, &dev_in, &dev_out, n);
    tsConstantGeometry_SB(FFT_INVERSE, &dev_out, &dev_in, n);
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);

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
    set_block_and_threads(&numBlocks, &threadsPerBlock, n / 2);
    _kernelCGSB48K KERNEL_ARGS3(numBlocks, threadsPerBlock, sizeof(cpx) * n)(*dev_in, *dev_out, w_angle, scale, depth, 32 - depth, n);
    cudaDeviceSynchronize();
}

__global__ void _kernelCGSB48K(cpx *in, cpx *out, const float angle, const cpx scale, const int depth, const unsigned int lead, const int n)
{
    extern __shared__ cpx shared[];
    cpx w, in_lower, in_upper;
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    const int in_high = (n >> 1) + tid;
    const int i = (tid << 1);
    const int ii = i + 1;

    /* Move (bit-reversed?) Global to Shared */
    mem_gtos_db(tid, in_high, 0, lead, shared, in);

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
    mem_stog_db(tid, in_high, 0, lead, scale, shared, out);
}