#include <stdio.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "tsTobb_SB.cuh"
#include "tsHelper.cuh"
#include "tsTest.cuh"

__global__ void _kernelTSB48K(cpx *in, cpx *out, cInt depth, cFloat angle, const cpx scale, cInt n);

__host__ int tsTobb_SB_Validate(cInt n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsTobb_SB(FFT_FORWARD, &dev_in, &dev_out, n);
    tsTobb_SB(FFT_INVERSE, &dev_out, &dev_in, n);
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);

    return fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out) != 1;
}

__host__ double tsTobb_SB_Performance(cInt n)
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

__host__ void tsTobb_SB(fftDir dir, cpx **dev_in, cpx **dev_out, cInt n)
{
    int threadsPerBlock, numBlocks;
    cFloat w_angle = dir * (M_2_PI / n);
    const cpx scale = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
    set_block_and_threads(&numBlocks, &threadsPerBlock, n / 2);
    _kernelTSB48K KERNEL_ARGS3(numBlocks, threadsPerBlock, sizeof(cpx) * n)(*dev_in, *dev_out, log2_32(n), w_angle, scale, n);
    cudaDeviceSynchronize();
}

__global__ void _kernelTSB48K(cpx *in, cpx *out, cInt depth, cFloat angle, const cpx scale, cInt n)
{
    extern __shared__ cpx shared[];
    cInt tid = (blockIdx.x * blockDim.x + threadIdx.x);
    cInt lead = 32 - depth;
    int bit = depth;
    int dist = n;
    int lower;
    cpx w, in_lower, in_upper;

    /* Move Global to Shared */
    mem_gtos(tid, tid + (n >> 1), 0, shared, in);
    SYNC_THREADS;

    /* Run FFT algorithm */
    for (int steps = 0; steps < depth; ++steps) {   
        --bit;
        dist /= 2;
        lower = tid + (tid & (0xFFFFFFFF << bit));
        in_lower = shared[lower];
        in_upper = shared[lower + dist];
        SIN_COS_F(angle * ((tid << steps) & ((dist - 1) << steps)), &w.y, &w.x);
        SYNC_THREADS;
        cpx_add_sub_mul(&(shared[lower]), &(shared[lower + dist]), in_lower, in_upper, w);
        SYNC_THREADS;
    }

    /* Move Shared to Global */
    mem_stog_db(tid, tid + (n >> 1), 0, lead, scale, shared, out);
}