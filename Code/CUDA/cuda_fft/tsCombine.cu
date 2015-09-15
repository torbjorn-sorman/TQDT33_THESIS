#include "tsCombine.cuh"

__global__ void _kernelCMBS(cpx *in, cpx *out, const float angle, const cpx scale, const int depth, unsigned int lead, const int n);
__global__ void _kernelCMB_M(cpx *in, cpx *out, const float angle, const cpx scale, const int depth, unsigned int lead, const int n);

__host__ int tsCombine_Validate(const int n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsCombine(FFT_FORWARD, &dev_in, &dev_out, n);
    tsCombine(FFT_INVERSE, &dev_out, &dev_in, n);
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
        tsCombine(FFT_FORWARD, &dev_in, &dev_out, n);
        measures[i] = stopTimer();
    }

    fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    return avg(measures, NUM_PERFORMANCE);
}

__host__ void tsCombine(fftDirection dir, cpx **dev_in, cpx **dev_out, const int n)
{
    int threadsPerBlock, numBlocks;
    const float w_angle = dir * (M_2_PI / n);
    const cpx scale = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
    const int depth = log2_32(n);
    setBlocksAndThreads(&numBlocks, &threadsPerBlock, n / 2);
    if (numBlocks == 1)
        _kernelCMBS KERNEL_ARGS3(numBlocks, threadsPerBlock, sizeof(cpx) * n)(*dev_in, *dev_out, w_angle, scale, depth, 32 - depth, n);
    else
        _kernelCMB_M KERNEL_ARGS3(numBlocks, threadsPerBlock, sizeof(cpx) * (n / numBlocks))(*dev_in, *dev_out, w_angle, scale, depth, 32 - depth, n);
    cudaDeviceSynchronize();
}

__global__ void _kernelCMBS(cpx *in, cpx *out, const float angle, const cpx scale, const int depth, unsigned int lead, const int n)
{
    extern __shared__ cpx shared[];
    cpx w, in_lower, in_upper;
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    const int in_high = (n >> 1) + tid;
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

__global__ void _kernelCMB_M(cpx *in, cpx *out, const float angle, const cpx scale, const int depth, unsigned int lead, const int n)
{
    extern __shared__ cpx shared[];
    cpx w1, w2, in_lower1, in_upper1, in_lower2, in_upper2;
    const int tid1 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int tid2 = tid1 + 1;
    const int in_high1 = ((n >> 1) + tid1) % blockDim.x;
    const int in_high2 = ((n >> 1) + tid2) % blockDim.x;
    const int in_low1 = tid1 % blockDim.x;
    const int in_low2 = tid2 % blockDim.x;
    const int i1 = (tid1 << 1);
    const int i2 = (tid2 << 1);
    const int ii1 = i1 + 1;
    const int ii2 = i2 + 1;

    /* Move (bit-reversed?) Global to Shared */
    globalToShared(tid1, in_high1, 0, lead, shared, in);
    globalToShared(tid2, in_high2, 0, lead, shared, in);

    /* Run FFT algorithm */
    for (int steps = 0; steps < depth; ++steps) {
        SYNC_THREADS;
        in_lower1 = shared[in_low1];
        in_lower2 = shared[in_low2];
        in_upper1 = shared[in_high1];
        in_upper2 = shared[in_high2];
        SIN_COS_F(angle * ((tid1 & (0xffffffff << steps))), &w1.y, &w1.x);
        SIN_COS_F(angle * ((tid2 & (0xffffffff << steps))), &w2.y, &w2.x);
        SYNC_THREADS;
        cpx_add_sub_mul(&(shared[i1]), &(shared[ii1]), in_lower1, in_upper1, w1);
        cpx_add_sub_mul(&(shared[i2]), &(shared[ii2]), in_lower2, in_upper2, w2);
    }

    /* Move Shared to Global (index bit-reversed) */
    SYNC_THREADS;
    sharedToGlobal(tid1, in_high1, 0, scale, lead, shared, out);
    sharedToGlobal(tid2, in_high2, 0, scale, lead, shared, out);
}