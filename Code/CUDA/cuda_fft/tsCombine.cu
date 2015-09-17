#include "tsCombine.cuh"

__global__ void _kernelAll(cpx *in, cpx *out, const float angle, const unsigned int lmask, const unsigned int pmask, const int steps, const int dist);
__global__ void _kernelBlock(cpx *in, cpx *out, const float angle, const cpx scale, const int depth, const unsigned int lead, const int n2);
__global__ void _kernelB(cpx *in, cpx *out, const float angle, const cpx scale, const int depth, const unsigned int lead, const int n2);

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
        //cudaDeviceSynchronize();
        //setBlocksAndThreads(&blocks, &threads, n);
        //bit_reverse KERNEL_ARGS2(blocks, threads)(*dev_in, *dev_out, (dir == FFT_FORWARD ? 1.f : 1.f / n), 32 - depth);
    }
    else {
        _kernelB KERNEL_ARGS3(1, threads, sizeof(cpx) * n)(*dev_in, *dev_out, w_angle, scaleCpx, depth, lead, n2);
    }
    cudaDeviceSynchronize();
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

    printf("(%d, %d) -> reversed(%d, %d) / (%d, %d)\n", in_low, in_high, BIT_REVERSE(global_low, lead), BIT_REVERSE(global_high, lead), global_low, global_high);

    //out[global_low] = shared[in_low];
    //out[global_high] = shared[in_high];
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