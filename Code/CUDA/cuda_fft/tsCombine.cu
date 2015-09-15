#include "tsCombine.cuh"

__global__ void _kernelAll(cpx *in, cpx *out, const float angle, const unsigned int lmask, const unsigned int pmask, const int steps, const int dist);
__global__ void _kernelBlock(cpx *in, cpx *out, const float angle, const int depth, const int n2);
__global__ void _kernelB(cpx *in, cpx *out, const float angle, const cpx scale, const int depth, const unsigned int lead, const int n2);

__host__ int tsCombine_Validate(const int n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    int mblock = (n > MAX_BLOCK_SIZE * 2) && 1;
    
    if (mblock) {
        printf("\nIN:\n");
        console_print(in, n);
        printf("\n");
    }
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsCombine(FFT_FORWARD, &dev_in, &dev_out, n);

    if (mblock) {
        cudaMemcpy(out, dev_out, n * sizeof(cpx), cudaMemcpyDeviceToHost);
        printf("\nOUT:\n");
        console_print(out, n);
        printf("\n");
    }

    tsCombine(FFT_INVERSE, &dev_out, &dev_in, n);
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);
    /*
    if (mblock) {
        printf("\n");
        console_print(in, 10);
        printf("\n");
    }
    */
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
    const int depth = log2_32(n);
    const int n2 = (n / 2);
    const int breakSize = log2_32(MAX_BLOCK_SIZE);
    int steps = 0;
    int bit = depth - 1;
    int dist = n2;

    // Set number of blocks and threads
    setBlocksAndThreads(&numBlocks, &threadsPerBlock, n2);
    if (numBlocks > 1) {
        //printf("lvl 1\n");
        const float scale = dir == FFT_FORWARD ? 1.f : 1.f / n;
        // Sync at device level until 
        _kernelAll KERNEL_ARGS2(numBlocks, threadsPerBlock)(*dev_in, *dev_out, w_angle, 0xFFFFFFFF << bit, (dist - 1) << steps, steps, dist);
        cudaDeviceSynchronize();
        while (bit-- > breakSize) {
            dist = dist >> 1;
            ++steps;
            _kernelAll KERNEL_ARGS2(numBlocks, threadsPerBlock)(*dev_out, *dev_out, w_angle, 0xFFFFFFFF << bit, (dist - 1) << steps, steps, dist);
            cudaDeviceSynchronize();
        }
        const int nBlock = n / numBlocks;
        printf("\nblocks: %d\tang: %f\tbit: %d\tn2: %d\n", numBlocks, dir * (M_2_PI / nBlock), log2_32(nBlock), nBlock / 2);
        
        cpx *out = (cpx *)malloc(sizeof(cpx) * n);
        cudaMemcpy(out, *dev_out, n * sizeof(cpx), cudaMemcpyDeviceToHost);
        console_print(out, n);


        _kernelBlock KERNEL_ARGS3(numBlocks, threadsPerBlock, sizeof(cpx) * nBlock)(*dev_out, *dev_in, dir * (M_2_PI / nBlock), log2_32(nBlock), nBlock / 2);
        cudaDeviceSynchronize();

        setBlocksAndThreads(&numBlocks, &threadsPerBlock, n);
        bit_reverse KERNEL_ARGS2(numBlocks, threadsPerBlock)(*dev_in, *dev_out, scale, 32 - depth);      
    }
    else {
        const cpx scale = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
        _kernelB KERNEL_ARGS3(1, threadsPerBlock, sizeof(cpx) * n)(*dev_in, *dev_out, w_angle, scale, depth, 32 - depth, n2);
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
__global__ void _kernelBlock(cpx *in, cpx *out, const float angle, const int depth, const int n2)
{
    extern __shared__ cpx shared[];
    cpx w, in_lower, in_upper;
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    const int in_low = threadIdx.x;
    const int in_high = n2 / 2 + in_low;
    const int i = (in_low << 1);
    const int ii = i + 1;
        
    /* Move Global to Shared */
    shared[in_low] = in[in_low + blockIdx.x * blockDim.x];
    shared[in_high] = in[in_high + blockIdx.x * blockDim.x];

    if (tid == n2 || tid == 0) {
        printf("%d\t(%d, %d) -> (%d, %d)\t%d\t%d\n", tid, in_low, in_high, i, ii, blockIdx.x, blockIdx.x * blockDim.x);
    }

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
    out[in_low + blockIdx.x * blockDim.x] = shared[in_low];
    out[in_high + blockIdx.x * blockDim.x] = shared[in_high];
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