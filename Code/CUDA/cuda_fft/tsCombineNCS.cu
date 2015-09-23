#include "tsCombineNCS.cuh"

__global__ void _kernelNCS(cpx *in, cpx *out, const float angle, const float bAngle, const int depth, const int breakSize, cpx scale, const int blocks, const int n);
__global__ void _kernelNCS2D(cpx *in, const int depth, const float angle, const float bAngle, cpx scale, const int n);

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

__host__ int tsCombineNCS2D_Test(const int n)
{
    char file[30];
    char outfile[40];
    sprintf_s(file, 30, "splash/%u.ppm", n);
    sprintf_s(outfile, 40, "out/img/splash_%u.ppm", n);
    int sz;
    cpx *in = read_image(file, &sz);
    if (sz != n) return -1;
    cpx *ref = read_image(file, &sz);
    if (sz != n) return -1;
    cpx *dev_in = 0;
    size_t size = n * n * sizeof(cpx);
    cudaMalloc((void**)&dev_in, n * n * sizeof(cpx));

    cudaMemcpy(dev_in, in, n * n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsCombineNCS2D(FFT_FORWARD, dev_in, n);
    cudaMemcpy(in, dev_in, n * n * sizeof(cpx), cudaMemcpyDeviceToHost);

    cudaFree(dev_in);
    write_image(outfile, in, n);
    return 1;
}

/*  Implementation is subject to sync over blocks and as such there are no reliable implementation to do over any number of blocks (yet).
Basic idea is that the limitation is bound to number of SMs. In practice 1, 2 and 4 are guaranteed to work, 8 seems to work at some
configurations.
*/

#define NO_STREAMING_MULTIPROCESSORS 7

__host__ int checkValidConfig(const int blocks, const int n)
{
    if (blocks > NO_STREAMING_MULTIPROCESSORS) {
        switch (MAX_BLOCK_SIZE)
        {
        case 256:   return blocks <= 32;    // 2^14
        case 512:   return blocks <= 16;    // 2^14
        case 1024:  return blocks <= 4;     // 2^13
        // Default is a configurable limit, essentially blocksize limits the number of treads that can perform the synchronization.
        default:    return n <= MAX_BLOCK_SIZE * MAX_BLOCK_SIZE; 
        }
    }
    return 1;
}

__host__ void tsCombineNCS(fftDirection dir, cpx **dev_in, cpx **dev_out, const int n)
{
    int threads, blocks;
    const int n2 = n / 2;
    set_block_and_threads(&blocks, &threads, n2);

    if (!checkValidConfig(blocks, n))
        return;

    const int nBlock = n / blocks;
    const float w_angle = dir * (M_2_PI / n);
    const float w_bangle = dir * (M_2_PI / nBlock);
    const cpx scale = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
    _kernelNCS KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (*dev_in, *dev_out, w_angle, w_bangle, log2_32(n), log2_32(MAX_BLOCK_SIZE), scale, blocks, n);
    cudaDeviceSynchronize();
}

__host__ void tsCombineNCS2D(fftDirection dir, cpx *dev_in, const int n)
{
    int threads;
    dim3 blocks;
    int n2 = n / 2;
    if (n2 > MAX_BLOCK_SIZE) {
        blocks.x = n;                   // rows
        blocks.y = n2 / MAX_BLOCK_SIZE;  // blocks per row
        blocks.z = 1;
        threads = MAX_BLOCK_SIZE;       // treads per block
    }
    else {
        blocks.x = n;                   // rows
        blocks.y = 1;
        blocks.z = 1;
        threads = n2;                    // treads per row
    }
    const cpx scale = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
    int nBlock = n / blocks.y;
    const float w_angle = dir * (M_2_PI / n);
    const float w_bangle = dir * (M_2_PI / nBlock);

    printf("Run with n:%d rows:%d blocks/row:%d threads:%d\n", n, blocks.x, blocks.y, threads);

    _kernelNCS2D KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (dev_in, log2_32(n), w_angle, w_bangle, scale, n);
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e) printf("%s: %s\n", cudaGetErrorName(e), cudaGetErrorString(e));
}

__device__ static __inline__ void _inner_kernel(cpx *in, cpx *out, const float angle, const int steps, const int tid, const unsigned int lmask, const unsigned int pmask, const int dist, const int blocks)
{
    cpx w;
    int in_low = tid + (tid & lmask);
    int in_high = in_low + dist;
    cpx in_lower = in[in_low];
    cpx in_upper = in[in_high];
    SIN_COS_F(angle * ((tid << steps) & pmask), &w.y, &w.x);
    cpx_add_sub_mul(&(out[in_low]), &(out[in_high]), in_lower, in_upper, w);
    SYNC_BLOCKS(blocks, steps);
}

__device__ static __inline__ int _global_body(cpx *in, cpx *out, int *in_high, cpx **stg2, const int start, const int breakSize, const float angle, const int tid, const int nBlocks, const int n2)
{
        int bit = start;
        int dist = n2;
        int steps = 0;
        init_sync(tid, nBlocks);
        _inner_kernel(in, out, angle, steps, tid, 0xFFFFFFFF << bit, (dist - 1) << steps, dist, nBlocks);
        while (--bit > breakSize) {
            dist = dist >> 1;
            ++steps;
            _inner_kernel(out, out, angle, steps, tid, 0xFFFFFFFF << bit, (dist - 1) << steps, dist, nBlocks);
        }
        *in_high = (n2 / nBlocks);
        *stg2 = out;
        return bit;
}

__device__ static __inline__ void _local_body(cpx *in, cpx *out, const int in_low, const int in_high, const int i, const int ii, const int offset, const int lead, const float angle, const cpx scale, cpx *shared, int bit)
{
    cpx w, in_lower, in_upper;
    /* Move Global to Shared */
    mem_gtos(in_low, in_high, offset, shared, in);

    /* Run FFT algorithm */
    for (int steps = 0; steps < bit; ++steps) {
        SYNC_THREADS;
        in_lower = shared[in_low];
        in_upper = shared[in_high];
        SYNC_THREADS;
        SIN_COS_F(angle * ((in_low & (0xFFFFFFFF << steps))), &w.y, &w.x);
        cpx_add_sub_mul(&(shared[i]), &(shared[ii]), in_lower, in_upper, w);
    }

    /* Move Shared to Global */
    SYNC_THREADS;
    mem_stog_db(in_low, in_high, offset, lead, scale, shared, out);
}

__global__ void _kernelNCS2D(cpx *in, const int depth, const float angle, const float bAngle, cpx scale, const int n)
{
    extern __shared__ cpx shared[];
    int global_offset = gridDim.y * blockDim.x * blockIdx.x * 2;    
    cpx *seq = &(in[global_offset]);
    int bit = depth - 1;
    cpx w, in_lower, in_upper;
    int in_high = n >> 1;
    int breakSize = log2(MAX_BLOCK_SIZE);
    
    if (gridDim.y > 1) {
        int tid = (blockIdx.y * blockDim.x + threadIdx.x);
        int dist = in_high;
        int steps = 0;
        init_sync(tid, gridDim.y);
        _inner_kernel(seq, seq, angle, steps, tid, 0xFFFFFFFF << bit, (dist - 1) << steps, dist, gridDim.y);
        while (--bit > breakSize) {
            dist = dist >> 1;
            ++steps;
            _inner_kernel(seq, seq, angle, steps, tid, 0xFFFFFFFF << bit, (dist - 1) << steps, dist, gridDim.y);
        }
        in_high = ((n / gridDim.y) >> 1);
    }
    
    const int in_low = threadIdx.x;
    in_high += in_low;
    const int offset = blockIdx.y * blockDim.x * 2;
    const int i = (in_low << 1);
    const int ii = i + 1;
    
    mem_gtos(in_low, in_high, offset, shared, seq);

    if (threadIdx.x == 0)
        printf("(%d, %d) -> (%d, %d)\n", in_low, in_high, in_low + offset + global_offset, in_high + offset + global_offset);

    ++bit;
    for (int steps = 0; steps < bit; ++steps) {
        SYNC_THREADS;
        in_lower = shared[in_low];
        in_upper = shared[in_high];
        SYNC_THREADS;
        SIN_COS_F(bAngle * ((in_low & (0xFFFFFFFF << steps))), &w.y, &w.x);
        cpx_add_sub_mul(&(shared[i]), &(shared[ii]), in_lower, in_upper, w);
    }

    SYNC_THREADS;
    mem_stog_db(in_low, in_high, offset + global_offset, 32 - depth, scale, shared, seq);
}

__global__ void _kernelNCS(cpx *in, cpx *out, const float angle, const float bAngle, const int depth, const int breakSize, const cpx scale, const int blocks, const int n)
{
    extern __shared__ cpx shared[];
    int bit = depth - 1;
    cpx w, in_lower, in_upper;
    int in_high = n >> 1;

    if (blocks > 1)
        bit = _global_body(in, out, &in_high, &in, bit, breakSize, angle, (blockIdx.x * blockDim.x + threadIdx.x), blocks, in_high);

    _local_body(in, out, threadIdx.x, in_high + threadIdx.x, threadIdx.x << 1, threadIdx.x + 1, blockIdx.x * blockDim.x * 2, 32 - depth, bAngle, scale, shared, ++bit);
    return;

    // Next step, run all in shared mem, copy of _kernelB(...)
    const int in_low = threadIdx.x;
    in_high += in_low;
    const int offset = blockIdx.x * blockDim.x * 2;
    const int i = (in_low << 1);
    const int ii = i + 1;

    // Move Global to Shared
    mem_gtos(in_low, in_high, offset, shared, in);
    
    // Run FFT algorithm
    ++bit;
    for (int steps = 0; steps < bit; ++steps) {
        SYNC_THREADS;
        in_lower = shared[in_low];
        in_upper = shared[in_high];
        SYNC_THREADS;
        SIN_COS_F(bAngle * ((in_low & (0xFFFFFFFF << steps))), &w.y, &w.x);
        cpx_add_sub_mul(&(shared[i]), &(shared[ii]), in_lower, in_upper, w);
    }

    // Move Shared to Global
    SYNC_THREADS;
    mem_stog_db(in_low, in_high, offset, 32 - depth, scale, shared, out);
    //*/
}