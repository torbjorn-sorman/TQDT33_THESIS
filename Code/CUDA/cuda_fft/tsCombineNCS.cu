#include "tsCombineNCS.cuh"

__global__ void _kernelNCS(cpx *in, cpx *out, cFloat angle, cFloat bAngle, cInt depth, cInt breakSize, cpx scale, cInt blocks, cInt n);
__global__ void _kernelNCS2D(cpx *in, cInt depth, cFloat angle, cFloat bAngle, cpx scale, cInt n);
__global__ void _kernelTranspose(cpx *in, cpx *out, cInt n);

__host__ int tsCombineNCS_Validate(cInt n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsCombineNCS(FFT_FORWARD, &dev_in, &dev_out, n);
    tsCombineNCS(FFT_INVERSE, &dev_out, &dev_in, n);
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);

    return fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out) != 1;
}

__host__ double tsCombineNCS_Performance(cInt n)
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

__host__ int tsCombineNCS2D_Test(cInt n)
{
    double measures[NUM_PERFORMANCE];
    char input_file[40];
    char refere_file[40];
    char freq_domain[40];
    char spatial_dom[40];
    sprintf_s(input_file, 40, "splash/%u.ppm", n);
    sprintf_s(refere_file, 40, "out/img/splash_%u_ref.ppm", n);
    sprintf_s(freq_domain, 40, "out/img/splash_%u_frequency.ppm", n);
    sprintf_s(spatial_dom, 40, "out/img/splash_%u_spatial.ppm", n);
    int sz;
    cpx *in = read_image(input_file, &sz);
    if (sz != n) return -1;
    cpx *ref = read_image(input_file, &sz);
    if (sz != n) return -1;
    size_t size = n * n * sizeof(cpx);
    cpx *dev_in = 0;
    cpx *dev_out = 0;
    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_out, size);

    //write_image(refere_file, in, n);
    cudaMemcpy(dev_in, in, size, cudaMemcpyHostToDevice);
    

    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        tsCombineNCS2D(FFT_FORWARD, dev_in, dev_out, n);
        measures[i] = stopTimer();
    }
    printf("%d: %.0f\n", n, avg(measures, NUM_PERFORMANCE));

    
    //tsCombineNCS2D(FFT_FORWARD, dev_in, dev_out, n);
    //cudaMemcpy(in, dev_out, size, cudaMemcpyDeviceToHost);
    //write_image(freq_domain, in, n);

    //tsCombineNCS2D(FFT_INVERSE, dev_out, dev_in, n);
    //cudaMemcpy(in, dev_in, size, cudaMemcpyDeviceToHost);
    //write_image(spatial_dom, in, n);

    cudaFree(dev_in);
    cudaFree(dev_out);
    free(in);
    free(ref);
    return 1;
}

/*  Implementation is subject to sync over blocks and as such there are no reliable implementation to do over any number of blocks (yet).
Basic idea is that the limitation is bound to number of SMs. In practice 1, 2 and 4 are guaranteed to work, 8 seems to work at some
configurations.
*/

#define NO_STREAMING_MULTIPROCESSORS 7

__host__ int checkValidConfig(cInt blocks, cInt n)
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

__host__ void tsCombineNCS(fftDirection dir, cpx **dev_in, cpx **dev_out, cInt n)
{
    int threads, blocks;
    cInt n2 = n / 2;
    set_block_and_threads(&blocks, &threads, n2);

    if (!checkValidConfig(blocks, n))
        return;

    cInt nBlock = n / blocks;
    cFloat w_angle = dir * (M_2_PI / n);
    cFloat w_bangle = dir * (M_2_PI / nBlock);
    cCpx scale = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
    _kernelNCS KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (*dev_in, *dev_out, w_angle, w_bangle, log2_32(n), log2_32(MAX_BLOCK_SIZE), scale, blocks, n);
    cudaDeviceSynchronize();
}

__host__ void tsCombineNCS2D(fftDirection dir, cpx *dev_in, cpx *dev_out, cInt n)
{
    dim3 threads, blocks, threads_trans, blocks_trans;
    set2DBlocksNThreads(&blocks, &threads, &blocks_trans, &threads_trans, n);
    //printf("Dim FFT   {%d\t%d}\tt{%d\t%d}\n", blocks.x, blocks.x, threads.x, threads.y);
    //printf("Dim Trans {%d\t%d}\tt{%d\t%d}\n", blocks_trans.x, blocks_trans.y, threads_trans.x, threads_trans.y);
    /*
    cCpx scale = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
    int nBlock = n / blocks.y;
    cFloat w_angle = dir * (M_2_PI / n);
    cFloat w_bangle = dir * (M_2_PI / nBlock);

    printf("Dim FFT Block: {%d, %d} FFT Threads {%d %d} Trans Block: {%d, %d} Trans Threads {%d %d}\n", blocks.x, blocks.x, threads.x, threads.x, blocks_trans.x, blocks_trans.x, threads_trans.x, threads_trans.x);
    */
    /*
    _kernelNCS2D KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (dev_in, log2_32(n), w_angle, w_bangle, scale, n);
    cudaDeviceSynchronize();
    */
    _kernelTranspose KERNEL_ARGS2(blocks_trans, threads_trans) (dev_in, dev_out, n);
    cudaDeviceSynchronize();
    /*
    _kernelNCS2D KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (dev_in, log2_32(n), w_angle, w_bangle, scale, n);
    cudaDeviceSynchronize();
    _kernelTranspose KERNEL_ARGS2(blocks_trans, threads_trans) (dev_in, dev_out, block_rows, n);
    cudaDeviceSynchronize();
    */
    cudaError_t e = cudaGetLastError();
    if (e) printf("%s: %s\n", cudaGetErrorName(e), cudaGetErrorString(e));
}

__device__ static __inline__ void inner_kernel(cpx *in, cpx *out, cFloat angle, cInt steps, cInt tid, cUInt lmask, cUInt pmask, cInt dist, cInt blocks)
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

__device__ static __inline__ int algorithm_complete(cpx *in, cpx *out, cInt bit_start, cInt breakSize, cFloat angle, cInt nBlocks, cInt n2)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    int dist = n2;
    int steps = 0;
    init_sync(tid, nBlocks);
    inner_kernel(in, out, angle, steps, tid, 0xFFFFFFFF << bit_start, (dist - 1) << steps, dist, nBlocks);
    for (int bit = bit_start - 1; bit > breakSize; --bit) {
        dist = dist >> 1;
        ++steps;
        inner_kernel(out, out, angle, steps, tid, 0xFFFFFFFF << bit, (dist - 1) << steps, dist, nBlocks);
    }
    return breakSize + 1;
}

__device__ static __inline__ void algorithm_partial(cpx *in, cpx *out, cInt input_offset, cInt offset, cInt depth, cFloat angle, cCpx scale, cpx *shared, cInt bit)
{
    cpx w, in_lower, in_upper;
    cInt in_low = threadIdx.x;
    cInt in_high = in_low + input_offset;
    cInt i = (in_low << 1);
    cInt ii = i + 1;
    mem_gtos(in_low, in_high, offset, shared, in);
    for (int steps = 0; steps < bit; ++steps) {
        SYNC_THREADS;
        in_lower = shared[in_low];
        in_upper = shared[in_high];
        SYNC_THREADS;
        SIN_COS_F(angle * ((in_low & (0xFFFFFFFF << steps))), &w.y, &w.x);
        cpx_add_sub_mul(&(shared[i]), &(shared[ii]), in_lower, in_upper, w);
    }
    SYNC_THREADS;
    mem_stog_db(in_low, in_high, offset, 32 - depth, scale, shared, out);
}

__global__ void _kernelTranspose(cpx *in, cpx *out, cInt n)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += THREAD_TILE_DIM)
        for (int i = 0; i < TILE_DIM; i += THREAD_TILE_DIM)
            tile[threadIdx.y + j][threadIdx.x + i] = in[(y + j) * n + (x + i)].x;    

    SYNC_THREADS;
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += THREAD_TILE_DIM)
        for (int i = 0; i < TILE_DIM; i += THREAD_TILE_DIM)
            out[(y + j) * n + (x + i)].x = tile[threadIdx.x + i][threadIdx.y + j];
}

__global__ void _kernelNCS2D(cpx *in, cInt depth, cFloat angle, cFloat bAngle, cpx scale, cInt n)
{
    extern __shared__ cpx shared[];

    cInt global_offset = gridDim.y * blockDim.x * blockIdx.x * 2;
    cpx *seq = &(in[global_offset]);
    int bit = depth - 1;
    int in_high_offset = n >> 1;
    if (gridDim.y > 1) {
        bit = algorithm_complete(seq, seq, bit, log2(MAX_BLOCK_SIZE), angle, gridDim.y, in_high_offset);
        in_high_offset >>= log2((int)gridDim.y);
    }
    algorithm_partial(seq, seq, in_high_offset, blockIdx.y * blockDim.x * 2, depth, bAngle, scale, shared, bit);
}

__global__ void _kernelNCS(cpx *in, cpx *out, cFloat angle, cFloat bAngle, cInt depth, cInt breakSize, cCpx scale, cInt nBlocks, cInt n)
{
    extern __shared__ cpx shared[];
    int bit = depth;
    int in_high = n >> 1;
    if (nBlocks > 1) {
        bit = algorithm_complete(in, out, depth - 1, breakSize, angle, nBlocks, in_high);
        in_high >>= log2(nBlocks);
        in = out;
    }
    algorithm_partial(in, out, in_high, blockIdx.x * blockDim.x * 2, depth, bAngle, scale, shared, bit);
    return;
}