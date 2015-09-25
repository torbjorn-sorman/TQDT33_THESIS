#include "tsCombineGPUSync.cuh"

__global__ void _kernelGPUSync(cpx *in, cpx *out, cFloat angle, cFloat bAngle, cInt depth, cInt breakSize, cpx scale, cInt blocks, cInt n);
__global__ void _kernelGPUSync2D(cpx *in, cFloat angle, cFloat bAngle, cInt depth, cpx scale, cInt n);
__global__ void _kernelTranspose(cpx *in, cpx *out, cInt n);

__host__ int tsCombineGPUSync_Validate(cInt n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);

    if (n <= 32) {
        printf("\n");
        console_print(in, n);
    }
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsCombineGPUSync(FFT_FORWARD, &dev_in, &dev_out, n);
    cudaMemcpy(out, dev_out, n * sizeof(cpx), cudaMemcpyDeviceToHost);

    if (n <= 32) {
        printf("\n");
        console_print(out, n);
    }
    tsCombineGPUSync(FFT_INVERSE, &dev_out, &dev_in, n);
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);

    return fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out) != 1;
}

__host__ double tsCombineGPUSync_Performance(cInt n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        tsCombineGPUSync(FFT_FORWARD, &dev_in, &dev_out, n);
        measures[i] = stopTimer();
    }

    fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    return avg(measures, NUM_PERFORMANCE);
}

__host__ void test2DSetup(cpx **in, cpx **ref, cpx **dev_i, cpx **dev_o, size_t *size, char *image_name, cInt sinus, cInt n)
{
    if (sinus) {
        *in = get_sin_img(n);
        *ref = get_sin_img(n);
    }
    else {
        char input_file[40];
        sprintf_s(input_file, 40, "%s/%u.ppm", image_name, n);
        int sz;
        *in = read_image(input_file, &sz);
        *ref = read_image(input_file, &sz);
    }
    *size = n * n * sizeof(cpx);
    cudaMalloc((void**)dev_i, *size);
    cudaMalloc((void**)dev_o, *size);
    cudaMemcpy(*dev_i, *in, *size, cudaMemcpyHostToDevice);
}

__host__ void test2DShakedown(cpx **in, cpx **ref, cpx **dev_i, cpx **dev_o)
{
    free(*in);
    free(*ref);
    cudaFree(*dev_i);
    cudaFree(*dev_o);
}

__host__ void test2DRun(fftDir dir, cpx *in, cpx *ref, cpx *dev_in, cpx *dev_out, char *image_name, char *t1, char *r1, size_t size, cInt write, cInt norm, cInt n)
{
    tsCombineGPUSync2D(dir, dev_in, dev_out, n);
    if (write) {
        cudaMemcpy(in, dev_in, size, cudaMemcpyDeviceToHost);
        if (norm) {
            normalized_image(in, n);
            cpx *tmp = fftShift(in, n);
            write_image(image_name, t1, tmp, n);
            free(tmp);
        }
        else {
            write_image(image_name, t1, in, n);
        }
    }
}

#define YES 1
#define NO 0

__host__ int tsCombineGPUSync2D_Test(cInt n)
{
    double measures[NUM_PERFORMANCE];
    char *image_name = "splash";
    cpx *in, *ref, *dev_in, *dev_out;
    size_t size;
    test2DSetup(&in, &ref, &dev_in, &dev_out, &size, image_name, NO, n);
    /*
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
    startTimer();
    tsCombineGPUSync2D(FFT_FORWARD, dev_in, dev_out, n);
    measures[i] = stopTimer();
    }
    printf("%d: %.0f\n", n, avg(measures, NUM_PERFORMANCE));
    */
    write_image(image_name, "original", in, n);
    test2DRun(FFT_FORWARD, in, ref, dev_in, dev_out, image_name, "f1", "f2", size, YES, YES, n);
    test2DRun(FFT_INVERSE, in, ref, dev_in, dev_out, image_name, "s1", "s2", size, YES, NO, n);
    test2DShakedown(&in, &ref, &dev_in, &dev_out);
    return 1;
}

/*  Implementation is subject to sync over blocks and as such there are no reliable implementation to do over any number of blocks (yet).
Basic idea is that the limitation is bound to number of SMs. In practice 1, 2 and 4 are guaranteed to work, 8 seems to work at some
configurations.
*/

// Fast in block FFT when (partial) list fits shared memory combined with a general intra-block synchronizing algorithm using global memory.
__host__ void tsCombineGPUSync(fftDir dir, cpx **dev_in, cpx **dev_out, cInt n)
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
    _kernelGPUSync KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (*dev_in, *dev_out, w_angle, w_bangle, log2_32(n), log2_32(MAX_BLOCK_SIZE), scale, blocks, n);
    cudaDeviceSynchronize();
}

// Fast in block 2D FFT when (partial) list fits shared memory combined with a general algorithm using global memory. Partially CPU synced and partially GPU-intra-block syncronized.
__host__ void tsCombineGPUSync2D(fftDir dir, cpx *dev_in, cpx *dev_out, cInt n)
{
    dim3 threads, blocks, threads_trans, blocks_trans;
    set2DBlocksNThreads(&blocks, &threads, &blocks_trans, &threads_trans, n);

    cCpx scale = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
    cInt nBlock = n / blocks.y;
    cFloat w_angle = dir * (M_2_PI / n);
    cFloat w_bangle = dir * (M_2_PI / nBlock);

    printf("Dim FFT   {%d\t%d}\tt{%d\t%d}\n", blocks.x, blocks.y, threads.x, threads.y);
    printf("Dim Trans {%d\t%d}\tt{%d\t%d}\n", blocks_trans.x, blocks_trans.y, threads_trans.x, threads_trans.y);

    _kernelGPUSync2D KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (dev_in, w_angle, w_bangle, log2_32(n), scale, n);
    cudaDeviceSynchronize();
    checkCudaError();

    _kernelTranspose KERNEL_ARGS2(blocks_trans, threads_trans)           (dev_in, dev_out, n);
    cudaDeviceSynchronize();
    checkCudaError();

    _kernelGPUSync2D KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (dev_out, w_angle, w_bangle, log2_32(n), scale, n);
    cudaDeviceSynchronize();
    checkCudaError();

    _kernelTranspose KERNEL_ARGS2(blocks_trans, threads_trans)           (dev_out, dev_in, n);
    cudaDeviceSynchronize();
    checkCudaError();
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

__device__ static __inline__ void algorithm_partial(cpx *in, cpx *out, cInt input_offset, cInt offset, cInt depth, cFloat angle, cCpx scale, cInt bit)
{
    extern __shared__ cpx shared[];
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

// Likely room for optimizations!
// In place swapping is problematic over several blocks and is not the task of this thesis work (solving block sync)

// Transpose data set with dimensions n x n
__global__ void _kernelTranspose(cpx *in, cpx *out, cInt n)
{
    // Banking issues when TILE_DIM % WARP_SIZE == 0, current WARP_SIZE == 32
    __shared__ cpx tile[TILE_DIM][TILE_DIM + 1];

    // Write to shared from Global (in)
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += THREAD_TILE_DIM)
        for (int i = 0; i < TILE_DIM; i += THREAD_TILE_DIM)
            tile[threadIdx.y + j][threadIdx.x + i] = in[(y + j) * n + (x + i)];

    SYNC_THREADS;
    // Write to global
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += THREAD_TILE_DIM)
        for (int i = 0; i < TILE_DIM; i += THREAD_TILE_DIM)
            out[(y + j) * n + (x + i)] = tile[threadIdx.x + i][threadIdx.y + j];
}

__global__ void _kernelGPUSync2D(cpx *in, cFloat angle, cFloat bAngle, cInt depth, cpx scale, cInt n)
{
    cInt global_offset = n * blockIdx.x;
    cpx *row = &(in[global_offset]);
    int bit = depth;
    int in_high_offset = n >> 1;

    if (gridDim.y > 1) {
        bit = algorithm_complete(row, row, bit, log2(MAX_BLOCK_SIZE), angle, gridDim.y, in_high_offset);
        in_high_offset >>= log2((int)gridDim.y);
    }
    algorithm_partial(row, row, in_high_offset, blockIdx.y * blockDim.x * 2, depth, bAngle, scale, bit);
}

__global__ void _kernelGPUSync(cpx *in, cpx *out, cFloat angle, cFloat bAngle, cInt depth, cInt breakSize, cCpx scale, cInt nBlocks, cInt n)
{
    int bit = depth;
    int in_high = n >> 1;
    if (nBlocks > 1) {
        bit = algorithm_complete(in, out, depth - 1, breakSize, angle, nBlocks, in_high);
        in_high >>= log2(nBlocks);
        in = out;
    }
    algorithm_partial(in, out, in_high, blockIdx.x * blockDim.x * 2, depth, bAngle, scale, bit);
    return;
}