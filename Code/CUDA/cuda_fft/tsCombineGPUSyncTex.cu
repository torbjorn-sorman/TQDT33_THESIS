#include "tsCombineGPUSyncTex.cuh"

__global__ void _kernelGPUSyncTex(cpx *in, cpx *out, cFloat angle, cFloat bAngle, cInt depth, cInt breakSize, cCpx scale, cInt nBlocks, cInt n2);
__global__ void _kernelGPUSyncTex2D(cudaSurfaceObject_t surface, cpx *in, cpx *out, cFloat angle, cFloat bAngle, cInt depth, cInt rowWise, cpx scale, cInt n);

__host__ int tsCombineGPUSyncTex_Validate(cInt n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsCombineGPUSyncTex(FFT_FORWARD, &dev_in, &dev_out, n);
    tsCombineGPUSyncTex(FFT_INVERSE, &dev_out, &dev_in, n);
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);

    return fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out) != 1;
}

__host__ double tsCombineGPUSyncTex_Performance(cInt n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        tsCombineGPUSyncTex(FFT_FORWARD, &dev_in, &dev_out, n);
        measures[i] = stopTimer();
    }

    fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    return avg(measures, NUM_PERFORMANCE);
}

__host__ void testTex2DSetup(cpx **in, cpx **ref, cpx **dev_i, cpx **dev_o, size_t *size, char *image_name, cInt sinus, cInt n)
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

__host__ void testTex2DShakedown(cpx **in, cpx **ref, cpx **dev_i, cpx **dev_o)
{
    free(*in);
    free(*ref);
    cudaFree(*dev_i);
    cudaFree(*dev_o);
}

__host__ void testTex2DRun(fftDir dir, cudaSurfaceObject_t surface, cpx *in, cpx *dev_in, cpx *dev_out, char *image_name, char *type, size_t size, cInt write, cInt norm, cInt n)
{
    tsCombineGPUSyncTex2D(dir, surface, dev_in, dev_out, n);
    if (write) {
        cudaMemcpy(in, dev_out, size, cudaMemcpyDeviceToHost);
        if (norm) {
            normalized_image(in, n);
            cpx *tmp = fftShift(in, n);
            write_image(image_name, type, tmp, n);
            free(tmp);
        }
        else {
            write_image(image_name, type, in, n);
        }
    }
}

__host__ int testTex2DCompare(cpx *in, cCpx *ref, cpx *dev, size_t size, cInt len)
{
    cudaMemcpy(in, dev, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < len; ++i) {
        if (cuCabsf(cuCsubf(in[i], ref[i])) > 0.0001) {
            return 0;
        }
    }
    return 1;
}

#define YES 1
#define NO 0

__host__ int tsCombineGPUSyncTex2D_Test(cInt n)
{
    double measures[NUM_PERFORMANCE];
    char *image_name = "shore";
    cpx *in, *ref, *dev_in, *dev_out;
    size_t size;
    testTex2DSetup(&in, &ref, &dev_in, &dev_out, &size, image_name, NO, n);
    
    // Allocate CUDA arrays in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();
    cudaArray* cuInputArray;
    cudaMallocArray(&cuInputArray, &channelDesc, n, n, cudaArraySurfaceLoadStore);
    cudaMemcpyToArray(cuInputArray, 0, 0, in, size, cudaMemcpyHostToDevice);

    // Specify surface
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    // Create the surface objects
    resDesc.res.array.array = cuInputArray;
    cudaSurfaceObject_t inputSurfObj = 0;
    cudaCreateSurfaceObject(&inputSurfObj, &resDesc);

    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        tsCombineGPUSyncTex2D(FFT_FORWARD, inputSurfObj, dev_in, dev_out, n);
        measures[i] = stopTimer();
    }
    printf("%.0f\t", avg(measures, NUM_PERFORMANCE));

    //printf("-1\t");
    cudaMemcpy(dev_in, in, size, cudaMemcpyHostToDevice);
    testTex2DRun(FFT_FORWARD, inputSurfObj, in, dev_in, dev_out, image_name, "frequency-domain", size, YES, YES, n);
    testTex2DRun(FFT_INVERSE, inputSurfObj, in, dev_in, dev_out, image_name, "spatial-domain", size, YES, NO, n);
    int res = testTex2DCompare(in, ref, dev_in, size, n * n);
    testTex2DShakedown(&in, &ref, &dev_in, &dev_out);

    cudaDestroySurfaceObject(inputSurfObj);
    cudaFreeArray(cuInputArray);

    return res;
}

/*  Implementation is subject to sync over blocks and as such there are no reliable implementation to do over any number of blocks (yet).
Basic idea is that the limitation is bound to number of SMs. In practice 1, 2 and 4 are guaranteed to work, 8 seems to work at some
configurations.
*/

// Fast in block FFT when (partial) list fits shared memory combined with a general intra-block synchronizing algorithm using global memory.
__host__ void tsCombineGPUSyncTex(fftDir dir, cpx **dev_in, cpx **dev_out, cInt n)
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
    _kernelGPUSyncTex KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (*dev_in, *dev_out, w_angle, w_bangle, log2_32(n), log2_32(MAX_BLOCK_SIZE), scale, blocks, n2);
    cudaDeviceSynchronize();
}

// Fast in block 2D FFT when (partial) list fits shared memory combined with a general algorithm using global memory. Partially CPU synced and partially GPU-intra-block syncronized.
__host__ void tsCombineGPUSyncTex2D(fftDir dir, cudaSurfaceObject_t sur, cpx *dev_in, cpx *dev_out, cInt n)
{
    dim3 threads, blocks, threads_trans, blocks_trans;
    set2DBlocksNThreads(&blocks, &threads, &blocks_trans, &threads_trans, n);

    cCpx scale = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
    cInt nBlock = n / blocks.y;
    cFloat w_angle = dir * (M_2_PI / n);
    cFloat w_bangle = dir * (M_2_PI / nBlock);

    _kernelGPUSyncTex2D KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (sur, dev_in, dev_out, w_angle, w_bangle, log2_32(n), 1, scale, n);
    cudaDeviceSynchronize();
    checkCudaError();

    _kernelGPUSyncTex2D KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (sur, dev_out, dev_in, w_angle, w_bangle, log2_32(n), 0, scale, n);
    cudaDeviceSynchronize();
    checkCudaError();
}

// Move to helper!
__device__ static __inline__ void init_sync2d(cInt tid, cInt blocks)
{
    if (tid < blocks) {
        _sync_array_2din[blockIdx.x][tid] = 0;
        _sync_array_2dout[blockIdx.x][tid] = 0;
    }
}

// Move to helper!
__device__ static __inline__ void __gpu_sync2d(cInt goal)
{
    volatile int *in = _sync_array_2din[blockIdx.x];
    volatile int *out = _sync_array_2dout[blockIdx.x];
    int bid = blockIdx.y;
    int tid = threadIdx.x;
    int nBlocks = gridDim.y;
    if (tid == 0) { in[bid] = goal; }
    if (bid == 1) { // Use bid == 1, if only one block this part will not run.
        SYNC_THREADS;
        if (tid < nBlocks) { while (in[tid] != goal){} }
        SYNC_THREADS;
        if (tid < nBlocks) { out[tid] = goal; }
    }
    if (tid == 0) { while (out[bid] != goal) {} }
    SYNC_THREADS;
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
    __gpu_sync(blocks + steps);
}

__device__ static __inline__ void inner_kernel_2d(cpx *in, cpx *out, cFloat angle, cInt steps, cInt tid, cUInt lmask, cUInt pmask, cInt dist, cInt blocks)
{
    cpx w;
    int in_low = tid + (tid & lmask);
    int in_high = in_low + dist;
    cpx in_lower = in[in_low];
    cpx in_upper = in[in_high];
    SIN_COS_F(angle * ((tid << steps) & pmask), &w.y, &w.x);
    cpx_add_sub_mul(&(out[in_low]), &(out[in_high]), in_lower, in_upper, w);
    if (blockIdx.x == 0 && threadIdx.x == 0) printf("Test sync!");
    __gpu_sync2d(blocks + steps);
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

__device__ static __inline__ int algorithm_complete_2d(cpx *in, cpx *out, cInt bit_start, cInt breakSize, cFloat angle, cInt nBlocks, cInt n2)
{
    int tid = (blockIdx.y * blockDim.x + threadIdx.x);
    int dist = n2;
    int steps = 0;
    init_sync2d(tid, nBlocks);
    inner_kernel_2d(in, out, angle, steps, tid, 0xFFFFFFFF << bit_start, (dist - 1) << steps, dist, nBlocks);
    for (int bit = bit_start - 1; bit > breakSize; --bit) {
        dist = dist >> 1;
        ++steps;
        inner_kernel_2d(out, out, angle, steps, tid, 0xFFFFFFFF << bit, (dist - 1) << steps, dist, nBlocks);
    }
    return breakSize + 1;
}

__device__ static __inline__ void algorithm_partial(cpx *shared, cInt in_high, cFloat angle, cInt bit)
{
    cpx w, in_lower, in_upper;
    cInt i = (threadIdx.x << 1);
    cInt ii = i + 1;
    for (int steps = 0; steps < bit; ++steps) {
        in_lower = shared[threadIdx.x];
        in_upper = shared[in_high];
        SYNC_THREADS;
        SIN_COS_F(angle * ((threadIdx.x & (0xFFFFFFFF << steps))), &w.y, &w.x);
        cpx_add_sub_mul(&(shared[i]), &(shared[ii]), in_lower, in_upper, w);
        SYNC_THREADS;
    }
}

#ifdef __CUDACC__
#define TEX_FETCH(t,i) (tex1Dfetch<cpx>((t), (i)))
#define SURF2D_READ(s,x,y) (surf2Dread((s), (x), (y)))
#else
#define SURF2D_READ(s,x,y) {1, 0}
#define TEX_FETCH(t,i) {1, 0}
#endif

__device__ static __inline__ void mem_gtos_col(cInt low, cInt high, cInt global_low, cInt offsetHigh, cpx *shared, cudaSurfaceObject_t sur, cpx *global)
{
    shared[low] = global[global_low];
    shared[high] = global[global_low + offsetHigh];
    shared[low] = SURF2D_READ(sur, blockIdx.x, threadIdx.x + blockIdx.y * blockDim.x);
}

__device__ static __inline__ void mem_stog_db_col(cInt shared_low, cInt shared_high, cInt offset, cUInt lead, cpx scale, cpx *shared, cudaSurfaceObject_t sur, cpx *global, cInt n)
{
    int row_low = BIT_REVERSE(shared_low + offset, lead);
    int row_high = BIT_REVERSE(shared_high + offset, lead);
    global[row_low * n + blockIdx.x] = cuCmulf(shared[shared_low], scale);
    global[row_high * n + blockIdx.x] = cuCmulf(shared[shared_high], scale);
}

__global__ void _kernelGPUSyncTex2D(cudaSurfaceObject_t sur, cpx *in, cpx *out, cFloat angle, cFloat bAngle, cInt depth, cInt rowWise, cpx scale, cInt n)
{
    extern __shared__ cpx shared[];
    cInt global_offset = n * blockIdx.x;
    int bit = depth;
    int in_high = n >> 1;

    if (gridDim.y > 1) {
        bit = algorithm_complete_2d(&(in[global_offset]), &(out[global_offset]), bit, log2(MAX_BLOCK_SIZE), angle, gridDim.y, in_high);
        in_high >>= log2((int)gridDim.y);
    }

    int offset = blockIdx.y * blockDim.x * 2;
    in_high += threadIdx.x;
    if (rowWise)
        mem_gtos(threadIdx.x, in_high, offset, shared, &(in[global_offset]));
    else {
        mem_gtos_col(threadIdx.x, in_high, (threadIdx.x + offset) * n + blockIdx.x, (n >> 1) * n, shared, sur, in);
    }
    SYNC_THREADS;
    algorithm_partial(shared, in_high, bAngle, bit);
    if (rowWise)
        mem_stog_db(threadIdx.x, in_high, offset, 32 - depth, scale, shared, &(out[global_offset]));
    else
        mem_stog_db_col(threadIdx.x, in_high, offset, 32 - depth, scale, shared, sur, out, n);
}

__global__ void _kernelGPUSyncTex(cpx *in, cpx *out, cFloat angle, cFloat bAngle, cInt depth, cInt breakSize, cCpx scale, cInt nBlocks, cInt n2)
{
    extern __shared__ cpx shared[];
    int bit = depth;
    int in_high = n2;
    if (nBlocks > 1) {
        bit = algorithm_complete(in, out, depth - 1, breakSize, angle, nBlocks, in_high);
        in_high >>= log2(nBlocks);
        in = out;
    }
    int offset = blockIdx.x * blockDim.x * 2;
    in_high += threadIdx.x;
    mem_gtos(threadIdx.x, in_high, offset, shared, in);
    algorithm_partial(shared, in_high, bAngle, bit);
    mem_stog_db(threadIdx.x, in_high, offset, 32 - depth, scale, shared, out);
    return;
}