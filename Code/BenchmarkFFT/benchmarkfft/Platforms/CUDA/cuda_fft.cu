#include "cuda_fft.cuh"

__global__ void _kernelAll(cpx *in, cpx *out, float angle, unsigned int lmask, int steps, int dist);
__global__ void _kernelAll2DRow(cpx *in, cpx *out, float angle, unsigned int lmask, int steps, int dist);
__global__ void _kernelAll2DCol(cpx *in, cpx *out, float angle, unsigned int lmask, int steps, int dist);
__global__ void _kernelGPUS(cpx *in, cpx *out, float angle, float bAngle, int depth, int lead, int breakSize, cpx scale, int nBlocks, int n2);
__global__ void _kernelGPUS2DRow(cpx *in, cpx *out, float angle, float bAngle, int depth, cpx scale, int n);
__global__ void _kernelGPUS2DCol(cpx *in, cpx *out, float angle, float bAngle, int depth, cpx scale, int n);

__device__ volatile int _sync_array_in[MAX_BLOCK_SIZE];
__device__ volatile int _sync_array_out[MAX_BLOCK_SIZE];

__host__ int tsCombine_Validate(int n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsCombine(FFT_FORWARD, &dev_in, &dev_out, n);
    tsCombine(FFT_INVERSE, &dev_out, &dev_in, n);
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);

    return fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out) != 1;
}

__host__ void testCombine2DRun(fftDir dir, cpx *in, cpx **dev_in, cpx **dev_out, char *type, size_t size, int write, int norm, int n)
{
    tsCombine2D(dir, dev_in, dev_out, n);
    if (write) {
        cudaMemcpy(in, *dev_out, size, cudaMemcpyDeviceToHost);
        if (norm) {
            normalized_image(in, n);
            cpx *tmp = fftShift(in, n);
            write_image("CUDA", type, tmp, n);
            free(tmp);
        }
        else {
            write_image("CUDA", type, in, n);
        }
    }
}

__host__ int tsCombine2D_Validate(int n)
{
    cpx *in, *ref, *dev_in, *dev_out;
    size_t size;
    fft2DSetup(&in, &ref, &dev_in, &dev_out, &size, n);

    cudaMemcpy(dev_in, in, size, cudaMemcpyHostToDevice);
    testCombine2DRun(FFT_FORWARD, in, &dev_in, &dev_out, "frequency-domain", size, 1, 1, n);
    testCombine2DRun(FFT_INVERSE, in, &dev_out, &dev_in, "spatial-domain", size, 1, 0, n);

    int res = fft2DCompare(in, ref, dev_in, size, n * n);
    fft2DShakedown(&in, &ref, &dev_in, &dev_out);
    return res;
}

__host__ double tsCombine_Performance(int n)
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
    double t = avg(measures, NUM_PERFORMANCE);
    return t;
}

__host__ double tsCombine2D_Performance(int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *dev_in, *dev_out;
    size_t size;
    fft2DSetup(&in, &ref, &dev_in, &dev_out, &size, n);

    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        tsCombine2D(FFT_FORWARD, &dev_in, &dev_out, n);
        measures[i] = stopTimer();
    }

    fft2DShakedown(&in, &ref, &dev_in, &dev_out);
    return avg(measures, NUM_PERFORMANCE);
}

// Seven physical "cores" can run blocks in "parallel" (and most important: sync over blocks).
// Essentially my algorithm handles (depending on scheduling and other factors) # blocks fewer than HW_LIMIT on the GPU, any # above is not trivially solved. cuFFT solves this.
__host__ void tsCombine(fftDir dir, cpx **dev_in, cpx **dev_out, int n)
{
    int threads, blocks;
    int depth = log2_32(n);
    int lead = 32 - depth;
    int n2 = (n / 2);
    int breakSize = log2_32(MAX_BLOCK_SIZE);
    cpx scaleCpx = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
    set_block_and_threads(&blocks, &threads, n2);
    int numBlocks = blocks;
    int nBlock = n / blocks;
    float w_angle = dir * (M_2_PI / n);
    int bSize = n2;

    if (blocks >= HW_LIMIT) {
        // Calculate sequence until parts fit into a block, syncronize on CPU until then.
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
        --depth;
        int steps = 0;
        int dist = n2;
        _kernelAll KERNEL_ARGS2(blocks, threads)(*dev_in, *dev_out, w_angle, 0xFFFFFFFF << depth, steps, dist);
        cudaDeviceSynchronize();
        while (--depth > breakSize) {
            dist >>= 1;
            ++steps;
            _kernelAll KERNEL_ARGS2(blocks, threads)(*dev_out, *dev_out, w_angle, 0xFFFFFFFF << depth, steps, dist);
            cudaDeviceSynchronize();
            
        }
        swap(dev_in, dev_out);
        ++depth;
        bSize = nBlock / 2;
        numBlocks = 1;
    }

    // Calculate complete sequence in one launch and syncronize on GPU
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    _kernelGPUS KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (*dev_in, *dev_out, w_angle, dir * (M_2_PI / nBlock), depth, lead, breakSize, scaleCpx, numBlocks, bSize);
    cudaDeviceSynchronize();
}

#define ROW_COL_KERNEL(rw, kr, kc) ((rw) ? (kr) : (kc))

__host__ static __inline void tsCombine2D_help(fftDir dir, cpx **dev_in, cpx **dev_out, int rowWise, int n)
{
    dim3 blocks;
    int threads;
    int depth = log2_32(n);
    const int breakSize = log2_32(MAX_BLOCK_SIZE);
    cpx scaleCpx = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
    set_block_and_threads2D(&blocks, &threads, n);

    const int nBlock = n / blocks.y;
    const float w_angle = dir * (M_2_PI / n);
    const float w_bangle = dir * (M_2_PI / nBlock);
    int bSize = n;
    if (blocks.y > 1) {

        // Calculate sequence until parts fit into a block, syncronize on CPU until then.
        --depth;
        int steps = 0;
        int dist = (n / 2);
        ROW_COL_KERNEL(rowWise, _kernelAll2DRow, _kernelAll2DCol) KERNEL_ARGS2(blocks, threads)(*dev_in, *dev_out, w_angle, 0xFFFFFFFF << depth, steps, dist);
        cudaDeviceSynchronize();
        while (--depth > breakSize) {
            dist >>= 1;
            ++steps;
            ROW_COL_KERNEL(rowWise, _kernelAll2DRow, _kernelAll2DCol) KERNEL_ARGS2(blocks, threads)(*dev_out, *dev_out, w_angle, 0xFFFFFFFF << depth, steps, dist);
            cudaDeviceSynchronize();
        }
        swap(dev_in, dev_out);
        ++depth;
        bSize = nBlock;
    }

    // Calculate complete sequence in one launch and syncronize on GPU
    ROW_COL_KERNEL(rowWise, _kernelGPUS2DRow, _kernelGPUS2DCol) KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (*dev_in, *dev_out, w_angle, w_bangle, depth, scaleCpx, bSize);
    cudaDeviceSynchronize();
}

__host__ void tsCombine2D(fftDir dir, cpx **dev_in, cpx **dev_out, int n)
{
    dim3 blocks, threads;
    set_block_and_threads_transpose(&blocks, &threads, n);

    if (n > 256) {
        tsCombine2D_help(dir, dev_in, dev_out, 1, n);
        kernelTranspose KERNEL_ARGS2(blocks, threads) (*dev_out, *dev_in, n);
        cudaDeviceSynchronize();
        tsCombine2D_help(dir, dev_in, dev_out, 1, n);
        kernelTranspose KERNEL_ARGS2(blocks, threads) (*dev_out, *dev_in, n);
        cudaDeviceSynchronize();
    }
    else {
        tsCombine2D_help(dir, dev_in, dev_out, 1, n);
        tsCombine2D_help(dir, dev_out, dev_in, 0, n);
    }
    swap(dev_in, dev_out);
}

__device__ static __inline__ void inner_k(cpx *in, cpx *out, float angle, int steps, unsigned int lmask, int dist)
{
    cpx w;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int in_low = tid + (tid & lmask);
    int in_high = in_low + dist;
    SIN_COS_F(angle * ((tid << steps) & ((dist - 1) << steps)), &w.y, &w.x);
    cpx_add_sub_mul(&in[in_low], &in[in_high], &out[in_low], &out[in_high], &w);
}

__device__ static __inline__ int algorithm_c(cpx *in, cpx *out, int bit_start, int breakSize, float angle, int nBlocks, int n2)
{
    int dist = n2;
    int steps = 0;
    init_sync(_sync_array_in, _sync_array_out, (blockIdx.x * blockDim.x + threadIdx.x), nBlocks);
    inner_k(in, out, angle, steps, 0xFFFFFFFF << bit_start, dist);
    __gpu_sync(_sync_array_in, _sync_array_out, nBlocks + steps);
    for (int bit = bit_start - 1; bit > breakSize; --bit) {
        dist >>= 1;
        ++steps;
        inner_k(out, out, angle, steps, 0xFFFFFFFF << bit, dist);
        __gpu_sync(_sync_array_in, _sync_array_out, nBlocks + steps);
    }
    return breakSize + 1;
}

__device__ static __inline__ void algorithm_p(cpx *shared, int in_high, float angle, int bit)
{
    //float x, y;
    cpx w, in_lower, in_upper;
    cpx *out_i = &(shared[threadIdx.x << 1]);
    cpx *out_ii = out_i + 1;
    cpx *in_l = &shared[threadIdx.x];
    cpx *in_u = &shared[in_high];
    for (int steps = 0; steps < bit; ++steps) {
        in_lower = *in_l;
        in_upper = *in_u;
        SYNC_THREADS;
        SIN_COS_F(angle * ((threadIdx.x & (0xFFFFFFFF << steps))), &w.y, &w.x);
        cpx_add_sub_mul(&in_lower, &in_upper, out_i, out_ii, &w);
        /*
        x = in_lower.x - in_upper.x;
        y = in_lower.y - in_upper.y;
        out_i->x = in_lower.x + in_upper.x;
        out_i->y = in_lower.y + in_upper.y;
        out_ii->x = (w.x * x) - (w.y * y);
        out_ii->y = (w.y * x) + (w.x * y);
        */
        SYNC_THREADS;
    }
}

// Take no usage of shared mem yet...
__global__ void _kernelAll(cpx *in, cpx *out, float angle, unsigned int lmask, int steps, int dist)
{
    inner_k(in, out, angle, steps, lmask, dist);
}

__device__ static __inline__ void inner_k2D(cpx *in, cpx *out, int x, int y, float angle, int steps, int dist)
{
    cpx w;
    int in_low = x + y * gridDim.x;
    int in_high = in_low + dist;
    SIN_COS_F(angle * (((blockIdx.y * blockDim.x + threadIdx.x) << steps) & ((dist - 1) << steps)), &w.y, &w.x);
    cpx_add_sub_mul(in, in_low, in_high, &(out[in_low]), &(out[in_high]), w);
}

__global__ void _kernelAll2DRow(cpx *in, cpx *out, float angle, unsigned int lmask, int steps, int dist)
{
    cpx w;
    int col_id = blockIdx.y * blockDim.x + threadIdx.x;
    int in_low = (col_id + (col_id & lmask)) + blockIdx.x * gridDim.x;
    int in_high = in_low + dist;
    SIN_COS_F(angle * ((col_id << steps) & ((dist - 1) << steps)), &w.y, &w.x);
    cpx_add_sub_mul(in, in_low, in_high, &(out[in_low]), &(out[in_high]), w);
}

__global__ void _kernelAll2DCol(cpx *in, cpx *out, float angle, unsigned int lmask, int steps, int dist)
{
    int row_id = blockIdx.y * blockDim.x + threadIdx.x;
    inner_k2D(in, out, blockIdx.x, (row_id + (row_id & lmask)), angle, steps, dist);
}

// Full blown block syncronized algorithm! In theory this should scale up but is limited by hardware (#cores)
__global__ void _kernelGPUS(cpx *in, cpx *out, float angle, float bAngle, int depth, int lead, int breakSize, cpx scale, int nBlocks, int n2)
{
    extern __shared__ cpx shared[];
    int bit = depth;
    int in_high = n2;
    if (nBlocks > 1) {
        bit = algorithm_c(in, out, depth - 1, breakSize, angle, nBlocks, in_high);
        in_high >>= log2(nBlocks);
        in = out;
    }
    int offset = blockIdx.x * blockDim.x * 2;
    in_high += threadIdx.x;
    mem_gtos(threadIdx.x, in_high, offset, shared, in);
    SYNC_THREADS;
    algorithm_p(shared, in_high, bAngle, bit);
    mem_stog_db(threadIdx.x, in_high, offset, lead, scale, shared, out);
}

__global__ void _kernelGPUS2DRow(cpx *in, cpx *out, float angle, float bAngle, int depth, cpx scale, int nBlock)
{
    extern __shared__ cpx shared[];
    int rowStart = gridDim.x * blockIdx.x;
    int in_high = (nBlock >> 1) + threadIdx.x;
    int rowOffset = blockIdx.y * blockDim.x * 2;
    mem_gtos(threadIdx.x, in_high, rowOffset, shared, &(in[rowStart]));
    SYNC_THREADS;
    algorithm_p(shared, in_high, bAngle, depth);
    mem_stog_db(threadIdx.x, in_high, rowOffset, (32 - log2((int)gridDim.x)), scale, shared, &(out[rowStart]));
}

__global__ void _kernelGPUS2DCol(cpx *in, cpx *out, float angle, float bAngle, int depth, cpx scale, int n)
{
    extern __shared__ cpx shared[];
    int in_high = n >> 1;
    int colOffset = blockIdx.y * blockDim.x * 2;
    in_high += threadIdx.x;
    mem_gtos_col(threadIdx.x, in_high, (threadIdx.x + colOffset) * n + blockIdx.x, (n >> 1) * n, shared, in);
    SYNC_THREADS;
    algorithm_p(shared, in_high, bAngle, depth);
    mem_stog_db_col(threadIdx.x, in_high, colOffset, 32 - log2((int)gridDim.x), scale, shared, out, n);
}

// ---------------------------------------------
//
// Surface / Texture object
//
// Experimental so see if it scales up better then using global memory
//
// ---------------------------------------------

__device__ __inline__ void surfaceAddSubMul(cpx inL, cpx inU, cuSurf out, int x_add, int y_add, int x_subm, int y_subm, cpx W)
{
    float x = inL.x - inU.x;
    float y = inL.y - inU.y;
    cpx outL, outU;
    outL.x = inL.x + inU.x;
    outL.y = inL.y + inU.y;
    outU.x = (W.x * x) - (W.y * y);
    outU.y = (W.y * x) + (W.x * y);
    SURF2D_WRITE(outL, out, x_add, y_add);
    SURF2D_WRITE(outU, out, x_subm, y_subm);
}

__device__ static __inline__ void inner_k2DRowSurf(cuSurf in, cuSurf out, int x, int y, float angle, int steps, int dist)
{
    cpx w, in_lower, in_upper;
    SURF2D_READ(&in_lower, in, x, y);
    SURF2D_READ(&in_upper, in, x + dist, y);
    SIN_COS_F(angle * (((blockIdx.y * blockDim.x + threadIdx.x) << steps) & ((dist - 1) << steps)), &w.y, &w.x);
    surfaceAddSubMul(in_lower, in_upper, out, x, y, x + dist, y, w);
}
__device__ static __inline__ void inner_k2DColSurf(cuSurf in, cuSurf out, int x, int y, float angle, int steps, int dist)
{
    cpx w, in_lower, in_upper;
    SURF2D_READ(&in_lower, in, x, y);
    SURF2D_READ(&in_upper, in, x, y + dist);
    SIN_COS_F(angle * (((blockIdx.y * blockDim.x + threadIdx.x) << steps) & ((dist - 1) << steps)), &w.y, &w.x);
    surfaceAddSubMul(in_lower, in_upper, out, x, y, x, y + dist, w);
}

__global__ void _kernelAll2DRowSurf(cuSurf in, cuSurf out, float angle, unsigned int lmask, int steps, int dist)
{
    int col_id = blockIdx.y * blockDim.x + threadIdx.x;
    inner_k2DRowSurf(in, out, (col_id + (col_id & lmask)), blockIdx.x, angle, steps, dist);
}

__global__ void _kernelAll2DColSurf(cuSurf in, cuSurf out, float angle, unsigned int lmask, int steps, int dist)
{
    int row_id = blockIdx.y * blockDim.x + threadIdx.x;
    inner_k2DColSurf(in, out, blockIdx.x, (row_id + (row_id & lmask)), angle, steps, dist);
}

__global__ void _kernelGPUS2DRowSurf(cuSurf in, cuSurf out, float angle, float bAngle, int depth, cpx scale, int nBlock)
{
    extern __shared__ cpx shared[];
    int in_high = (nBlock >> 1) + threadIdx.x;
    int rowOffset = blockIdx.y * blockDim.x * 2;
    mem_gtos_row(threadIdx.x, in_high, rowOffset, shared, in);
    SYNC_THREADS;
    algorithm_p(shared, in_high, bAngle, depth);
    mem_stog_dbt_row(threadIdx.x, in_high, rowOffset, (32 - log2((int)gridDim.x)), scale, shared, out);
}

__global__ void _kernelGPUS2DColSurf(cuSurf in, cuSurf out, float angle, float bAngle, int depth, cpx scale, int n)
{
    extern __shared__ cpx shared[];
    int in_high = n >> 1;
    int colOffset = blockIdx.y * blockDim.x * 2;
    in_high += threadIdx.x;
    mem_gtos_col((int)threadIdx.x, in_high, colOffset, shared, in);
    SYNC_THREADS;
    algorithm_p(shared, in_high, bAngle, depth);
    mem_stog_db_col(threadIdx.x, in_high, colOffset, 32 - log2((int)gridDim.x), scale, shared, out);
}

__host__ void tsCombine2DSurf_help(fftDir dir, cuSurf *surfaceIn, cuSurf *surfaceOut, int rowWise, int n)
{
    dim3 blocks;
    int threads;
    int depth = log2_32(n);
    const int breakSize = log2_32(MAX_BLOCK_SIZE);
    cpx scaleCpx = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
    set_block_and_threads2D(&blocks, &threads, n);

    const int nBlock = n / blocks.y;
    const float w_angle = dir * (M_2_PI / n);
    const float w_bangle = dir * (M_2_PI / nBlock);
    int bSize = n;
    if (blocks.y > 1) {

        // Calculate sequence until parts fit into a block, syncronize on CPU until then.
        --depth;
        int steps = 0;
        int dist = (n / 2);
        ROW_COL_KERNEL(rowWise, _kernelAll2DRowSurf, _kernelAll2DColSurf) KERNEL_ARGS2(blocks, threads)(*surfaceIn, *surfaceOut, w_angle, 0xFFFFFFFF << depth, steps, dist);
        cudaDeviceSynchronize();
        while (--depth > breakSize) {
            dist >>= 1;
            ++steps;
            ROW_COL_KERNEL(rowWise, _kernelAll2DRowSurf, _kernelAll2DColSurf) KERNEL_ARGS2(blocks, threads)(*surfaceOut, *surfaceOut, w_angle, 0xFFFFFFFF << depth, steps, dist);
            cudaDeviceSynchronize();
        }
        devSwap(surfaceIn, surfaceOut);
        ++depth;
        bSize = nBlock;
    }
    // Calculate complete sequence in one launch and syncronize on GPU
    ROW_COL_KERNEL(rowWise, _kernelGPUS2DRowSurf, _kernelGPUS2DColSurf) KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (*surfaceIn, *surfaceOut, w_angle, w_bangle, depth, scaleCpx, bSize);
    cudaDeviceSynchronize();
}

__host__ void tsCombine2DSurf(fftDir dir, cuSurf *surfaceIn, cuSurf *surfaceOut, int n)
{
    tsCombine2DSurf_help(dir, surfaceIn, surfaceOut, 1, n);
    tsCombine2DSurf_help(dir, surfaceOut, surfaceIn, 1, n);
    devSwap(surfaceIn, surfaceOut);
}

__host__ void _testTex2DShakedown(cpx **in, cpx **ref, cuSurf *sObjIn, cuSurf *sObjOut, cudaArray **cuAIn, cudaArray **cuAOut)
{
    free(*in);
    free(*ref);
    cudaDestroySurfaceObject(*sObjIn);
    cudaFreeArray(*cuAIn);
    if (sObjOut != NULL) {
        cudaDestroySurfaceObject(*sObjOut);
        cudaFreeArray(*cuAOut);
    }
}

__host__ void _testTex2DRun(fftDir dir, cpx *in, cudaArray *dev, cuSurf *surfIn, cuSurf *surfOut, char *type, size_t size, int write, int norm, int n)
{
    tsCombine2DSurf(dir, surfIn, surfOut, n);
    if (write) {
        cudaMemcpyFromArray(in, dev, 0, 0, size, cudaMemcpyDeviceToHost);
        if (norm) {
            normalized_image(in, n);
            cpx *tmp = fftShift(in, n);
            write_image("CUDA TEX", type, tmp, n);
            free(tmp);
        }
        else {
            write_image("CUDA TEX", type, in, n);
        }
    }
}

__host__ int _testTex2DCompare(cpx *in, cpx *ref, cudaArray *dev, size_t size, int len)
{
    cudaMemcpyFromArray(in, dev, 0, 0, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < len; ++i) {
        if (cuCabsf(cuCsubf(in[i], ref[i])) > 0.0001) {
            return 0;
        }
    }
    return 1;
}

#define YES 1
#define NO 0

__host__ int tsCombine2DSurf_Validate(int n)
{
    int res;
    cpx *in, *ref;
    size_t size;
    cudaArray *inArr, *outArr;
    cuSurf inSurf, outSurf;
    fft2DSurfSetup(&in, &ref, &size, NO, n, &inArr, &outArr, &inSurf, &outSurf);
    cudaMemcpyToArray(inArr, 0, 0, in, size, cudaMemcpyHostToDevice);
    _testTex2DRun(FFT_FORWARD, in, inArr, &inSurf, &outSurf, "surf-frequency-domain", size, YES, YES, n);
    _testTex2DRun(FFT_INVERSE, in, inArr, &outSurf, &inSurf, "surf-spatial-domain", size, YES, NO, n);
    res = _testTex2DCompare(in, ref, inArr, size, n * n);
    _testTex2DShakedown(&in, &ref, &inSurf, &outSurf, &inArr, &outArr);
    return res;
}

__host__ double tsCombine2DSurf_Performance(int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref;
    size_t size;
    cudaArray *inArr, *outArr;
    cuSurf inSurf, outSurf;
    fft2DSurfSetup(&in, &ref, &size, NO, n, &inArr, &outArr, &inSurf, &outSurf);

    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        tsCombine2DSurf(FFT_FORWARD, &inSurf, &outSurf, n);
        cudaCheckError();
        measures[i] = stopTimer();
    }
    _testTex2DShakedown(&in, &ref, &inSurf, &outSurf, &inArr, &outArr);
    return avg(measures, NUM_PERFORMANCE);
}