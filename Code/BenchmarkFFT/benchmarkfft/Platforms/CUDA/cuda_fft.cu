#include "cuda_fft.cuh"

__global__ void cuda_kernel_global(cpx *in, cpx *out, float global_angle, unsigned int lmask, int steps, int dist);
__global__ void cuda_kernel_global_row(cpx *in, cpx *out, float global_angle, unsigned int lmask, int steps, int dist);
__global__ void cuda_kernel_global_col(cpx *in, cpx *out, float global_angle, unsigned int lmask, int steps, int dist);
__global__ void cuda_kernel_local(cpx *in, cpx *out, float global_angle, float local_angle, int steps_left, int leading_bits, int steps_gpu, float scalar, int number_of_blocks, int n_half);
__global__ void cuda_kernel_local_row(cpx *in, cpx *out, float global_angle, float local_angle, int steps_left, float scalar, int n);
__global__ void cuda_kernel_local_col(cpx *in, cpx *out, float global_angle, float local_angle, int steps_left, float scalar, int n);

__device__ volatile int sync_array_in[MAX_BLOCK_SIZE];
__device__ volatile int sync_array_out[MAX_BLOCK_SIZE];

__host__ int cuda_validate(int n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out;
    cuda_setup_buffers(n, &dev_in, &dev_out, &in, &ref, &out);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    cuda_fft(FFT_FORWARD, &dev_in, &dev_out, n);
    cuda_fft(FFT_INVERSE, &dev_out, &dev_in, n);
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);
    return cuda_shakedown(n, &dev_in, &dev_out, &in, &ref, &out) != 1;
}

__host__ void testCombine2DRun(fftDir dir, cpx *in, cpx **dev_in, cpx **dev_out, size_t size, bool write, bool norm, int n)
{
    cuda_fft_2d(dir, dev_in, dev_out, n);
    if (write) {
        cudaMemcpy(in, *dev_out, size, cudaMemcpyDeviceToHost);
        if (norm) {
            write_image("CUDA", "frequency - not norm", in, n);
            write_normalized_image("CUDA", "frequency", in, n, true);
        }
        else
            write_image("CUDA", "spatial", in, n);
    }
}

__host__ int cuda_2d_validate(int n)
{
    cpx *host_buffer, *ref, *dev_in, *dev_out;
    size_t size;
    cuda_setup_buffers_2d(&host_buffer, &ref, &dev_in, &dev_out, &size, n);

    cudaMemcpy(dev_in, host_buffer, size, cudaMemcpyHostToDevice);
    cuda_fft_2d(FFT_FORWARD, &dev_in, &dev_out, n);
    cudaMemcpy(host_buffer, dev_out, size, cudaMemcpyDeviceToHost);
    write_normalized_image("CUDA", "frequency", host_buffer, n, true);

    cuda_fft_2d(FFT_INVERSE, &dev_out, &dev_in, n);
    cudaMemcpy(host_buffer, dev_in, size, cudaMemcpyDeviceToHost);
    write_image("CUDA", "spatial", host_buffer, n);

    int res = cuda_compare_result(host_buffer, ref, dev_in, size, n * n);
    cuda_shakedown_2d(&host_buffer, &ref, &dev_in, &dev_out);
    return res;
}

__host__ double cuda_performance(int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *out, *dev_in, *dev_out;
    cuda_setup_buffers(n, &dev_in, &dev_out, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        cuda_fft(FFT_FORWARD, &dev_in, &dev_out, n);
        measures[i] = stopTimer();
    }
    cuda_shakedown(n, &dev_in, &dev_out, &in, &ref, &out);
    double t = average_best(measures, NUM_PERFORMANCE);
    return t;
}

__host__ double cuda_2d_performance(int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *dev_in, *dev_out;
    size_t size;

    cuda_setup_buffers_2d(&in, &ref, &dev_in, &dev_out, &size, n);

    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        cuda_fft_2d(FFT_FORWARD, &dev_in, &dev_out, n);
        measures[i] = stopTimer();
    }

    cuda_shakedown_2d(&in, &ref, &dev_in, &dev_out);
    return average_best(measures, NUM_PERFORMANCE);
}

// Seven physical "cores" can run blocks in "parallel" (and most important: sync over blocks).
// Essentially my algorithm handles (depending on scheduling and other factors) # blocks fewer than HW_LIMIT on the GPU, any # above is not trivially solved. cuFFT solves this.
__host__ void cuda_fft(fftDir dir, cpx **in, cpx **out, int n)
{
    int threads, blocks;


    int n_half = (n >> 1);
    int steps_left = log2_32(n);
    int leading_bits = 32 - steps_left;
    int steps_gpu = log2_32(MAX_BLOCK_SIZE);
    float scalar = (dir == FFT_FORWARD ? 1.f : 1.f / n);
    set_block_and_threads(&blocks, &threads, n_half);
    int number_of_blocks = blocks;
    int n_per_block = n / number_of_blocks;
    float global_angle = dir * (M_2_PI / n);
    float local_angle = dir * (M_2_PI / n_per_block);
    int block_range_half = n_half;
    if (number_of_blocks >= HW_LIMIT) {
        // Calculate sequence until parts fit into a block, syncronize on CPU until then.
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
        --steps_left;
        int steps = 0;
        int dist = n_half;
        cuda_kernel_global KERNEL_ARGS2(number_of_blocks, threads)(*in, *out, global_angle, 0xFFFFFFFF << steps_left, steps, dist);
        cudaDeviceSynchronize();
        // Instead of swapping input/output, run in place. The argGPU kernel needs to swap once.                
        while (--steps_left > steps_gpu) {
            dist >>= 1;
            ++steps;
            cuda_kernel_global KERNEL_ARGS2(number_of_blocks, threads)(*out, *out, global_angle, 0xFFFFFFFF << steps_left, steps, dist);
            cudaDeviceSynchronize();
        }
        swap_buffer(in, out);
        ++steps_left;
        number_of_blocks = 1;
        block_range_half = n_per_block >> 1;
    }
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cuda_kernel_local KERNEL_ARGS3(blocks, threads, sizeof(cpx) * n_per_block) (*in, *out, global_angle, local_angle, steps_left, leading_bits, steps_gpu, scalar, number_of_blocks, block_range_half);
    cudaDeviceSynchronize();
}

#define ROW_COL_KERNEL(rw, kr, kc) ((rw) ? (kr) : (kc))

__host__ static __inline void cuda_fft_2d_helper(fftDir dir, cpx **dev_in, cpx **dev_out, int row_wise, int n)
{
    dim3 blocks;
    int threads;
    int steps_left = log2_32(n);
    const int steps_gpu = log2_32(MAX_BLOCK_SIZE);
    float scalar = (dir == FFT_FORWARD ? 1.f : 1.f / n);
    set_block_and_threads2D(&blocks, &threads, n);
    const int n_per_block = n / blocks.y;
    const float global_angle = dir * (M_2_PI / n);
    const float local_angle = dir * (M_2_PI / n_per_block);
    int block_range_half = n;
    if (blocks.y > 1) {
        // Calculate sequence until parts fit into a block, syncronize on CPU until then.
        --steps_left;
        int steps = 0;
        int dist = n >> 1;
        ROW_COL_KERNEL(row_wise, cuda_kernel_global_row, cuda_kernel_global_col) KERNEL_ARGS2(blocks, threads)(*dev_in, *dev_out, global_angle, 0xFFFFFFFF << steps_left, steps, dist);
        cudaDeviceSynchronize();
        // Instead of swapping input/output, run in place. The argGPU kernel needs to swap once.                
        while (--steps_left > steps_gpu) {
            dist >>= 1;
            ++steps;
            ROW_COL_KERNEL(row_wise, cuda_kernel_global_row, cuda_kernel_global_col) KERNEL_ARGS2(blocks, threads)(*dev_out, *dev_out, global_angle, 0xFFFFFFFF << steps_left, steps, dist);
            cudaDeviceSynchronize();
        }
        swap_buffer(dev_in, dev_out);
        ++steps_left;
        block_range_half = n_per_block;
    }
    // Calculate complete sequence in one launch and syncronize on GPU
    ROW_COL_KERNEL(row_wise, cuda_kernel_local_row, cuda_kernel_local_col) KERNEL_ARGS3(blocks, threads, sizeof(cpx) * n_per_block) (*dev_in, *dev_out, global_angle, local_angle, steps_left, scalar, block_range_half);
    cudaDeviceSynchronize();
}

__host__ void cuda_fft_2d(fftDir dir, cpx **dev_in, cpx **dev_out, int n)
{
    dim3 blocks, threads;
    set_block_and_threads_transpose(&blocks, &threads, n);

    if (n > 256) {
        cuda_fft_2d_helper(dir, dev_in, dev_out, 1, n);
        cuda_transpose_kernel KERNEL_ARGS2(blocks, threads) (*dev_out, *dev_in, n);
        cudaDeviceSynchronize();
        cuda_fft_2d_helper(dir, dev_in, dev_out, 1, n);
        cuda_transpose_kernel KERNEL_ARGS2(blocks, threads) (*dev_out, *dev_in, n);
        cudaDeviceSynchronize();
    }
    else {
        cuda_fft_2d_helper(dir, dev_in, dev_out, 1, n);
        cuda_fft_2d_helper(dir, dev_out, dev_in, 0, n);
    }
    swap_buffer(dev_in, dev_out);
}

__device__ static __inline__ void cuda_inner_kernel(cpx *in, cpx *out, float angle, int steps, unsigned int lmask, int dist)
{
    cpx w;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int in_low = tid + (tid & lmask);
    in += in_low;
    out += in_low;
    SIN_COS_F(angle * ((tid << steps) & ((dist - 1) << steps)), &w.y, &w.x);
    cpx_add_sub_mul(in, in + dist, out, out + dist, &w);
}

__device__ static __inline__ int cuda_algorithm_global_sync(cpx *in, cpx *out, int bit_start, int steps_gpu, float angle, int number_of_blocks, int n_half)
{
    int dist = n_half;
    int steps = 0;
    cuda_block_sync_init(sync_array_in, sync_array_out, (blockIdx.x * blockDim.x + threadIdx.x), number_of_blocks);
    cuda_inner_kernel(in, out, angle, steps, 0xFFFFFFFF << bit_start, dist);
    cuda_block_sync(sync_array_in, sync_array_out, number_of_blocks + steps);
    for (int bit = bit_start - 1; bit > steps_gpu; --bit) {
        dist >>= 1;
        ++steps;
        cuda_inner_kernel(out, out, angle, steps, 0xFFFFFFFF << bit, dist);
        cuda_block_sync(sync_array_in, sync_array_out, number_of_blocks + steps);
    }
    return steps_gpu + 1;
}

__device__ static __inline__ void cuda_algorithm_local(cpx *shared, int in_high, float angle, int bit)
{
    cpx w, in_lower, in_upper;

    cpx *out_i = shared + (threadIdx.x << 1);
    cpx *out_ii = out_i + 1;
    cpx *in_l = shared + threadIdx.x;
    cpx *in_u = shared + in_high;
    for (int steps = 0; steps < bit; ++steps) {
        SIN_COS_F(angle * (threadIdx.x & (0xFFFFFFFF << steps)), &w.y, &w.x);
        in_lower = *in_l;
        in_upper = *in_u;
        SYNC_THREADS;
        cpx_add_sub_mul(&in_lower, &in_upper, out_i, out_ii, &w);
        SYNC_THREADS;
    }
}

// Take no usage of shared mem yet...
__global__ void cuda_kernel_global(cpx *in, cpx *out, float angle, unsigned int lmask, int steps, int dist)
{
    cuda_inner_kernel(in, out, angle, steps, lmask, dist);
}

__global__ void cuda_kernel_global_row(cpx *in, cpx *out, float angle, unsigned int lmask, int steps, int dist)
{
    cpx w;
    int col_id = blockIdx.y * blockDim.x + threadIdx.x;
    int in_low = (col_id + (col_id & lmask)) + blockIdx.x * gridDim.x;
    in += in_low;
    out += in_low;
    SIN_COS_F(angle * ((col_id << steps) & ((dist - 1) << steps)), &w.y, &w.x);
    cpx_add_sub_mul(in, in + dist, out, out + dist, &w);
}

__global__ void cuda_kernel_global_col(cpx *in, cpx *out, float angle, unsigned int lmask, int steps, int dist)
{
    int row_id = blockIdx.y * blockDim.x + threadIdx.x;
    cpx w;
    int in_low = blockIdx.x + (row_id + (row_id & lmask)) * gridDim.x;
    in += in_low;
    out += in_low;
    SIN_COS_F(angle * (((blockIdx.y * blockDim.x + threadIdx.x) << steps) & ((dist - 1) << steps)), &w.y, &w.x);
    cpx_add_sub_mul(in, in + dist, out, out + dist, &w);
}

// Full blown block syncronized algorithm! In theory this should scalar up but is limited by hardware (#cores)
__global__ void cuda_kernel_local(cpx *in, cpx *out, float angle, float local_angle, int steps_left, int leading_bits, int steps_gpu, float scalar, int number_of_blocks, int n_half)
{
    extern __shared__ cpx shared[];
    int bit = steps_left;
    int in_high = n_half;
    if (number_of_blocks > 1) {
        bit = cuda_algorithm_global_sync(in, out, steps_left - 1, steps_gpu, angle, number_of_blocks, in_high);
        in_high >>= log2(number_of_blocks);
        in = out;
    }

    int offset = blockIdx.x * blockDim.x * 2;
    in_high += threadIdx.x;
    in += offset;
    shared[threadIdx.x] = in[threadIdx.x];
    shared[in_high] = in[in_high];
    SYNC_THREADS;
    cuda_algorithm_local(shared, in_high, local_angle, bit);


    out[BIT_REVERSE(threadIdx.x + offset, leading_bits)] = { shared[threadIdx.x].x * scalar, shared[threadIdx.x].y *scalar };
    out[BIT_REVERSE(in_high + offset, leading_bits)] = { shared[in_high].x * scalar, shared[in_high].y *scalar };
}

__global__ void cuda_kernel_local_row(cpx *in, cpx *out, float angle, float local_angle, int steps_left, float scalar, int n_per_block)
{
    extern __shared__ cpx shared[];
    int leading_bits = (32 - log2((int)gridDim.x));

    int in_high = (n_per_block >> 1) + threadIdx.x;
    int rowStart = gridDim.x * blockIdx.x;
    int rowOffset = blockIdx.y * blockDim.x * 2;
    in += rowStart + rowOffset;
    out += rowStart;
    shared[threadIdx.x] = in[threadIdx.x];
    shared[in_high] = in[in_high];
    cuda_algorithm_local(shared, in_high, local_angle, steps_left);


    out[BIT_REVERSE(threadIdx.x + rowOffset, leading_bits)] = { shared[threadIdx.x].x * scalar, shared[threadIdx.x].y *scalar };
    out[BIT_REVERSE(in_high + rowOffset, leading_bits)] = { shared[in_high].x * scalar, shared[in_high].y *scalar };
}

__global__ void cuda_kernel_local_col(cpx *in, cpx *out, float angle, float local_angle, int steps_left, float scalar, int n)
{
    extern __shared__ cpx shared[];
    int in_high = n >> 1;
    int colOffset = blockIdx.y * blockDim.x * 2;
    in_high += threadIdx.x;

    in += (threadIdx.x + colOffset) * n + blockIdx.x;
    out += blockIdx.x;

    shared[threadIdx.x] = *in;
    shared[in_high] = *(in + (n >> 1) * n);
    SYNC_THREADS;
    cuda_algorithm_local(shared, in_high, local_angle, steps_left);

    int leading_bits = 32 - log2((int)gridDim.x);
    out[BIT_REVERSE(threadIdx.x + colOffset, leading_bits) * n] = { shared[threadIdx.x].x * scalar, shared[threadIdx.x].y *scalar };
    out[BIT_REVERSE(in_high + colOffset, leading_bits) * n] = { shared[in_high].x * scalar, shared[in_high].y *scalar };
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

__global__ void _kernelGPUS2DRowSurf(cuSurf in, cuSurf out, float angle, float local_angle, int steps_left, cpx scalar, int n_per_block)
{
    extern __shared__ cpx shared[];
    int in_high = (n_per_block >> 1) + threadIdx.x;
    int rowOffset = blockIdx.y * blockDim.x * 2;
    mem_gtos_row(threadIdx.x, in_high, rowOffset, shared, in);
    SYNC_THREADS;
    cuda_algorithm_local(shared, in_high, local_angle, steps_left);
    mem_stog_dbt_row(threadIdx.x, in_high, rowOffset, (32 - log2((int)gridDim.x)), scalar, shared, out);
}

__global__ void _kernelGPUS2DColSurf(cuSurf in, cuSurf out, float angle, float local_angle, int steps_left, cpx scalar, int n)
{
    extern __shared__ cpx shared[];
    int in_high = n >> 1;
    int colOffset = blockIdx.y * blockDim.x * 2;
    in_high += threadIdx.x;
    mem_gtos_col((int)threadIdx.x, in_high, colOffset, shared, in);
    SYNC_THREADS;
    cuda_algorithm_local(shared, in_high, local_angle, steps_left);
    mem_stog_db_col(threadIdx.x, in_high, colOffset, 32 - log2((int)gridDim.x), scalar, shared, out);
}

__host__ void tsCombine2DSurf_help(fftDir dir, cuSurf *surfaceIn, cuSurf *surfaceOut, int row_wise, int n)
{
    dim3 blocks;
    int threads;
    int steps_left = log2_32(n);
    const int steps_gpu = log2_32(MAX_BLOCK_SIZE);
    cpx scaleCpx = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
    set_block_and_threads2D(&blocks, &threads, n);

    const int n_per_block = n / blocks.y;
    const float global_angle = dir * (M_2_PI / n);
    const float local_angle = dir * (M_2_PI / n_per_block);
    int block_range_half = n;
    if (blocks.y > 1) {

        // Calculate sequence until parts fit into a block, syncronize on CPU until then.
        --steps_left;
        int steps = 0;
        int dist = (n / 2);
        ROW_COL_KERNEL(row_wise, _kernelAll2DRowSurf, _kernelAll2DColSurf) KERNEL_ARGS2(blocks, threads)(*surfaceIn, *surfaceOut, global_angle, 0xFFFFFFFF << steps_left, steps, dist);
        cudaDeviceSynchronize();
        while (--steps_left > steps_gpu) {
            dist >>= 1;
            ++steps;
            ROW_COL_KERNEL(row_wise, _kernelAll2DRowSurf, _kernelAll2DColSurf) KERNEL_ARGS2(blocks, threads)(*surfaceOut, *surfaceOut, global_angle, 0xFFFFFFFF << steps_left, steps, dist);
            cudaDeviceSynchronize();
        }
        cuda_surface_swap(surfaceIn, surfaceOut);
        ++steps_left;
        block_range_half = n_per_block;
    }
    // Calculate complete sequence in one launch and syncronize on GPU
    ROW_COL_KERNEL(row_wise, _kernelGPUS2DRowSurf, _kernelGPUS2DColSurf) KERNEL_ARGS3(blocks, threads, sizeof(cpx) * n_per_block) (*surfaceIn, *surfaceOut, global_angle, local_angle, steps_left, scaleCpx, block_range_half);
    cudaDeviceSynchronize();
}

__host__ void tsCombine2DSurf(fftDir dir, cuSurf *surfaceIn, cuSurf *surfaceOut, int n)
{
    tsCombine2DSurf_help(dir, surfaceIn, surfaceOut, 1, n);
    tsCombine2DSurf_help(dir, surfaceOut, surfaceIn, 1, n);
    cuda_surface_swap(surfaceIn, surfaceOut);
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
            cpx *tmp = (cpx *)malloc(sizeof(cpx) * n * n);
            fftShift(tmp, in, n);
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
    return average_best(measures, NUM_PERFORMANCE);
}