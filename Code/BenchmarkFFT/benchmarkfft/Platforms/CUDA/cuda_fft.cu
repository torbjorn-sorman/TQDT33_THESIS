#include "cuda_fft.cuh"

#define CU_BLOCK_SIZE 1024
#define CU_TILE_DIM 64 // 32K local/shared mem
#define CU_BLOCK_DIM 32 // 1024 Threads

__global__ void cuda_kernel_global(cpx *in, float global_angle, unsigned int lmask, int steps, int dist);
__global__ void cuda_kernel_global_row(cpx *in, float global_angle, unsigned int lmask, int steps, int dist);

__global__ void cuda_kernel_local(cpx *in, cpx *out, float local_angle, int steps_left, int leading_bits, float scalar, int block_range_half);
__global__ void cuda_kernel_local_row(cpx *in, cpx *out, float local_angle, int steps_left, int leading_bits, float scalar, int block_range_half);

__global__ void cuda_transpose_kernel(cpx *in, cpx *out, int n);

// -------------------------------
//
// Testing
//
// -------------------------------

#include "../../Common/cpx_debug.h"

__host__ int cuda_validate(int n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out;
    cuda_setup_buffers(n, &dev_in, &dev_out, &in, &ref, &out);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    cuda_fft(FFT_FORWARD, dev_in, dev_out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(out, dev_out, n * sizeof(cpx), cudaMemcpyDeviceToHost);
    double diff = diff_forward_sinus(out, n);
    cuda_fft(FFT_INVERSE, dev_out, dev_in, n);
    cudaDeviceSynchronize();
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);

#ifdef SHOW_BLOCKING_DEBUG
    cpx_to_console(in, "CUDA Out", 32);
    printf("%f\n", diff);
    getchar();
#endif

    return (cuda_shakedown(n, &dev_in, &dev_out, &in, &ref, &out) != 1) && (diff <= RELATIVE_ERROR_MARGIN);
}

__host__ int cuda_2d_validate(int n, bool write_img)
{
    cpx *host_buffer, *ref, *dev_in, *dev_out;
    size_t size;
    cuda_setup_buffers_2d(&host_buffer, &ref, &dev_in, &dev_out, &size, n);

    cudaMemcpy(dev_in, host_buffer, size, cudaMemcpyHostToDevice);
    cuda_fft_2d(FFT_FORWARD, dev_in, dev_out, n);
    cudaDeviceSynchronize();
    if (write_img) {
        cudaMemcpy(host_buffer, dev_out, size, cudaMemcpyDeviceToHost);
#ifndef TRANSPOSE_ONLY
        write_normalized_image("CUDA", "freq", host_buffer, n, true);
#else
        write_image("CUDA", "trans", host_buffer, n);
#endif
    }
    cuda_fft_2d(FFT_INVERSE, dev_out, dev_in, n);
    cudaDeviceSynchronize();
    if (write_img) {
        cudaMemcpy(host_buffer, dev_in, size, cudaMemcpyDeviceToHost);
        write_image("CUDA", "spat", host_buffer, n);
    }

    int res = cuda_compare_result(host_buffer, ref, dev_in, size, n * n);
    cuda_shakedown_2d(&host_buffer, &ref, &dev_in, &dev_out);
    return res;
}

#ifndef MEASURE_BY_TIMESTAMP
__host__ double cuda_performance(int n)
{
    double measures[NUM_TESTS];
    cpx *in, *ref, *out, *dev_in, *dev_out;
    cuda_setup_buffers(n, &dev_in, &dev_out, &in, &ref, &out);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_TESTS; ++i) {
        startTimer();
        cuda_fft(FFT_FORWARD, dev_in, dev_out, n);
        cudaDeviceSynchronize();
        measures[i] = stopTimer();
    }
    cuda_shakedown(n, &dev_in, &dev_out, &in, &ref, &out);
    double t = average_best(measures, NUM_TESTS);
    return t;
}
__host__ double cuda_2d_performance(int n)
{
    double measures[NUM_TESTS];
    cpx *in, *ref, *dev_in, *dev_out;
    size_t size;
    cuda_setup_buffers_2d(&in, &ref, &dev_in, &dev_out, &size, n);
    for (int i = 0; i < NUM_TESTS; ++i) {
        startTimer();
        cuda_fft_2d(FFT_FORWARD, dev_in, dev_out, n);
        cudaDeviceSynchronize();
        measures[i] = stopTimer();
    }
    cuda_shakedown_2d(&in, &ref, &dev_in, &dev_out);
    return average_best(measures, NUM_TESTS);
}
#else
__host__ double cuda_performance(int n)
{
    double measures[NUM_TESTS];
    cpx *in, *ref, *out, *dev_in, *dev_out;
    cuda_setup_buffers(n, &dev_in, &dev_out, &in, &ref, &out);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    for (int i = 0; i < NUM_TESTS; ++i) {
        cudaEventRecord(start);
        cuda_fft(FFT_FORWARD, dev_in, dev_out, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&milliseconds, start, stop);
        measures[i] = milliseconds * 1000;
    }
    cuda_shakedown(n, &dev_in, &dev_out, &in, &ref, &out);
    double t = average_best(measures, NUM_TESTS);
    return t;
}
__host__ double cuda_2d_performance(int n)
{
    double measures[NUM_TESTS];
    cpx *in, *ref, *dev_in, *dev_out;
    size_t size;
    cuda_setup_buffers_2d(&in, &ref, &dev_in, &dev_out, &size, n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaDeviceSynchronize();
    for (int i = 0; i < NUM_TESTS; ++i) {
        cudaEventRecord(start);
        cuda_fft_2d(FFT_FORWARD, dev_in, dev_out, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&milliseconds, start, stop);
        measures[i] = milliseconds * 1000;
    }
    cuda_shakedown_2d(&in, &ref, &dev_in, &dev_out);
    return average_best(measures, NUM_TESTS);
}
#endif

// -------------------------------
//
// Algorithm
//
// -------------------------------

__host__ void cuda_fft(transform_direction dir, cpx *in, cpx *out, int n)
{
    int threads, blocks;
    int n_half = (n >> 1);
    int steps_left = log2_32(n);
    int leading_bits = 32 - steps_left;
    float scalar = (dir == FFT_FORWARD ? 1.f : 1.f / n);
    set_block_and_threads(&blocks, &threads, CU_BLOCK_SIZE, n_half);
    int n_per_block = n / blocks;
    float local_angle = dir * (M_2_PI / n_per_block);
    int block_range_half = n_half;
    /*
    if (blocks > 1) {
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
        float global_angle = dir * (M_2_PI / n);
        int steps = 0;
        int dist = n;
        int steps_gpu = log2_32(CU_BLOCK_SIZE);
        while (--steps_left > steps_gpu) {
            cuda_kernel_global KERNEL_ARGS2(blocks, threads)(in, global_angle, 0xFFFFFFFF << steps_left, steps++, dist >>= 1);
        }
        ++steps_left;
        block_range_half = n_per_block >> 1;
    }
    */
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cuda_kernel_local KERNEL_ARGS3(blocks, threads, sizeof(cpx) * n_per_block) (in, out, local_angle, steps_left, leading_bits, scalar, block_range_half);
}

__host__ static __inline void cuda_fft_2d_helper(transform_direction dir, cpx *dev_in, cpx *dev_out, int n)
{
    dim3 blocks;
    int threads;
    int steps_left = log2_32(n);
    int leading_bits = 32 - steps_left;
    float scalar = (dir == FFT_FORWARD ? 1.f : 1.f / n);
    set_block_and_threads2D(&blocks, &threads, CU_BLOCK_SIZE, n);
    const int n_per_block = n / blocks.y;
    const float global_angle = dir * (M_2_PI / n);
    const float local_angle = dir * (M_2_PI / n_per_block);
    int block_range = n;
    if (blocks.y > 1) {
        // Calculate sequence until parts fit into a block, syncronize on CPU until then.
        const int steps_gpu = log2_32(CU_BLOCK_SIZE);
        int steps = 0;
        int dist = n;
        // Instead of swapping input/output, run in place. The arg_gpu kernel needs to swap once.                
        while (--steps_left > steps_gpu) {
            cuda_kernel_global_row KERNEL_ARGS2(blocks, threads)(dev_in, global_angle, 0xFFFFFFFF << steps_left, steps++, dist >>= 1);
        }
        ++steps_left;
        block_range = n_per_block;
    }
    // Calculate complete sequence in one launch and syncronize on GPU
    cuda_kernel_local_row KERNEL_ARGS3(blocks, threads, sizeof(cpx) * n_per_block) (dev_in, dev_out, local_angle, steps_left, leading_bits, scalar, block_range >> 1);
}

__host__ void cuda_fft_2d(transform_direction dir, cpx *dev_in, cpx *dev_out, int n)
{
    dim3 blocks;
    dim3 threads;
    set_block_and_threads_transpose(&blocks, &threads, CU_TILE_DIM, CU_BLOCK_DIM, n);
#ifndef TRANSPOSE_ONLY
    cuda_fft_2d_helper(dir, dev_in, dev_out, n);
    cuda_transpose_kernel KERNEL_ARGS2(blocks, threads) (dev_out, dev_in, n);
    cuda_fft_2d_helper(dir, dev_in, dev_out, n);
    cuda_transpose_kernel KERNEL_ARGS2(blocks, threads) (dev_out, dev_in, n);        
    swap_buffer(&dev_in, &dev_out);
#else
    cuda_transpose_kernel KERNEL_ARGS2(blocks, threads) (dev_in, dev_out, n);
#endif
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

__global__ void cuda_kernel_global(cpx *in, float angle, unsigned int lmask, int steps, int dist)
{
    cpx w;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    in += tid + (tid & lmask);
    SIN_COS_F(angle * ((tid << steps) & ((dist - 1) << steps)), &w.y, &w.x);
    cpx_add_sub_mul(in, in + dist, &w);
}

__global__ void cuda_kernel_global_row(cpx *in, float angle, unsigned int lmask, int steps, int dist)
{
    cpx w;
    int col_id = blockIdx.y * blockDim.x + threadIdx.x;
    in += (col_id + (col_id & lmask)) + blockIdx.x * gridDim.x;
    SIN_COS_F(angle * ((col_id << steps) & ((dist - 1) << steps)), &w.y, &w.x);
    cpx_add_sub_mul(in, in + dist, &w);
}

// Full blown block syncronized algorithm! In theory this should scalar up but is limited by hardware (#cores)
__global__ void cuda_kernel_local(cpx *in, cpx *out, float local_angle, int steps_left, int leading_bits, float scalar, int block_range_half)
{
    extern __shared__ cpx shared[];
    int in_high = threadIdx.x + block_range_half;
    int offset = blockIdx.x * blockDim.x * 2;
    in += offset;
    shared[threadIdx.x] = in[threadIdx.x];
    shared[in_high] = in[in_high];
    cuda_algorithm_local(shared, in_high, local_angle, steps_left);
    out[BIT_REVERSE(threadIdx.x + offset, leading_bits)] = { shared[threadIdx.x].x * scalar, shared[threadIdx.x].y *scalar };
    out[BIT_REVERSE(in_high + offset, leading_bits)] = { shared[in_high].x * scalar, shared[in_high].y *scalar };
}

__global__ void cuda_kernel_local_row(cpx *in, cpx *out, float local_angle, int steps_left, int leading_bits, float scalar, int block_range_half)
{
    extern __shared__ cpx shared[];
    int in_high = block_range_half + threadIdx.x;
    int row_start = gridDim.x * blockIdx.x;
    int row_offset = (blockIdx.y * blockDim.x) << 1;
    in += row_start + row_offset;
    out += row_start;
    shared[threadIdx.x] = in[threadIdx.x];
    shared[in_high] = in[in_high];
    cuda_algorithm_local(shared, in_high, local_angle, steps_left);
    out[BIT_REVERSE(threadIdx.x + row_offset, leading_bits)] = { shared[threadIdx.x].x * scalar, shared[threadIdx.x].y *scalar };
    out[BIT_REVERSE(in_high + row_offset, leading_bits)] = { shared[in_high].x * scalar, shared[in_high].y *scalar };
}

__global__ void cuda_transpose_kernel(cpx *in, cpx *out, int n)
{
    // Banking issues when CU_TILE_DIM % WARP_SIZE == 0, current WARP_SIZE == 32
    __shared__ cpx tile[CU_TILE_DIM][CU_TILE_DIM + 1];

    // Write to shared from Global (in)
    int x = blockIdx.x * CU_TILE_DIM + threadIdx.x;
    int y = blockIdx.y * CU_TILE_DIM + threadIdx.y;
    for (int j = 0; j < CU_TILE_DIM; j += CU_BLOCK_DIM)
        for (int i = 0; i < CU_TILE_DIM; i += CU_BLOCK_DIM)
            tile[threadIdx.y + j][threadIdx.x + i] = in[(y + j) * n + (x + i)];

    SYNC_THREADS;
    // Write to global
    x = blockIdx.y * CU_TILE_DIM + threadIdx.x;
    y = blockIdx.x * CU_TILE_DIM + threadIdx.y;
    for (int j = 0; j < CU_TILE_DIM; j += CU_BLOCK_DIM)
        for (int i = 0; i < CU_TILE_DIM; i += CU_BLOCK_DIM)
            out[(y + j) * n + (x + i)] = tile[threadIdx.x + i][threadIdx.y + j];
}