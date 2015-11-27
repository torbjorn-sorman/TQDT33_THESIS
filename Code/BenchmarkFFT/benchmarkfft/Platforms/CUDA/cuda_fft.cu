#include "cuda_fft.cuh"
#if defined(_NVIDIA)
// Comment: Max values all over seems to give the biggest benefit when running large sets of data
//          Smaller sets benefits a little from smaller groups/blocks.
//          I my opinion, the large values seems to suit CUDA best.

#define CU_BLOCK_SIZE 1024
#define CU_TILE_DIM 64 // Sets local/shared mem when transposing
#define CU_BLOCK_DIM 32 // Sets threads when transposing

__global__ void cuda_kernel_global(cpx *in, float global_angle, unsigned int lmask, int steps, int dist);
__global__ void cuda_kernel_global_row(cpx *in, float global_angle, unsigned int lmask, int steps, int dist);

__global__ void cuda_kernel_local(cpx *in, cpx *out, float local_angle, int steps_left, int leading_bits, float scalar, int block_range);
__global__ void cuda_kernel_local_row(cpx *in, cpx *out, float local_angle, int steps_left, int leading_bits, float scalar, int block_range);

__global__ void cuda_transpose_kernel(cpx *in, cpx *out, int n);

// -------------------------------
//
// Testing
//
// -------------------------------

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
    return (cuda_shakedown(n, &dev_in, &dev_out, &in, &ref, &out) != 1) && (diff <= RELATIVE_ERROR_MARGIN);
}

__host__ int cuda_2d_validate(int n, bool write_img)
{
    cpx *host_buffer, *ref, *dev_in, *dev_out;
    size_t size;
    cuda_setup_buffers_2d(&host_buffer, &ref, &dev_in, &dev_out, &size, n);
    cudaMemcpy(dev_in, host_buffer, size, cudaMemcpyHostToDevice);
    cuda_fft_2d(FFT_FORWARD, &dev_in, &dev_out, n);
    cudaDeviceSynchronize();
    if (write_img) {
        cudaMemcpy(host_buffer, dev_out, size, cudaMemcpyDeviceToHost);
        write_normalized_image("CUDA", "freq", host_buffer, n, true);
    }
    cuda_fft_2d(FFT_INVERSE, &dev_out, &dev_in, n);
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
    double measures[64];
    cpx *in, *ref, *out, *dev_in, *dev_out;
    cuda_setup_buffers(n, &dev_in, &dev_out, &in, &ref, &out);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    for (int i = 0; i < number_of_tests; ++i) {
        start_timer();
        cuda_fft(FFT_FORWARD, dev_in, dev_out, n);
        cudaDeviceSynchronize();
        measures[i] = stop_timer();
    }
    cuda_shakedown(n, &dev_in, &dev_out, &in, &ref, &out);
    double t = average_best(measures, number_of_tests);
    return t;
}
__host__ double cuda_2d_performance(int n)
{
    double measures[64];
    cpx *in, *ref, *dev_in, *dev_out;
    size_t size;
    cuda_setup_buffers_2d(&in, &ref, &dev_in, &dev_out, &size, n);
    for (int i = 0; i < number_of_tests; ++i) {
        start_timer();
        cuda_fft_2d(FFT_FORWARD, &dev_in, &dev_out, n);
        cudaDeviceSynchronize();
        measures[i] = stop_timer();
    }
    cuda_shakedown_2d(&in, &ref, &dev_in, &dev_out);
    return average_best(measures, number_of_tests);
}
#else
__host__ double cuda_performance(int n)
{
    double measures[64];
    cpx *in, *ref, *out, *dev_in, *dev_out;
    cuda_setup_buffers(n, &dev_in, &dev_out, &in, &ref, &out);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    for (int i = 0; i < number_of_tests; ++i) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        cuda_fft(FFT_FORWARD, dev_in, dev_out, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        measures[i] = milliseconds * 1000;
    }
    cuda_shakedown(n, &dev_in, &dev_out, &in, &ref, &out);
    double t = average_best(measures, number_of_tests);
    return t;
}
__host__ double cuda_2d_performance(int n)
{
    double measures[64];
    cpx *in, *ref, *dev_in, *dev_out;
    size_t size;
    cuda_setup_buffers_2d(&in, &ref, &dev_in, &dev_out, &size, n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaDeviceSynchronize();
    for (int i = 0; i < number_of_tests; ++i) {
        cudaEventRecord(start);
        cuda_fft_2d(FFT_FORWARD, &dev_in, &dev_out, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        measures[i] = milliseconds * 1000;
    }
    cuda_shakedown_2d(&in, &ref, &dev_in, &dev_out);
    return average_best(measures, number_of_tests);
}
#endif

// -------------------------------
//
// Algorithm
//
// -------------------------------

__host__ __inline void set_block_and_threads(dim3 *number_of_blocks, int *threads_per_block, const int block_size, const bool dim2, const int n)
{
    const int n_half = n >> 1;
    const bool multi_blocks = (n_half > block_size);
    *threads_per_block = multi_blocks ? block_size : n_half;
    number_of_blocks->x = dim2 ? n : multi_blocks ? n_half / block_size : 1;
    number_of_blocks->y = dim2 ? multi_blocks ? n_half / block_size : 1 : number_of_blocks->x;
}

__global__ void exp_cuda_kernel(cpx *in, float angle, float global_angle, unsigned int steps_left, int steps_local, int n_half);
__global__ void exp_cuda_kernel_local_row(cpx *in, cpx *out, float local_angle, int steps_left, int leading_bits, float scalar, int block_range);

#define TEST_1D

__host__ void cuda_fft(transform_direction dir, cpx *in, cpx *out, int n)
{
    fft_args args;
    int threads, blocks;
    set_block_and_threads(&blocks, &threads, CU_BLOCK_SIZE, (n >> 1));
    set_fft_arguments(&args, dir, blocks, CU_BLOCK_SIZE, n);

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    if (blocks > 1) {
#if defined(TEST_1D)
        if (args.steps_left == args.steps_gpu + 1) {
            cuda_kernel_global KERNEL_ARGS2(blocks, threads)(in, args.global_angle, 0xFFFFFFFF << args.steps_left, args.steps++, args.dist >>= 1);
        }
        else {

            exp_cuda_kernel KERNEL_ARGS2(blocks, threads)(in, a.local_angle, a.global_angle, a.steps_left, args.steps_gpu, n >> 1);
        }
        args.steps_left = args.steps_gpu + 1;
#else
        while (--args.steps_left > args.steps_gpu) {
            cuda_kernel_global KERNEL_ARGS2(blocks, threads)(in, args.global_angle, 0xFFFFFFFF << args.steps_left, args.steps++, args.dist >>= 1);
        }
        ++args.steps_left;
#endif
    }
    cuda_kernel_local KERNEL_ARGS3(blocks, threads, sizeof(cpx) * args.n_per_block) (in, out, args.local_angle, args.steps_left, args.leading_bits, args.scalar, args.block_range);
}

//#define TEST_2D

__host__ __inline void cuda_fft_2d_helper(transform_direction dir, cpx *dev_in, cpx *dev_out, int n)
{
    fft_args args;
    dim3 blocks;
    int threads;
    set_block_and_threads_2d(&blocks, &threads, CU_BLOCK_SIZE, n);
    set_fft_arguments(&args, dir, blocks.y, CU_BLOCK_SIZE, n);
    if (blocks.y > 1) {
        while (--args.steps_left > args.steps_gpu) {
            cuda_kernel_global_row KERNEL_ARGS2(blocks, threads)(dev_in, args.global_angle, 0xFFFFFFFF << args.steps_left, args.steps++, args.dist >>= 1);
        }
        ++args.steps_left;
    }
#if defined(TEST_2D)
    exp_cuda_kernel_local_row KERNEL_ARGS3(blocks, threads, sizeof(cpx) * args.n_per_block) (dev_in, dev_out, args.local_angle, args.steps_left, args.leading_bits, args.scalar, args.block_range);
#else
    cuda_kernel_local_row KERNEL_ARGS3(blocks, threads, sizeof(cpx) * args.n_per_block) (dev_in, dev_out, args.local_angle, args.steps_left, args.leading_bits, args.scalar, args.block_range);
#endif
}

__host__ void cuda_fft_2d(transform_direction dir, cpx **dev_in, cpx **dev_out, int n)
{
#if defined(TEST_2D)
    cuda_fft_2d_helper(dir, *dev_in, *dev_out, n);
    cuda_fft_2d_helper(dir, *dev_out, *dev_in, n);
    swap_buffer(dev_in, dev_out);
#else
    dim3 blocks;
    dim3 threads;
    set_block_and_threads_transpose(&blocks, &threads, CU_TILE_DIM, CU_BLOCK_DIM, n);    
    cuda_fft_2d_helper(dir, *dev_in, *dev_out, n);
    cuda_transpose_kernel KERNEL_ARGS2(blocks, threads) (*dev_out, *dev_in, n);
    cuda_fft_2d_helper(dir, *dev_in, *dev_out, n);
    cuda_transpose_kernel KERNEL_ARGS2(blocks, threads) (*dev_out, *dev_in, n);
    swap_buffer(dev_in, dev_out);
#endif
}

// -------------------------------
//
// Device
//
// -------------------------------

//--- Experimental ---

__device__ __inline void cu_contant_geometry(cpx *shared, cpx *in_l, cpx *in_h, float angle, int steps_limit);

__device__ __inline void cu_cooley_tukey(cpx *low, float angle, float angle_offset, int steps_limit, int dist)
{
    cpx w, l, h;
    float x, y;
    for (int steps = 0; steps < steps_limit; ++steps) {
        l = *low;
        h = *(low + dist);
        x = l.x - h.x;
        y = l.y - h.y;
        SIN_COS_F(angle * ((threadIdx.x << steps) & ((dist - 1) << steps)) + angle_offset, &w.y, &w.x);    // c-t    
        SYNC_THREADS;
        *low = { l.x + h.x, l.y + h.y };
        *(low + dist) = { (w.x * x) - (w.y * y), (w.y * x) + (w.x * y) };
        dist >>= 1;
        SYNC_THREADS;
    }
}

__global__ void exp_cuda_kernel(cpx *in, float angle, float global_angle, unsigned int steps_left, int steps_local, int n_half)
{
    __shared__ cpx shared[256];
    int in_l = threadIdx.x * (n_half >> steps_local) + blockIdx.x,
        in_h = in_l + n_half;
    cpx *sh_l = shared + threadIdx.x,
        *sh_h = sh_l + (blockDim.x >> 1);

    *sh_l = in[in_l];
    *sh_h = in[in_h];
    SYNC_THREADS;

    cu_contant_geometry(shared, sh_l, sh_h, angle + global_angle * blockIdx.x, steps_left - steps_local);

    in[in_l] = *sh_l;
    in[in_h] = *sh_h;
}
/*
__device__ __inline void cuda_partial(cpx *in, cpx *out, cpx *shared, unsigned int in_high, unsigned int offset, float local_angle, int steps_left, int leading_bits, float scalar)
{
cpx *in_l = shared + threadIdx.x,
*in_u = shared + in_high;
*in_l = in[threadIdx.x];
*in_u = in[in_high];

cu_contant_geometry(shared, in_l, in_u, local_angle, steps_left);

cuda_to_global(out, threadIdx.x + offset, leading_bits, in_l, scalar);
cuda_to_global(out, in_high + offset, leading_bits, in_u, scalar);
}
*/

#define GLOBAL_Y (blockIdx.x)
#define GLOBAL_X (((blockIdx.y * blockDim.x) << 1) + threadIdx.x)
#define ROW_DIM (gridDim.x)

__global__ void exp_cuda_kernel_local_row(cpx *in, cpx *out, float local_angle, int steps_left, int leading_bits, float scalar, int block_range)
{
    extern __shared__ cpx shared[];
    cpx *in_l = shared + threadIdx.x,
        *in_u = in_l + block_range;

    in += GLOBAL_Y * ROW_DIM + GLOBAL_X;
    *in_l = *in;
    *in_u = in[block_range];

    cu_contant_geometry(shared, in_l, in_u, local_angle, steps_left);

    out += GLOBAL_Y;
    out[BIT_REVERSE(GLOBAL_X, leading_bits) * ROW_DIM] = cpx{ in_l->x * scalar, in_l->y * scalar };
    out[BIT_REVERSE(GLOBAL_X + block_range, leading_bits) * ROW_DIM] = cpx{ in_u->x * scalar, in_u->y * scalar };
}

//--- END Experimental END ---

__device__ __inline void cu_contant_geometry(cpx *shared, cpx *in_l, cpx *in_h, float angle, int steps_limit)
{
    cpx w, l, h;
    cpx *out_i = shared + (threadIdx.x << 1),
        *out_ii = out_i + 1;
    float x, y;
    for (int steps = 0; steps < steps_limit; ++steps) {
        l = *in_l;
        h = *in_h;
        x = l.x - h.x;
        y = l.y - h.y;
        SIN_COS_F(angle * (threadIdx.x & (0xFFFFFFFF << steps)), &w.y, &w.x);
        SYNC_THREADS;
        *out_i = { l.x + h.x, l.y + h.y };
        *out_ii = { (w.x * x) - (w.y * y), (w.y * x) + (w.x * y) };
        SYNC_THREADS;
    }
}

__device__ __inline void cu_global(cpx *in, int tid, float angle, int steps, int dist)
{
    cpx w;
    SIN_COS_F(angle * ((tid << steps) & ((dist - 1) << steps)), &w.y, &w.x);
    cpx l = *in;
    cpx h = in[dist];
    float x = l.x - h.x;
    float y = l.y - h.y;
    *in = { l.x + h.x, l.y + h.y };
    in[dist] = { (w.x * x) - (w.y * y), (w.y * x) + (w.x * y) };
}

__global__ void cuda_kernel_global(cpx *in, float angle, unsigned int lmask, int steps, int dist)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    cu_global(in + tid + (tid & lmask), tid, angle, steps, dist);
}

__global__ void cuda_kernel_global_row(cpx *in, float angle, unsigned int lmask, int steps, int dist)
{
    int col_id = blockIdx.y * blockDim.x + threadIdx.x;
    cu_global(in + (col_id + (col_id & lmask)) + blockIdx.x * gridDim.x, col_id, angle, steps, dist);
}

__device__ __inline void cuda_to_global(cpx *out, unsigned int dst, unsigned int leading, cpx *sh, float c)
{
    out[BIT_REVERSE(dst, leading)] = cpx{ sh->x * c, sh->y * c };
}

__device__ __inline void cuda_partial(cpx *in, cpx *out, cpx *shared, unsigned int in_high, unsigned int offset, float local_angle, int steps_left, int leading_bits, float scalar)
{
    cpx *in_l = shared + threadIdx.x,
        *in_u = shared + in_high;
    *in_l = in[threadIdx.x];
    *in_u = in[in_high];

    cu_contant_geometry(shared, in_l, in_u, local_angle, steps_left);

    cuda_to_global(out, threadIdx.x + offset, leading_bits, in_l, scalar);
    cuda_to_global(out, in_high + offset, leading_bits, in_u, scalar);
}

__global__ void cuda_kernel_local(cpx *in, cpx *out, float local_angle, int steps_left, int leading_bits, float scalar, int block_range)
{
    extern __shared__ cpx shared[];
    int offset = blockIdx.x * blockDim.x * 2;
    cuda_partial(in + offset, out, shared, threadIdx.x + block_range, offset, local_angle, steps_left, leading_bits, scalar);
}

__global__ void cuda_kernel_local_row(cpx *in, cpx *out, float local_angle, int steps_left, int leading_bits, float scalar, int block_range)
{
    extern __shared__ cpx shared[];
    int row_start = gridDim.x * blockIdx.x;
    int offset = (blockIdx.y * blockDim.x) << 1;
    cuda_partial(in + row_start + offset, out + row_start, shared, block_range + threadIdx.x, offset, local_angle, steps_left, leading_bits, scalar);
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
#endif