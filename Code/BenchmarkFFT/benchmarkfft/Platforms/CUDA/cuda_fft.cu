#include "cuda_fft.cuh"

__global__ void cuda_kernel_global(     cpx *in, float global_angle, unsigned int lmask, int steps, int dist);
__global__ void cuda_kernel_global_row( cpx *in, float global_angle, unsigned int lmask, int steps, int dist);

__global__ void cuda_kernel_local(      cpx *in, cpx *out, float local_angle, int steps_left, int leading_bits, float scalar, int block_range_half);
__global__ void cuda_kernel_local_row(  cpx *in, cpx *out, float local_angle, int steps_left, int leading_bits, float scalar, int block_range_half);
__global__ void cuda_kernel_local_col(  cpx *in, cpx *out, float local_angle, int steps_left, int leading_bits, float scalar, int block_range);

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

__host__ void testCombine2DRun(transform_direction dir, cpx *in, cpx *dev_in, cpx *dev_out, size_t size, bool write, bool norm, int n)
{
    cuda_fft_2d(dir, dev_in, dev_out, n);
    if (write) {
        cudaMemcpy(in, dev_out, size, cudaMemcpyDeviceToHost);
        if (norm) {
            write_image("CUDA", "frequency - not norm", in, n);
            write_normalized_image("CUDA", "freq", in, n, true);
        }
        else
            write_image("CUDA", "spat", in, n);
    }
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
        write_normalized_image("CUDA", "freq", host_buffer, n, true);
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
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *out, *dev_in, *dev_out;
    cuda_setup_buffers(n, &dev_in, &dev_out, &in, &ref, &out);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        cuda_fft(FFT_FORWARD, &dev_in, &dev_out, n);
        cudaDeviceSynchronize();
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
        cudaDeviceSynchronize();
        measures[i] = stopTimer();
    }
    cuda_shakedown_2d(&in, &ref, &dev_in, &dev_out);
    return average_best(measures, NUM_PERFORMANCE);
}
#else
__host__ double cuda_performance(int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *out, *dev_in, *dev_out;
    cuda_setup_buffers(n, &dev_in, &dev_out, &in, &ref, &out);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {        
        cudaEventRecord(start);
        cuda_fft(FFT_FORWARD, dev_in, dev_out, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);        
        measures[i] = milliseconds * 1000;
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        cudaEventRecord(start);
        cuda_fft_2d(FFT_FORWARD, dev_in, dev_out, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        measures[i] = milliseconds * 1000;
    }
    cuda_shakedown_2d(&in, &ref, &dev_in, &dev_out);
    return average_best(measures, NUM_PERFORMANCE);
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
    set_block_and_threads(&blocks, &threads, n_half);    
    int n_per_block = n / blocks;
    float global_angle = dir * (M_2_PI / n);
    float local_angle = dir * (M_2_PI / n_per_block);
    int block_range_half = n_half;
    if (blocks > 1) {
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);        
        int steps = 0;
        int dist = n;
        int steps_gpu = log2_32(MAX_BLOCK_SIZE);
        while (--steps_left > steps_gpu) {
            cuda_kernel_global KERNEL_ARGS2(blocks, threads)(in, global_angle, 0xFFFFFFFF << steps_left, steps++, dist >>= 1);
        }
        ++steps_left;
        block_range_half = n_per_block >> 1;        
    }
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
    set_block_and_threads2D(&blocks, &threads, n);
    const int n_per_block = n / blocks.y;
    const float global_angle = dir * (M_2_PI / n);
    const float local_angle = dir * (M_2_PI / n_per_block);
    int block_range = n;
    if (blocks.y > 1) {
        // Calculate sequence until parts fit into a block, syncronize on CPU until then.
        const int steps_gpu = log2_32(MAX_BLOCK_SIZE);
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
    if (n > 1) {
        dim3 threads;
        set_block_and_threads_transpose(&blocks, &threads, n);
        cuda_fft_2d_helper(dir, dev_in, dev_out, n);
        cuda_transpose_kernel KERNEL_ARGS2(blocks, threads) (dev_out, dev_in, n);
        cuda_fft_2d_helper(dir, dev_in, dev_out, n);
        cuda_transpose_kernel KERNEL_ARGS2(blocks, threads) (dev_out, dev_in, n);        
    }
    else {
        int threads;
        const int steps_left = log2_32(n);
        const int leading_bits = 32 - steps_left;
        const float scalar = (dir == FFT_FORWARD ? 1.f : 1.f / n);
        const float global_angle = dir * (M_2_PI / n);
        set_block_and_threads2D(&blocks, &threads, n);
        cuda_kernel_local_row KERNEL_ARGS3(blocks, threads, sizeof(cpx) * n) (dev_in, dev_out, global_angle, steps_left, leading_bits, scalar, n >> 1);
        cuda_kernel_local_col KERNEL_ARGS3(blocks, threads, sizeof(cpx) * n) (dev_out, dev_in, global_angle, steps_left, leading_bits, scalar, n);
    }
    swap_buffer(&dev_in, &dev_out);
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
    SYNC_THREADS;
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

__global__ void cuda_kernel_local_col(cpx *in, cpx *out, float local_angle, int steps_left, int leading_bits, float scalar, int block_range)
{
    extern __shared__ cpx shared[];

    int in_high = (block_range >> 1) + threadIdx.x;
    int colOffset = blockIdx.y * blockDim.x * 2;
    in += (threadIdx.x + colOffset) * block_range + blockIdx.x;
    out += blockIdx.x;
    shared[threadIdx.x] = *in;
    shared[in_high] = *(in + ((block_range >> 1) * block_range));
    SYNC_THREADS;
    cuda_algorithm_local(shared, in_high, local_angle, steps_left);
    out[BIT_REVERSE(threadIdx.x + colOffset, leading_bits) * block_range] = { shared[threadIdx.x].x * scalar, shared[threadIdx.x].y *scalar };
    out[BIT_REVERSE(in_high + colOffset, leading_bits) * block_range] = { shared[in_high].x * scalar, shared[in_high].y *scalar };
}