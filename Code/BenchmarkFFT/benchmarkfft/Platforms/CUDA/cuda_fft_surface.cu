#include "cuda_fft_surface.cuh"

void cuda_surface_setup(cpx **in, cpx **ref, size_t *size, int n, cudaArray **cuInputArray, cudaArray **cuOutputArray, cuSurf *inputSurfObj, cuSurf *outputSurfObj)
{
    char input_file[40];
    sprintf_s(input_file, 40, "Images/%u.ppm", n);
    int sz;
    *in = (cpx *)malloc(sizeof(cpx) * n * n);
    read_image(*in, input_file, &sz);
    *ref = (cpx *)malloc(sizeof(cpx) * n * n);
    memcpy(*ref, *in, sizeof(cpx) * n * n);
    *size = n * n * sizeof(cpx);
    // Allocate CUDA arrays in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();
    cudaMallocArray(cuInputArray, &channelDesc, n, n, cudaArraySurfaceLoadStore);
    cudaCheckError();
    if (cuOutputArray != NULL) {
        cudaMallocArray(cuOutputArray, &channelDesc, n, n, cudaArraySurfaceLoadStore);
    }
    // Specify surface
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    // Create the surface objects
    resDesc.res.array.array = *cuInputArray;
    *inputSurfObj = 0;
    cudaCreateSurfaceObject(inputSurfObj, &resDesc);
    cudaCheckError();
    if (outputSurfObj != NULL) {
        resDesc.res.array.array = *cuOutputArray;
        *outputSurfObj = 0;
        cudaCreateSurfaceObject(outputSurfObj, &resDesc);
        cudaCheckError();
    }
}

__host__ void cuda_surface_shakedown(cpx **in, cpx **ref, cuSurf *sObjIn, cuSurf *sObjOut, cudaArray **cuAIn, cudaArray **cuAOut)
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

__host__ int cuda_surface_compare(cpx *in, cpx *ref, cudaArray *dev, size_t size, int len)
{
    cudaMemcpyFromArray(in, dev, 0, 0, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < len; ++i) {
        if (cuCabsf(cuCsubf(in[i], ref[i])) > 0.0001) {
            return 0;
        }
    }
    return 1;
}

__host__ int cuda_surface_validate(int n)
{
    int res;
    cpx *in, *ref;
    size_t size;
    cudaArray *array_in, *array_out;
    cuSurf surface_in, surface_out;
    cuda_surface_setup(&in, &ref, &size, n, &array_in, &array_out, &surface_in, &surface_out);
    cudaMemcpyToArray(array_in, 0, 0, in, size, cudaMemcpyHostToDevice);

    cuda_surface_fft(FFT_FORWARD, &surface_in, &surface_out, n);
    cudaMemcpyFromArray(in, array_in, 0, 0, size, cudaMemcpyDeviceToHost);
    write_normalized_image("CUDA", "Surface-Frequency", in, n, true);

    cuda_surface_fft(FFT_INVERSE, &surface_out, &surface_in, n);
    cudaMemcpyFromArray(in, array_in, 0, 0, size, cudaMemcpyDeviceToHost);
    write_image("CUDA", "Surface-Spatial", in, n);

    res = cuda_surface_compare(in, ref, array_in, size, n * n);
    cuda_surface_shakedown(&in, &ref, &surface_in, &surface_out, &array_in, &array_out);
    return res;
}

__host__ double cuda_surface_performance(int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref;
    size_t size;
    cudaArray *array_in, *array_out;
    cuSurf surface_in, surface_out;
    cuda_surface_setup(&in, &ref, &size, n, &array_in, &array_out, &surface_in, &surface_out);

    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        cuda_surface_fft(FFT_FORWARD, &surface_in, &surface_out, n);
        cudaCheckError();
        measures[i] = stopTimer();
    }
    cuda_surface_shakedown(&in, &ref, &surface_in, &surface_out, &array_in, &array_out);
    return average_best(measures, NUM_PERFORMANCE);
}

// ---------------------------------------------
//
// Surface / Texture object
//
// Experimental so see if it scales up better then using global memory
//
// ---------------------------------------------

__device__ __inline__ void add_sub_mul(cpx *inL, cpx *inU, cuSurf out, int x_add, int y_add, int x_subm, int y_subm, cpx *W)
{
    float x = inL->x - inU->x;
    float y = inL->y - inU->y;
    cpx outL = { inL->x + inU->x, inL->y + inU->y };
    cpx outU = { (W->x * x) - (W->y * y), (W->y * x) + (W->x * y) };
    SURF2D_WRITE(outL, out, x_add, y_add);
    SURF2D_WRITE(outU, out, x_subm, y_subm);
}

__global__ void kernel_row(cuSurf in, cuSurf out, float angle, unsigned int lmask, int steps, int dist)
{
    int col_id = blockIdx.y * blockDim.x + threadIdx.x;
    int x = (col_id + (col_id & lmask));
    cpx w, in_lower, in_upper;
    SURF2D_READ(&in_lower, in, x, blockIdx.x);
    SURF2D_READ(&in_upper, in, x + dist, blockIdx.x);
    SIN_COS_F(angle * (((blockIdx.y * blockDim.x + threadIdx.x) << steps) & ((dist - 1) << steps)), &w.y, &w.x);
    add_sub_mul(&in_lower, &in_upper, out, x, blockIdx.x, x + dist, blockIdx.x, &w);
}

__global__ void kernel_col(cuSurf in, cuSurf out, float angle, unsigned int lmask, int steps, int dist)
{
    int row_id = blockIdx.y * blockDim.x + threadIdx.x;
    int y = (row_id + (row_id & lmask));
    cpx w, in_lower, in_upper;
    SURF2D_READ(&in_lower, in, blockIdx.x, y);
    SURF2D_READ(&in_upper, in, blockIdx.x, y + dist);
    SIN_COS_F(angle * (((blockIdx.y * blockDim.x + threadIdx.x) << steps) & ((dist - 1) << steps)), &w.y, &w.x);
    add_sub_mul(&in_lower, &in_upper, out, blockIdx.x, y, blockIdx.x, y + dist, &w);
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

__global__ void _kernelGPUS2DRowSurf(cuSurf in, cuSurf out, float angle, float local_angle, int steps_left, cpx scalar, int n_per_block)
{
    extern __shared__ cpx shared[];
    int in_high = (n_per_block >> 1) + threadIdx.x;
    int rowOffset = blockIdx.y * blockDim.x * 2;
    mem_gtos_row(threadIdx.x, in_high, rowOffset, shared, in);
    SYNC_THREADS;
    cuda_algorithm_local(shared, in_high, local_angle, steps_left);
    mem_stog_dbt_row(threadIdx.x, in_high, rowOffset, 32 - log2((int)gridDim.x), scalar, shared, out);
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

#define ROW_COL_KERNEL(rw, kr, kc) ((rw) ? (kr) : (kc))

__host__ void cuda_surface_fft_helper(fftDir dir, cuSurf *surface_in, cuSurf *surface_out, int row_wise, int n)
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
        ROW_COL_KERNEL(row_wise, kernel_row, kernel_col) KERNEL_ARGS2(blocks, threads)(*surface_in, *surface_out, global_angle, 0xFFFFFFFF << steps_left, steps, dist);
        cudaDeviceSynchronize();
        while (--steps_left > steps_gpu) {
            dist >>= 1;
            ++steps;
            ROW_COL_KERNEL(row_wise, kernel_row, kernel_col) KERNEL_ARGS2(blocks, threads)(*surface_out, *surface_out, global_angle, 0xFFFFFFFF << steps_left, steps, dist);
            cudaDeviceSynchronize();
        }
        cuda_surface_swap(surface_in, surface_out);
        ++steps_left;
        block_range_half = n_per_block;
    }
    // Calculate complete sequence in one launch and syncronize on GPU
    ROW_COL_KERNEL(row_wise, _kernelGPUS2DRowSurf, _kernelGPUS2DColSurf) KERNEL_ARGS3(blocks, threads, sizeof(cpx) * n_per_block) (*surface_in, *surface_out, global_angle, local_angle, steps_left, scaleCpx, block_range_half);
    cudaDeviceSynchronize();
}

__host__ void cuda_surface_fft(fftDir dir, cuSurf *surface_in, cuSurf *surface_out, int n)
{
    cuda_surface_fft_helper(dir, surface_in, surface_out, 1, n);
    cuda_surface_fft_helper(dir, surface_out, surface_in, 0, n);
    cuda_surface_swap(surface_in, surface_out);
}