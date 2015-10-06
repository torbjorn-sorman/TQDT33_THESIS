#include "tsCombineGPUSyncTex.cuh"

__global__ void _kernelGPUSyncTex2DRow(cuSurf surfIn, cuSurf surfOut, float angle, float bAngle, int depth, cpx scale, int n);
__global__ void _kernelGPUSyncTex2DCol(cuSurf surfIn, cuSurf surfOut, float angle, float bAngle, int depth, cpx scale, int n);

__host__ void testTex2DShakedown(cpx **in, cpx **ref, cuSurf *sObjIn, cuSurf *sObjOut, cudaArray **cuAIn, cudaArray **cuAOut)
{
    free(*in);
    free(*ref);
    cudaDestroySurfaceObject(*sObjIn);
    cudaDestroySurfaceObject(*sObjOut);
    cudaFreeArray(*cuAIn);
    cudaFreeArray(*cuAOut);
}

__host__ void testTex2DRun(fftDir dir, cpx *in, cudaArray *dev, cuSurf surfIn, cuSurf surfOut, char *image_name, char *type, size_t size, int write, int norm, int n)
{
    tsCombineGPUSyncTex2D(dir, surfIn, surfOut, n);
    if (write) {
        cudaMemcpyFromArray(in, dev, 0, 0, size, cudaMemcpyDeviceToHost);
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

__host__ int testTex2DCompare(cpx *in, cpx *ref, cudaArray *dev, size_t size, int len)
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

__host__ int tsCombineGPUSyncTex2D_Test(int n)
{
    double measures[NUM_PERFORMANCE];
    char *image_name = "shore";
    cpx *in, *ref;
    size_t size;
    cudaArray *cuInputArray, *cuOutputArray;
    cuSurf inputSurfObj, outputSurfObj;
    fft2DSurfSetup(&in, &ref, &size, image_name, NO, n, &cuInputArray, &cuOutputArray, &inputSurfObj, &outputSurfObj);

    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        tsCombineGPUSyncTex2D(FFT_FORWARD, inputSurfObj, outputSurfObj, n);
        measures[i] = stopTimer();
    }
    printf("\t(%.0f)", avg(measures, NUM_PERFORMANCE));

    cudaMemcpyToArray(cuInputArray, 0, 0, in, size, cudaMemcpyHostToDevice);
    testTex2DRun(FFT_FORWARD, in, cuInputArray, inputSurfObj, outputSurfObj, image_name, "frequency-domain", size, YES, YES, n);
    testTex2DRun(FFT_INVERSE, in, cuInputArray, outputSurfObj, inputSurfObj, image_name, "spatial-domain", size, YES, NO, n);
    int res = testTex2DCompare(in, ref, cuInputArray, size, n * n);
    testTex2DShakedown(&in, &ref, &inputSurfObj, &outputSurfObj, &cuInputArray, &cuOutputArray);

    if (res != 1)
        printf("!");
    return res;
}

// Fast in block 2D FFT when (partial) list fits shared memory combined with a general algorithm using global memory. Partially CPU synced and partially GPU-intra-block syncronized.
__host__ void tsCombineGPUSyncTex2D(fftDir dir, cuSurf surfIn, cuSurf surfOut, int n)
{
    dim3 threads, blocks, threads_trans, blocks_trans;
    set2DBlocksNThreads(&blocks, &threads, &blocks_trans, &threads_trans, n);

    cpx scale = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
    int nBlock = n / blocks.y;
    float w_angle = dir * (M_2_PI / n);
    float w_bangle = dir * (M_2_PI / nBlock);

    _kernelGPUSyncTex2DRow KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (surfIn, surfOut, w_angle, w_bangle, log2_32(n), scale, n);
    cudaDeviceSynchronize();
    _kernelGPUSyncTex2DCol KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (surfOut, surfIn, w_angle, w_bangle, log2_32(n), scale, n);
    cudaDeviceSynchronize();
}

__device__ static __inline__ void algorithm_partial(cpx *shared, int in_high, float angle, int bit)
{
    cpx w, in_lower, in_upper;
    int i = (threadIdx.x << 1);
    int ii = i + 1;
    for (int steps = 0; steps < bit; ++steps) {
        in_lower = shared[threadIdx.x];
        in_upper = shared[in_high];
        SYNC_THREADS;
        SIN_COS_F(angle * ((threadIdx.x & (0xFFFFFFFF << steps))), &w.y, &w.x);
        cpx_add_sub_mul(&(shared[i]), &(shared[ii]), in_lower, in_upper, w);
        SYNC_THREADS;
    }
}

__global__ void _kernelGPUSyncTex2DRow(cuSurf surfIn, cuSurf surfOut, float angle, float bAngle, int depth, cpx scale, int n)
{
    extern __shared__ cpx shared[];
    int bit = depth;
    int in_high = n >> 1;
    int offset = blockIdx.y * blockDim.x * 2;
    in_high += threadIdx.x;
    mem_gtos_row(threadIdx.x, in_high, offset, shared, surfIn);
    SYNC_THREADS;
    algorithm_partial(shared, in_high, bAngle, bit);
    mem_stog_db_row(threadIdx.x, in_high, offset, 32 - depth, scale, shared, surfOut);
}

__global__ void _kernelGPUSyncTex2DCol(cuSurf surfIn, cuSurf surfOut, float angle, float bAngle, int depth, cpx scale, int n)
{
    extern __shared__ cpx shared[];
    int bit = depth;
    int in_high = n >> 1;
    int offset = blockIdx.y * blockDim.x * 2;
    in_high += threadIdx.x;
    mem_gtos_col(threadIdx.x, in_high, offset, shared, surfIn);
    SYNC_THREADS;
    algorithm_partial(shared, in_high, bAngle, bit);
    mem_stog_db_col(threadIdx.x, in_high, offset, 32 - depth, scale, shared, surfOut);
}