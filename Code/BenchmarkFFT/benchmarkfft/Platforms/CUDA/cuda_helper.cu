#include "cuda_helper.cuh"

__global__ void kernelTranspose(cpx *in, cpx *out, int n)
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

__global__ void kernelTranspose(cuSurf in, cuSurf out, int n)
{
    // Banking issues when TILE_DIM % WARP_SIZE == 0, current WARP_SIZE == 32
    __shared__ cpx tile[TILE_DIM][TILE_DIM + 1];

    // Write to shared from Global (in)
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += THREAD_TILE_DIM)
        for (int i = 0; i < TILE_DIM; i += THREAD_TILE_DIM)
            SURF2D_READ(&(tile[threadIdx.y + j][threadIdx.x + i]), in, x + i, y + j);
    //tile[threadIdx.y + j][threadIdx.x + i] = in[(y + j) * n + (x + i)];

    SYNC_THREADS;
    // Write to global
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += THREAD_TILE_DIM)
        for (int i = 0; i < TILE_DIM; i += THREAD_TILE_DIM)
            SURF2D_WRITE(tile[threadIdx.x + i][threadIdx.y + j], out, x + i, y + j);
    //out[(y + j) * n + (x + i)] = tile[threadIdx.x + i][threadIdx.y + j];
}

void set_block_and_threads(int *numBlocks, int *threadsPerBlock, int size)
{
    if (size > MAX_BLOCK_SIZE) {
        *numBlocks = size / MAX_BLOCK_SIZE;
        *threadsPerBlock = MAX_BLOCK_SIZE;
    }
    else {
        *numBlocks = 1;
        *threadsPerBlock = size;
    }
}

void set_block_and_threads2D(dim3 *numBlocks, int *threadsPerBlock, int n)
{
    numBlocks->x = n;
    int n2 = n >> 1;
    if (n2 > MAX_BLOCK_SIZE) {
        numBlocks->y = n2 / MAX_BLOCK_SIZE;
        *threadsPerBlock = MAX_BLOCK_SIZE;
    }
    else {
        numBlocks->y = 1;
        *threadsPerBlock = n2;
    }
}

void set_block_and_threads_transpose(dim3 *bTrans, dim3 *tTrans, int n)
{
    bTrans->z = tTrans->z = 1;
    bTrans->x = bTrans->y = (n / TILE_DIM);
    tTrans->x = tTrans->y = THREAD_TILE_DIM;
}

void checkCudaError(char *msg)
{
    cudaError_t e;
    if (e = cudaGetLastError()) printf("%s:\n%s: %s\n", msg, cudaGetErrorName(e), cudaGetErrorString(e));
}

void checkCudaError()
{
    cudaError_t e;
    if (e = cudaGetLastError()) printf("%s: %s\n", cudaGetErrorName(e), cudaGetErrorString(e));
}

#define ERROR_MARGIN 0.0001

// Useful functions for debugging
void console_print(cpx *seq, int n)
{
    for (int i = 0; i < n; ++i) printf("%f\t%f\n", seq[i].x, seq[i].y);
}

void console_print_cpx_img(cpx *seq, int n)
{
    printf("\n");
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            printf("%.2f\t", seq[y * n + x].x);
        }
        printf("\n");
    }
    printf("\n");
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            printf("%.2f\t", seq[y * n + x].y);
        }
        printf("\n");
    }
}

cpx *get_sin_img(int n)
{
    cpx *seq;
    seq = (cpx *)malloc(sizeof(cpx) * n * n);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x)
            seq[y * n + x] = make_cuFloatComplex((float)sin(M_2_PI * (((double)x) / n)), 0.f);
    return seq;
}

void _cudaMalloc(int n, cpx **dev_in, cpx **dev_out, cpx **dev_W)
{
    *dev_in = 0;
    *dev_out = 0;
    cudaMalloc((void**)dev_in, n * sizeof(cpx));
    cudaMalloc((void**)dev_out, n * sizeof(cpx));
    if (dev_W != NULL) {
        *dev_W = 0;
        cudaMalloc((void**)dev_W, (n / 2) * sizeof(cpx));
    }
}

void _fftTestSeq(int n, cpx **in, cpx **ref, cpx **out)
{
    *in = get_seq(n, 1);
    *ref = get_seq(n, *in);
    *out = get_seq(n);
}

void fftMalloc(int n, cpx **dev_in, cpx **dev_out, cpx **dev_W, cpx **in, cpx **ref, cpx **out)
{
    _cudaMalloc(n, dev_in, dev_out, dev_W);
    if (in == NULL && ref == NULL && out == NULL)
        return;
    _fftTestSeq(n, in, ref, out);
}

void _cudaFree(cpx **dev_in, cpx **dev_out, cpx **dev_W)
{
    cudaFree(*dev_in);
    cudaFree(*dev_out);
    if (dev_W != NULL) cudaFree(*dev_W);
}

void _fftFreeSeq(cpx **in, cpx **ref, cpx **out)
{
    free(*in);
    free(*ref);
    free(*out);
}

int fftResultAndFree(int n, cpx **dev_in, cpx **dev_out, cpx **dev_W, cpx **in, cpx **ref, cpx **out)
{
    _cudaFree(dev_in, dev_out, dev_W);
    cudaDeviceSynchronize();
    if (in == NULL && ref == NULL && out == NULL)
        return 0;
    double diff = diff_seq(*in, *ref, n);
    _fftFreeSeq(in, out, ref);
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return diff > ERROR_MARGIN;
}

void fft2DSetup(cpx **in, cpx **ref, cpx **dev_i, cpx **dev_o, size_t *size, int n)
{
    char input_file[40];
    sprintf_s(input_file, 40, "Images/%u.ppm", n);
    int sz;
    *in = (cpx *)malloc(sizeof(cpx) * n * n); 
    read_image(*in, input_file, &sz);
    *ref = (cpx *)malloc(sizeof(cpx) * n * n);
    memcpy(*ref, *in, sizeof(cpx) * n * n);
    *size = n * n * sizeof(cpx);
    cudaMalloc((void**)dev_i, *size);
    cudaMalloc((void**)dev_o, *size);
    cudaMemcpy(*dev_i, *in, *size, cudaMemcpyHostToDevice);
}

void fft2DShakedown(cpx **in, cpx **ref, cpx **dev_i, cpx **dev_o)
{    
    free(*in);
    free(*ref);
    cudaFree(*dev_i);
    cudaFree(*dev_o);
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
}

int fft2DCompare(cpx *in, cpx *ref, cpx *dev, size_t size, int len)
{
    cudaMemcpy(in, dev, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < len; ++i) {
        if (cuCabsf(cuCsubf(in[i], ref[i])) > 0.0001) {
            return 0;
        }
    }
    return 1;
}

#define max(a, b) ((a) > (b) ? (a) : (b))

int fft2DCompare(cpx *in, cpx *ref, cpx *dev, size_t size, int len, double *relDiff)
{
    double mDiff = 0.0;
    double mVal = -1;
    cudaMemcpy(in, dev, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < len; ++i) {
        mVal = max(mVal, max(cuCabsf(in[i]), cuCabsf(ref[i])));
        double tmp = cuCabsf(cuCsubf(in[i], ref[i]));
        mDiff = tmp > mDiff ? tmp : mDiff;
    }
    *relDiff = (mDiff / mVal);
    return *relDiff < 0.00001;
}

void cudaCheckError(cudaError_t err)
{
    if (err) {
        printf("\n%s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        getchar();
        exit(err);
    }
}

void cudaCheckError()
{
    cudaCheckError(cudaGetLastError());
}

void fft2DSurfSetup(cpx **in, cpx **ref, size_t *size, int sinus, int n, cudaArray **cuInputArray, cudaArray **cuOutputArray, cuSurf *inputSurfObj, cuSurf *outputSurfObj)
{
    if (sinus) {
        *in = get_sin_img(n);
        *ref = get_sin_img(n);
    }
    else {
        char input_file[40];
        sprintf_s(input_file, 40, "Images/%u.ppm", n);
        int sz;
        *in = (cpx *)malloc(sizeof(cpx) * n * n);
        read_image(*in, input_file, &sz);
        *ref = (cpx *)malloc(sizeof(cpx) * n * n);
        memcpy(*ref, *in, sizeof(cpx) * n * n);
    }
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