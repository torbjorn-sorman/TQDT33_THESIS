#include "MyHelperCUDA.cuh"

__global__ void twiddle_factors(cpx *W, float angle, int n)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    SIN_COS_F(angle * i, &W[i].y, &W[i].x);
}

__global__ void bit_reverse(cpx *in, cpx *out, float scale, int lead)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int p = BIT_REVERSE(i, lead);
    out[p].x = in[i].x * scale;
    out[p].y = in[i].y * scale;
}

__global__ void bit_reverse(cpx *x, float dir, int lead, int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int p = BIT_REVERSE(i, lead);
    cpx tmp;
    if (i < p) {
        tmp = x[i];
        x[i] = x[p];
        x[p] = tmp;
    }
    if (dir > 0) {
        x[i].x = x[i].x / (float)n;
        x[i].y = x[i].y / (float)n;
    }
}

__global__ void _kernelTranspose(cpx *in, cpx *out, int n)
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

__global__ void _kernelTranspose(cuSurf in, cuSurf out, int n)
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

int checkValidConfig(int blocks, int n)
{
    if (blocks > NO_STREAMING_MULTIPROCESSORS) {
        switch (MAX_BLOCK_SIZE)
        {
        case 256:   return blocks <= 32;    // 2^14
        case 512:   return blocks <= 16;    // 2^14
        case 1024:  return blocks <= 4;     // 2^13
            // Default is a configurable limit, essentially blocksize limits the number of treads that can perform the synchronization.
        default:    return n <= MAX_BLOCK_SIZE * MAX_BLOCK_SIZE;
        }
    }
    return 1;
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

void set2DBlocksNThreads(dim3 *bFFT, dim3 *tFFT, dim3 *bTrans, dim3 *tTrans, int n)
{
    int n2 = n >> 1;
    (*bFFT).x = n;
    (*bFFT).z = (*tFFT).y = (*tFFT).z = (*bTrans).z = (*tTrans).z = 1;
    (*bTrans).x = (*bTrans).y = (n / TILE_DIM);
    (*tTrans).x = (*tTrans).y = THREAD_TILE_DIM;
    if (n2 > MAX_BLOCK_SIZE) {
        (*bFFT).y = n2 / MAX_BLOCK_SIZE;
        (*tFFT).x = MAX_BLOCK_SIZE;
    }
    else {
        (*bFFT).y = 1;
        (*tFFT).x = n2;
    }
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

cpx* read_image(char *name, int *n)
{
    image image;
    color_component *cp;
    FILE *fp;
    fopen_s(&fp, name, "rb");
    image = get_ppm(fp);
    if (!image || image->width != image->height)
        return NULL;

    int size = *n = image->width;
    cpx *seq = (cpx *)malloc(sizeof(cpx) * size * size);
    for (int y = 0; y < (int)image->height; ++y) {
        for (int x = 0; x < (int)image->width; ++x) {
            cp = GET_PIXEL(image, x, y);
            seq[y * size + x] = make_cuComplex((cp[0] + cp[1] + cp[2]) / (3.f * 255.f), 0.f);
        }
    }
    free_img(image);
    return seq;
}

void normalized_cpx_values(cpx* seq, int n, double *min_val, double *range, double *avg)
{
    double min_v = 99999999999;
    double max_v = -99999999999;
    double sum_v = 0.0;
    double tmp = 0.0;
    for (int i = 0; i < n; ++i) {
        tmp = cuCabsf(seq[i]);
        min_v = min(min_v, tmp);
        max_v = max(max_v, tmp);
        sum_v += tmp;
    }
    *min_val = min_v;
    *range = max_v - min_v;
    *avg = sum_v / (double)n;
}

void write_normalized_image(char *name, cpx* seq, int n)
{
    image image;
    FILE  *fp;
    double minVal, range, avg, mag, val;
    normalized_cpx_values(seq, n, &minVal, &range, &avg);
    double avg_pos = 0.4;
    double scale = tan(avg_pos * (M_PI / 2)) / ((avg - minVal) / range);
    image = alloc_img(n, n);
    fopen_s(&fp, name, "wb");
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            mag = cuCabsf(seq[y * n + x]);
            val = ((mag - minVal) / range);
            val = (atan(val * scale) / (M_PI / 2.0)) * 255.0;
            color_component col = (unsigned char)(val > 255.0 ? 255 : val);
            put_pixel_unsafe(image, x, y, col, col, col);
        }
    }
    output_ppm(fp, image);
    fclose(fp);
    free_img(image);
}

void normalized_image(cpx* seq, int n)
{
    double minVal, range, avg, mag, val;
    normalized_cpx_values(seq, n, &minVal, &range, &avg);
    double avg_pos = 0.8;
    double scale = tan(avg_pos * (M_PI / 2)) / ((avg - minVal) / range);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            mag = cuCabsf(seq[y * n + x]);
            val = ((mag - minVal) / range);
            val = (atan(val * scale) / (M_PI / 2.0));
            seq[y * n + x] = make_cuFloatComplex((float)(val > 1.0 ? 1 : val), 0.f);
        }
    }
}

void write_image(char *name, char *type, cpx* seq, int n)
{
    image image;
    FILE  *fp;
    image = alloc_img(n, n);
    char filename[50];
    sprintf_s(filename, 50, "out/img/%s_%u_%s.ppm", name, n, type);
    fopen_s(&fp, filename, "wb");
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            color_component val = (unsigned char)((seq[y * n + x].x) * 255.f);
            put_pixel_unsafe(image, x, y, val, val, val);
        }
    }
    output_ppm(fp, image);
    fclose(fp);
    free_img(image);
}

void clear_image(cpx* seq, int n)
{
    for (int i = 0; i < n; ++i)
        seq[i] = make_cuFloatComplex(1.f, 1.f);
}

void cpPixel(int px, int px2, cpx *in, cpx *out)
{
    int p, p2;
    p = px * 3;
    p2 = px2 * 3;
    out[p] = in[p2];
    out[p + 1] = in[p2 + 1];
    out[p + 2] = in[p2 + 2];
}

cpx* fftShift(cpx *seq, int n)
{
    cpx *out = (cpx *)malloc(sizeof(cpx)*n*n);
    int px1, px2;
    int n2 = n / 2;
    for (int y = 0; y < n2; ++y) {
        for (int x = 0; x < n2; ++x) {
            px1 = y * n + x;
            px2 = (y + n2) * n + (x + n2);
            out[px1] = seq[px2];
            out[px2] = seq[px1];
        }
    }
    for (int y = 0; y < n2; ++y) {
        for (int x = n2; x < n; ++x) {
            px1 = y * n + x;
            px2 = (y + n2) * n + (x - n2);
            out[px1] = seq[px2];
            out[px2] = seq[px1];
        }
    }
    return out;
}

#define ERROR_MARGIN 0.0001

static LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds, Frequency;

void startTimer()
{
    QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&StartingTime);
}

double stopTimer()
{
    QueryPerformanceCounter(&EndingTime);
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
    return (double)ElapsedMicroseconds.QuadPart;
}

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

unsigned int power(unsigned int base, unsigned int exp)
{
    if (exp == 0)
        return 1;
    unsigned int value = base;
    for (unsigned int i = 1; i < exp; ++i) {
        value *= base;
    }
    return value;
}

unsigned int power2(unsigned int exp)
{
    return power(2, exp);
}

int checkError(cpx *seq, cpx *ref, float refScale, int n, int print)
{
    int j;
    double re, im, i_val, r_val;
    re = im = 0.0;
    for (j = 0; j < n; ++j) {
        r_val = abs(refScale * seq[j].x - ref[j].x);
        i_val = abs(refScale * seq[j].y - ref[j].y);
        re = re > r_val ? re : r_val;
        im = im > i_val ? im : i_val;
    }
    if (print == 1) printf("Error\tre(e): %f\t im(e): %f\t@%u\n", re, im, n);
    return re > ERROR_MARGIN || im > ERROR_MARGIN;
}

int checkError(cpx *seq, cpx *ref, int n, int print)
{
    return checkError(seq, ref, 1.f, n, print);
}

int checkError(cpx *seq, cpx *ref, int n)
{
    return checkError(seq, ref, n, 0);
}

cpx *get_seq(int n)
{
    return get_seq(n, 0);
}

cpx *get_seq(int n, int sinus)
{
    int i;
    cpx *seq;
    seq = (cpx *)malloc(sizeof(cpx) * n);
    for (i = 0; i < n; ++i) {
        seq[i].x = sinus == 0 ? 0.f : (float)sin(M_2_PI * (((double)i) / n));
        seq[i].y = 0.f;
    }
    return seq;
}

cpx *get_seq(int n, cpx *src)
{
    int i;
    cpx *seq;
    seq = (cpx *)malloc(sizeof(cpx) * n);
    for (i = 0; i < n; ++i) {
        seq[i].x = src[i].x;
        seq[i].y = src[i].y;
    }
    return seq;
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

int cmp(const void *x, const void *y)
{
    double xx = *(double*)x, yy = *(double*)y;
    if (xx < yy) return -1;
    if (xx > yy) return  1;
    return 0;
}

double avg(double m[], int n)
{
    int i, cnt, end;
    double sum;
    qsort(m, n, sizeof(double), cmp);
    sum = 0.0;
    cnt = 0;
    end = n < 5 ? n - 1 : 5;
    for (i = 0; i < end; ++i) {
        sum += m[i];
        ++cnt;
    }
    return (sum / (double)cnt);
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
    int result;
    _cudaFree(dev_in, dev_out, dev_W);
    cudaDeviceSynchronize();
    if (in == NULL && ref == NULL && out == NULL)
        return 0;
    result = checkError(*in, *ref, n);
    _fftFreeSeq(in, out, ref);
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return result;
}

void fft2DSetup(cpx **in, cpx **ref, cpx **dev_i, cpx **dev_o, size_t *size, char *image_name, int sinus, int n)
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

void fft2DSurfSetup(cpx **in, cpx **ref, size_t *size, char *image_name, int sinus, int n, cudaArray **cuInputArray, cudaArray **cuOutputArray, cuSurf *inputSurfObj, cuSurf *outputSurfObj)
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