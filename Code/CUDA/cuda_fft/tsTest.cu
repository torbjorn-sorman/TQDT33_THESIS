#include <Windows.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "math.h"

#include "tsTest.cuh"

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

__host__ void cudaCheckError(cudaError_t err)
{
    if (err) {
        printf("\n%s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        getchar();
        exit(err);
    }
}

__host__ void cudaCheckError()
{
    cudaCheckError(cudaGetLastError());
}

__host__ void fft2DSurfSetup(cpx **in, cpx **ref, size_t *size, char *image_name, int sinus, int n, cudaArray **cuInputArray, cudaArray **cuOutputArray, cuSurf *inputSurfObj, cuSurf *outputSurfObj)
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