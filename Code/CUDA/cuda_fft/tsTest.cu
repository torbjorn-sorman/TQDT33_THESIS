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
void console_print(cpx *seq, cInt n)
{
    for (int i = 0; i < n; ++i) printf("%f\t%f\n", seq[i].x, seq[i].y);
}

void console_print_cpx_img(cpx *seq, cInt n)
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

unsigned int power(cUInt base, cUInt exp)
{
    if (exp == 0)
        return 1;
    unsigned int value = base;
    for (unsigned int i = 1; i < exp; ++i) {
        value *= base;
    }
    return value;
}

unsigned int power2(cUInt exp)
{
    return power(2, exp);
}

int checkError(cpx *seq, cpx *ref, float refScale, cInt n, int print)
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

int checkError(cpx *seq, cpx *ref, cInt n, int print)
{
    return checkError(seq, ref, 1.f, n, print);
}

int checkError(cpx *seq, cpx *ref, cInt n)
{
    return checkError(seq, ref, n, 0);
}

cpx *get_seq(cInt n)
{
    return get_seq(n, 0);
}

cpx *get_seq(cInt n, cInt sinus)
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

cpx *get_seq(cInt n, cpx *src)
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

cpx *get_sin_img(cInt n)
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