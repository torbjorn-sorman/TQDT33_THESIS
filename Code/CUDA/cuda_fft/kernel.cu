#include <stdio.h>
#include <Windows.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include <cufft.h>

#include "definitions.cuh"
#include "fft_test.cuh"
#include "fft_helper.cuh"

#include "FFTConstantGeom.cuh"
#include "FFTRegular.cuh"
#include "FFTTobb.cuh"

#define NO_TESTS 32
#define MAX_LENGTH 2097152 / 2

/* Performance measure on Windows, result in micro seconds */

LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds, Frequency;
#define QPF QueryPerformanceFrequency
#define QPC QueryPerformanceCounter
#define START_TIME QPF(&Frequency); QPC(&StartingTime)
#define STOP_TIME(RES) QPC(&EndingTime); ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart; ElapsedMicroseconds.QuadPart *= 1000000; ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;(RES) = (double)ElapsedMicroseconds.QuadPart

cudaError_t FFT_CUDA(fftDirection direction, cuFloatComplex *in, cuFloatComplex *out, double measures[], int n);
cudaError_t FFT_ConstantGeometry(fftDirection direction, cuFloatComplex *in, cuFloatComplex *out, cuFloatComplex *W, double measures[], int n);
cudaError_t FFT_ConstantGeometry2(fftDirection direction, cuFloatComplex *in, cuFloatComplex *out, double measures[], int n);
cudaError_t FFT_Tobb(fftDirection direction, cuFloatComplex *in, cuFloatComplex *out, double measures[], int n);

unsigned int power(unsigned int base, int exp)
{
    if (exp == 0)
        return 1;
    unsigned int value = base;
    for (int i = 0; i < exp; ++i) {
        value *= base;
    }
    return value;
}

unsigned int power2(int exp)
{
    return power(2, exp);
}

int main()
{
    int n;
    double measures[20];
    cuFloatComplex *in, *out, *ref_in, *ref_out, *W;
    cudaError_t cudaStatus;
    
    printf("\tcuFFT\ttbFFT\ttbFFT\n");
    for (n = power2(2); n < power2(19); n *= 2) {
        in = get_seq(n, 1);
        ref_in = get_seq(n, in);
        out = get_seq(n);
        ref_out = get_seq(n);
        W = get_seq(n);
        
        // cuFFT
        cudaStatus = FFT_CUDA(FFT_FORWARD, ref_in, ref_out, measures, n);
        printf("%d:\t%.0f", n, avg(measures, 20));
        cudaStatus = FFT_CUDA(FFT_INVERSE, ref_out, ref_in, measures, n);
        
        cudaStatus = FFT_ConstantGeometry2(FFT_FORWARD, in, out, measures, n);
        printf("\t%.0f", avg(measures, 20));
        cudaStatus = FFT_ConstantGeometry2(FFT_INVERSE, out, in, measures, n);        
        if (checkError(in, ref_in, (float)n, n, 0) == 1) printf("!");

        free(in);
        in = get_seq(n, 1);
        cudaStatus = FFT_ConstantGeometry(FFT_FORWARD, in, out, W, measures, n);
        printf("\t%.0f", avg(measures, 20));
        cudaStatus = FFT_ConstantGeometry(FFT_INVERSE, out, in, W, measures, n);
        if (checkError(in, ref_in, (float)n, n, 0) == 1) printf("!");

        free(in);
        in = get_seq(n, 1);
        cudaStatus = FFT_Tobb(FFT_FORWARD, in, out, measures, n);
        printf("\t%.0f", avg(measures, 20));
        cudaStatus = FFT_Tobb(FFT_INVERSE, out, in, measures, n);
        if (checkError(in, ref_in, (float)n, n, 0) == 1) printf("!");
                
        printf("\n");
        free(in);
        free(out);
        free(W);
        free(ref_in);
        free(ref_out);
    }
    printf("\n");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        getchar();
        return 1;
    }

    getchar();
    return 0;
}

cudaError_t FFT_CUDA(fftDirection direction, cuFloatComplex *in, cuFloatComplex *out, double measures[], int n)
{
    cufftHandle plan;
    cuFloatComplex *dev_in;
    cuFloatComplex *dev_out;
    cudaMalloc((void**)&dev_in, sizeof(cuFloatComplex) * n);
    cudaMalloc((void**)&dev_out, sizeof(cuFloatComplex) * n);
    cudaMemcpy(dev_in, in, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cufftPlan1d(&plan, n, CUFFT_C2C, 1);        

    cufftExecC2C(plan, dev_in, dev_out, direction);
    cudaDeviceSynchronize();    
    
    cudaMemcpy(out, dev_out, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);    
    for (int i = 0; i < 20; ++i) {
        START_TIME;
        cufftExecC2C(plan, dev_in, dev_out, direction);
        cudaDeviceSynchronize();
        STOP_TIME(measures[i]);
    }
    cufftDestroy(plan);
    cudaFree(dev_in);
    cudaFree(dev_out);
    return cudaDeviceSynchronize();
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t FFT_ConstantGeometry(fftDirection direction, cuFloatComplex *in, cuFloatComplex *out, cuFloatComplex *W, double measures[], int n)
{
    unsigned int bufferswitch;
    cuFloatComplex *dev_in = 0;
    cuFloatComplex *dev_out = 0;
    cuFloatComplex *dev_W = 0;

    cudaError_t cudaStatus = cudaSetDevice(0);

    cudaStatus = cudaMalloc((void**)&dev_in, n * sizeof(cuFloatComplex));
    cudaStatus = cudaMalloc((void**)&dev_out, n * sizeof(cuFloatComplex));
    cudaStatus = cudaMalloc((void**)&dev_W, (n / 2) * sizeof(cuFloatComplex));
    cudaStatus = cudaMemcpy(dev_in, in, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    FFTConstGeom(direction, dev_in, dev_out, dev_W, &bufferswitch, n);
    cudaDeviceSynchronize();
        
    cudaStatus = cudaMemcpy(out, (bufferswitch == 1) ? dev_out : dev_in, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(in, (bufferswitch == 0) ? dev_out : dev_in, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(W, dev_W, (n / 2) * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 20; ++i) {
        START_TIME;
        FFTConstGeom(direction, dev_in, dev_out, dev_W, &bufferswitch, n);
        cudaDeviceSynchronize();
        STOP_TIME(measures[i]);
    }

    cudaFree(dev_in);
    cudaFree(dev_out);
    cudaFree(dev_W);

    return cudaDeviceSynchronize();
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t FFT_ConstantGeometry2(fftDirection direction, cuFloatComplex *in, cuFloatComplex *out, double measures[], int n)
{
    unsigned int bufferswitch;
    cuFloatComplex *dev_in = 0;
    cuFloatComplex *dev_out = 0;
    cudaError_t cudaStatus = cudaSetDevice(0);
    cudaStatus = cudaMalloc((void**)&dev_in, n * sizeof(cuFloatComplex));
    cudaStatus = cudaMalloc((void**)&dev_out, n * sizeof(cuFloatComplex));
    cudaStatus = cudaMemcpy(dev_in, in, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    FFTConstGeom2(direction, dev_in, dev_out, &bufferswitch, n);
    cudaDeviceSynchronize();
    cudaStatus = cudaMemcpy(out, (bufferswitch == 1) ? dev_out : dev_in, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(in, (bufferswitch == 0) ? dev_out : dev_in, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 20; ++i) {
        START_TIME;
        FFTConstGeom2(direction, dev_in, dev_out, &bufferswitch, n);
        cudaDeviceSynchronize();
        STOP_TIME(measures[i]);
    }
    cudaFree(dev_in);
    cudaFree(dev_out);
    return cudaDeviceSynchronize();
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t FFT_Tobb(fftDirection direction, cuFloatComplex *in, cuFloatComplex *out, double measures[], int n)
{
    unsigned int bufferswitch;
    cuFloatComplex *dev_in = 0;
    cuFloatComplex *dev_out = 0;
    cuFloatComplex *dev_W = 0;
    cudaError_t cudaStatus = cudaSetDevice(0);
    cudaStatus = cudaMalloc((void**)&dev_in, n * sizeof(cuFloatComplex));
    cudaStatus = cudaMalloc((void**)&dev_out, n * sizeof(cuFloatComplex));
    cudaStatus = cudaMalloc((void**)&dev_W, (n / 2) * sizeof(cuFloatComplex));
    cudaStatus = cudaMemcpy(dev_in, in, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    FFTTobb(direction, dev_in, dev_out, dev_W, &bufferswitch, n);
    cudaDeviceSynchronize();
    cudaStatus = cudaMemcpy(out, (bufferswitch == 1) ? dev_out : dev_in, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(in, (bufferswitch == 0) ? dev_out : dev_in, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 20; ++i) {
        START_TIME;
        FFTTobb(direction, dev_in, dev_out, dev_W, &bufferswitch, n);
        cudaDeviceSynchronize();
        STOP_TIME(measures[i]);
    }
    cudaFree(dev_in);
    cudaFree(dev_out);
    cudaFree(dev_W);
    return cudaDeviceSynchronize();
}

/*
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> >(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
*/
