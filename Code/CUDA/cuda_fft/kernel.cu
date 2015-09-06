
#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"

#include <stdio.h>

#include "definitions.cuh"
#include "fft_test.cuh"
#include "fft_const_geo.cuh"
#include "fft_helper.cuh"

#define NO_TESTS 32
#define MAX_LENGTH 2097152 / 2

/* Performance measure on Windows, result in micro seconds */

LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds, Frequency;
#define QPF QueryPerformanceFrequency
#define QPC QueryPerformanceCounter
#define START_TIME QPF(&Frequency); QPC(&StartingTime)
#define STOP_TIME(RES) QPC(&EndingTime); ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart; ElapsedMicroseconds.QuadPart *= 1000000; ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;(RES) = (double)ElapsedMicroseconds.QuadPart

cudaError_t fftCuda(float direction, cpx *in, cpx *out, double measures[], int n);



int main()
{
    int n;
    double measures[20];
    cpx *in, *out, *ref;
    cudaError_t cudaStatus;
    
    for (n = 4; n < 256; n *= 2) {        
        in = get_seq(n, 1);
        ref = get_seq(n, in);
        out = get_seq(n);
        //printf("\nN: %d\n", n);
        cudaStatus = fftCuda(-1.f, in, out, measures, n);
        printf("%.0f\n", avg(measures, 20));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "fftCuda Forward failed!\n");
        cudaStatus = fftCuda(1.f, out, in, measures, n);
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "fftCuda Inverse failed!\n");
        checkError(in, ref, n, 1);
        free(in);
        free(out);
        free(ref);
    }

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

// Helper function for using CUDA to add vectors in parallel.
cudaError_t fftCuda(float direction, cpx *in, cpx *out, double measures[], int n)
{
    unsigned int bufferswitch;
    cpx *dev_in = 0;
    cpx *dev_out = 0;
    cpx *dev_W = 0;
    
    cudaError_t cudaStatus = cudaSetDevice(0);

    cudaStatus = cudaMalloc((void**)&dev_in, n * sizeof(cpx));
    cudaStatus = cudaMalloc((void**)&dev_out, n * sizeof(cpx));
    cudaStatus = cudaMalloc((void**)&dev_W, (n / 2) * sizeof(cpx));
    cudaStatus = cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    
    fft_const_geo(direction, dev_in, dev_out, dev_W, &bufferswitch, n);
    cudaDeviceSynchronize();

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Last error: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(out, (bufferswitch == 1) ? dev_out : dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy dev_out -> out failed!");
        goto Error;
    }

    for (int i = 0; i < 20; ++i) {
        START_TIME;
        fft_const_geo(direction, dev_in, dev_out, dev_W, &bufferswitch, n);
        cudaDeviceSynchronize();
        STOP_TIME(measures[i]);
    }

Error:
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
