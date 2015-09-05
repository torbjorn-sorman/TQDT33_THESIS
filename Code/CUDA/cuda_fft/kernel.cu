
#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"

#include <stdio.h>

#include "definitions.cuh"
#include "fft_test.cuh"
#include "fft_const_geom.cuh"
#include "fft_helper.cuh"

#define NO_TESTS 32
#define MAX_LENGTH 2097152 / 2

/* Performance measure on Windows, result in micro seconds */

LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds, Frequency;
#define QPF QueryPerformanceFrequency
#define QPC QueryPerformanceCounter
#define START_TIME QPF(&Frequency); QPC(&StartingTime)
#define STOP_TIME(RES) QPC(&EndingTime); ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart; ElapsedMicroseconds.QuadPart *= 1000000; ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;(RES) = (double)ElapsedMicroseconds.QuadPart

cudaError_t fftCuda(float direction, cpx *in, cpx *out, int n);

int main()
{
    int n;
    double measures[2];
    cpx *in, *out, *ref;
    cudaError_t cudaStatus;

    n = 8;
    in = get_seq(n, 1);
    ref = get_seq(n, in);
    out = get_seq(n);

    printf("Go go go!\n");
    getchar();

    printf("Forward\n");
    START_TIME;
    cudaStatus = fftCuda(-1.f, in, out, n);
    STOP_TIME(measures[0]);
    
    console_print(in, n);

    printf("Inverse\n");
    START_TIME;
    //cudaStatus = fftCuda(1.f, out, in, n);
    STOP_TIME(measures[1]);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        getchar();
        return 1;
    }

    printf("Happened: %f & %f\n", measures[0], measures[1]);
    checkError(in, ref, n, 1);

    free(in);
    free(out);
    free(ref);

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
cudaError_t fftCuda(float direction, cpx *in, cpx *out, int n)
{
    cpx *dev_in = 0;
    cpx *dev_out = 0;
    cpx *dev_W = 0;
    
    cudaError_t cudaStatus;
        
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    printf("Got device!\n");

    cudaStatus = cudaMalloc((void**)&dev_in, n * sizeof(cpx));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_out, n * sizeof(cpx));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_W, (n / 2) * sizeof(cpx));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    fft_const_geom(direction, dev_in, dev_out, dev_W, n);
        
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Last error: %s\n", cudaGetErrorString(cudaStatus));
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
    cudaStatus = cudaMemcpy(out, dev_out, n * sizeof(cpx), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_in);
    cudaFree(dev_out);
    cudaFree(dev_W);

    return cudaStatus;
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
