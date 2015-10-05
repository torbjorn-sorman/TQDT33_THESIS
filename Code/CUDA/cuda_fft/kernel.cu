
//#define PROFILER
#define IMAGE_TEST
#define RUN_CUFFT

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cuda_profiler_api.h>
#include <cufft.h>

#include "tsDefinitions.cuh"
#include "tsTest.cuh"

#include "tsCombine.cuh"
#include "tsCombine2.cuh"
#include "tsCombineGPUSync.cuh"
#include "tsCombineGPUSyncTex.cuh"

__host__ double cuFFT_Performance(int n);
__host__ double cuFFT_2D_Performance(int n);
__host__ void printDevProp(cudaDeviceProp devProp);
__host__ void toFile(const char *name, const double m[], int ms);

#define RUNS 10

int main()
{
    //cudaDeviceProp prop;
    //cudaGetDeviceProperties(&prop, 0);
    //printDevProp(prop);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    double measures[RUNS];
    double measures_cuFFT[RUNS];
    int measureIndex = 0;
    int start = 2;
    int end = start + RUNS;

#if defined(PROFILER)
    cudaProfilerStart();
    for (unsigned int n = power2(start); n < power2(end); n *= 2) {
#ifdef RUN_CUFFT
        measures_cuFFT[measureIndex] = cuFFT_Performance(n);
#endif
        measures[measureIndex] = tsCombine_Performance(n);
        measureIndex++;
    }
    cudaProfilerStop();
#ifdef RUN_CUFFT
    toFile("cuFFT", measures_cuFFT, RUNS);
#endif
    toFile("Combine CPU and GPU Sync", measures, RUNS);
#elif defined(IMAGE_TEST)
    printf("\n2D validation & performance test!\nn\t");
#ifdef RUN_CUFFT
    printf("\tcuFFT");
#endif
    printf("\tMy");
    for (unsigned int n = TILE_DIM / 2; n <= power2(end); n *= 2) {
        printf("\n%d:", n);
        char *fmt = (n > 1000000 ? "\t%.0f" : "\t\t%.0f");
#ifdef RUN_CUFFT
        printf(fmt, measures_cuFFT[measureIndex] = cuFFT_2D_Performance(n));
        printf("\t%.0f", measures[measureIndex] = tsCombine2D_Performance(n));
        if (tsCombine2D_Validate(n) == 0) printf("!");
        /*
        if (n < 4096) {
        tsCombineGPUSync2D_Test(n);
        tsCombineGPUSyncTex2D_Test(n);
        }
        */
#else
        printf(fmt, measures[measureIndex] = tsCombine2D_Performance(n));
        if (tsCombine2D_Validate(n) == 0) printf("!");
#endif
        ++measureIndex;
    }
    printf("\n\n");
#ifdef RUN_CUFFT
    toFile("cuFFT 2D", measures_cuFFT, RUNS);
#endif
    toFile("Combine 2D CPU and GPU Sync", measures, RUNS);
    getchar();
#else
    printf("\n\t");
#ifdef RUN_CUFFT
    printf("\tcuFFT");
#endif
    printf("\tComb");
    printf("\n");

    for (unsigned int n = power2(start); n < power2(end); n *= 2) {

        printf("\n%d:", n);
        char *fmt = n > 1000000 ? "\t%.0f" : "\t\t%.0f";
#ifdef RUN_CUFFT
        printf(fmt, measures_cuFFT[measureIndex] = cuFFT_Performance(n));
        printf("\t%.0f", measures[measureIndex] = tsCombine_Performance(n));
        if (tsCombine_Validate(n) == 0) printf("!");
#else
        printf(fmt, measures[measureIndex] = tsCombine_Performance(n));
        if (tsCombine_Validate(n) == 0) printf("!");
#endif
        // Combine

        ++measureIndex;
    }

    printf("\n\n");
#ifdef RUN_CUFFT
    toFile("cuFFT", measures_cuFFT, RUNS);
#endif
    toFile("Combine CPU and GPU Sync", measures, RUNS);
    printf("\nDone...");
    getchar();
#endif
    return 0;
}

__host__ double cuFFT_Performance(int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, NULL, NULL, NULL);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, 1);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        cufftExecC2C(plan, dev_in, dev_out, CUFFT_FORWARD);
        cudaDeviceSynchronize();
        measures[i] = stopTimer();
    }
    cufftDestroy(plan);
    fftResultAndFree(n, &dev_in, &dev_out, NULL, NULL, NULL, NULL);
    return avg(measures, NUM_PERFORMANCE);
}

__host__ double cuFFT_2D_Performance(int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *dev_in, *dev_out;
    fftMalloc(n * n, &dev_in, &dev_out, NULL, NULL, NULL, NULL);
    cufftHandle plan;
    cufftPlan2d(&plan, n, n, CUFFT_C2C);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        cufftExecC2C(plan, dev_in, dev_out, CUFFT_FORWARD);
        cudaDeviceSynchronize();
        measures[i] = stopTimer();
    }
    cufftDestroy(plan);
    fftResultAndFree(n, &dev_in, &dev_out, NULL, NULL, NULL, NULL);
    return avg(measures, NUM_PERFORMANCE);
}

__host__ void toFile(const char *name, const double m[], int ms)
{
    char filename[64] = "";
    FILE *f;
    strcat_s(filename, "out/");
    strcat_s(filename, name);
    strcat_s(filename, ".txt");
    fopen_s(&f, filename, "w");
    for (int i = 0; i < ms; ++i)
        fprintf_s(f, "%0.f\n", m[i]);
    fclose(f);
#ifdef PROFILER
    printf("\nWrote '%s'\n", filename);
#endif
}

__host__ void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n", devProp.major);
    printf("Minor revision number:         %d\n", devProp.minor);
    printf("Name:                          %s\n", devProp.name);
    printf("Total global memory:           %u\n", devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n", devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n", devProp.regsPerBlock);
    printf("Warp size:                     %d\n", devProp.warpSize);
    printf("Maximum memory pitch:          %u\n", devProp.memPitch);
    printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n", devProp.clockRate);
    printf("Total constant memory:         %u\n", devProp.totalConstMem);
    printf("Texture alignment:             %u\n", devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}