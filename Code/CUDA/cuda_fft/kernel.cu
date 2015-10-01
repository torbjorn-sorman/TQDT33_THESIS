
//#define PROFILER
//#define IMAGE_TEST

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#ifndef PROFILER
#include <cufft.h>
#endif

#include "tsDefinitions.cuh"
#include "tsTest.cuh"

#include "tsCombine.cuh"
#include "tsCombineGPUSync.cuh"
#include "tsCombineGPUSyncTex.cuh"

#ifndef PROFILER

__host__ double cuFFT_Performance(int n);
__host__ double cuFFT_2D_Performance(int n);
__host__ void toFile(const char *name, const double m[], int ms);

// Print device properties
void printDevProp(cudaDeviceProp devProp)
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

#endif

#define RUNS 16

int main()
{    
    //cudaDeviceProp prop;
    //cudaGetDeviceProperties(&prop, 0);
    //printDevProp(prop);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        
#if defined(PROFILER)
    int start = 2;
    int end = start + RUNS;
    for (unsigned int n = power2(start); n < power2(end); n *= 2)
        tsCombine_Performance(n);
#elif defined(IMAGE_TEST)
    printf("\n2D validation & performance test!\n");
    printf("\tn\tcuFFT\tMy\tMy Surf\t");
    for (unsigned int n = TILE_DIM / 2; n <= 4096; n *= 2) {
        printf("\n\t%d:", n);
        printf("\t%.0f", cuFFT_2D_Performance(n));

        if (n < 4096) {            
            tsCombineGPUSync2D_Test(n);
            tsCombineGPUSyncTex2D_Test(n);
        }
    }
    getchar();
#else
    int start = 2;
    int end = start + RUNS;
    int index = 0;
    double cuFFTm[RUNS];
    double combineFFTm[RUNS];
    printf("\n\t\tcuFFT\tComb");
    printf("\n");
    for (unsigned int n = power2(start); n < power2(end); n *= 2) {        
        printf("\n%d:", n);
        
        char *fmt = n > 1000000 ? "\t%.0f" : "\t\t%.0f";
        
        // cuFFT
        printf(fmt, cuFFTm[index] = cuFFT_Performance(n));
                
        // Combine
        printf("\t%.0f", combineFFTm[index] = tsCombine_Performance(n));
        if (tsCombine_Validate(n) == 0) printf("!");
        
        ++index;
    }
    printf("\n\n");
    toFile("cuFFT", cuFFTm, RUNS);
    toFile("Block Combine CPU Sync", combineFFTm, RUNS);
    
    printf("\nDone...");
    getchar();
#endif
    return 0;
}

#ifndef PROFILER

__host__ double cuFFT_Performance(int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *dev_in,*dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, NULL, NULL, NULL);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, 1);
    for (int i = 0; i < 20; ++i) {
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
    for (int i = 0; i < 20; ++i) {
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

    printf("File '%s' written.\n", filename);
    fclose(f);
}

#endif