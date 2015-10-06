#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cuda_profiler_api.h>
#include <cufft.h>
#include <vector>

#include "definitions.h"
#include "tsTest.cuh"
#include "getopt.h"
#include "MyCUDA.h"
#include "tsCombine.cuh"

__host__ double cuFFT_Performance(int n);
__host__ double cuFFT_2D_Performance(int n);
__host__ int cuFFT_2D_Compare(int n, double *diff);
__host__ void printDevProp(cudaDeviceProp devProp);
__host__ void toFile(std::string name, std::vector<double> results, int ms);

//#define PROFILER
#define IMAGE_TEST
#define RUN_CUFFT

//#define START_ELEM 7
#define RUNS 9

std::vector<Platform> getPlatforms(benchmarkArgument *args)
{
    std::vector<Platform> platforms;
    if (args->platform_cuda)
        platforms.insert(platforms.begin(), MyCUDA(args->dimensions));
    return platforms;
}

int main(int argc, const char* argv[])
{
    benchmarkArgument args;

    if (!parseArguments(&args, argc, argv))
        return 0;

    if (args.show_cuprop) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printDevProp(prop);
    }

    if (args.test_platform) {
        int measureIndex = 0;
        cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        std::vector<Platform> platforms = getPlatforms(&args);
        if (args.profiler) cudaProfilerStart();        
        for (int n = power2(args.start); n < power2(args.end); ++n, ++measureIndex) {
            for (Platform platform : platforms) {
                platform.performance(n);
            }
        }
        if (args.profiler) cudaProfilerStop();
        if (args.display) {
            std::cout << "n\t";
            for (Platform platform : platforms)
                std::cout << platform.name << "\t";            
            std::cout << "\n";
            char *fmt = "%.0f\t";
            std::cout << std::setprecision(0);
            for (int i = 0; i < measureIndex; ++i) {
                std::cout << power2(i + args.start) << "\t";
                for (Platform platform : platforms)
                    std::cout << platform.results.at(i) << (args.validate == 1 ? platform.validate(power2(args.start + i)) : "") << "\t";
                std::cout << "\n";
            }
        }
        if (args.write_file)
            for (Platform platform : platforms)
                toFile(platform.name, platform.results, args.number_of_lengths);
    }


    double measures[RUNS];
    double measures_surf[RUNS];
    double measures_cuFFT[RUNS];
    int measureIndex = 0;
#ifdef START_ELEM
    int start = START_ELEM;
    int end = START_ELEM + RUNS - 1;
#else
    int start = log2_32(TILE_DIM >> 1);
    int end = log2_32(TILE_DIM >> 1) + RUNS - 1;
#endif

#if defined(PROFILER)
    cudaProfilerStart();
    for (unsigned int n = power2(start); n <= power2(end); n *= 2) {
#ifndef IMAGE_TEST
#ifdef RUN_CUFFT
        measures_cuFFT[measureIndex] = cuFFT_Performance(n);
#endif
        measures[measureIndex] = tsCombine_Performance(n);            
#else
#ifdef RUN_CUFFT
        measures_cuFFT[measureIndex] = cuFFT_2D_Performance(n);
#endif
        measures[measureIndex] = tsCombine2D_Performance(n);            
#endif
        measureIndex++;
    }
    cudaProfilerStop();
#ifdef RUN_CUFFT
    toFile("cuFFT", measures_cuFFT, RUNS);
#endif
    toFile("Combine CPU and GPU Sync", measures, RUNS);
#elif defined(IMAGE_TEST)
    printf("\n2D validation & performance test!\n\nn\t");
#ifdef RUN_CUFFT
    printf("\tcuFFT");
#endif
    printf("\tMyCUDA");
    double diff = 0.0;
    for (unsigned int n = power2(start); n <= power2(end); n *= 2) {
        printf("\n%d:", n);
        char *fmt = (n > 1000000 ? "\t%.0f" : "\t\t%.0f");
#ifdef RUN_CUFFT
        printf(fmt, measures_cuFFT[measureIndex] = cuFFT_2D_Performance(n));

        printf("\t%.0f", measures[measureIndex] = tsCombine2D_Performance(n));
        if (tsCombine2D_Validate(n) == 0) printf("!");
        if (cuFFT_2D_Compare(n, &diff) == 0) printf("Fishy(%f)", diff);

        if (n < 8192 && 0) {
            printf("\t%.0f", measures_surf[measureIndex] = tsCombine2DSurf_Performance(n));
            if (tsCombine2DSurf_Validate(n) == 0) printf("!");
    }
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
    toFile("Combine 2D Surface CPU and GPU Sync", measures_surf, RUNS);
    printf("\nDone...");
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

__host__ int cuFFT_2D_Compare(int n, double *diff)
{
    cpx *cuFFT, *myCUDA;
    cpx *dev_in, *dev_out;
    size_t size;
    fft2DSetup(&cuFFT, &myCUDA, &dev_in, &dev_out, &size, "shore", 0, n);

    cufftHandle plan;
    cufftPlan2d(&plan, n, n, CUFFT_C2C);
    cufftExecC2C(plan, dev_in, dev_out, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    cudaMemcpy(cuFFT, dev_out, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(dev_in, myCUDA, size, cudaMemcpyHostToDevice);
    tsCombine2D(FFT_FORWARD, &dev_in, &dev_out, n);

    cufftDestroy(plan);
    int res = fft2DCompare(myCUDA, cuFFT, dev_out, size, n * n, diff);
    fft2DShakedown(&myCUDA, &cuFFT, &dev_in, &dev_out);
    return res;
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
#ifndef PROFILER
    printf("Wrote '%s'\n", filename);
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