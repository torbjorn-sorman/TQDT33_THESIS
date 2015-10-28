
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include "Definitions.h"
#include "Common\parsearg.h"
#include "Platforms\Platform.h"
#include "Platforms\MyCUDA.h"
#include "Platforms\MyOpenCL.h"
#include "Platforms\MyCuFFT.h"
#include "Platforms\MyOpenMP.h"
#include "Platforms\MyC.h"
#include "Platforms\MyFFTW.h"
#include "Platforms\MyDirectX.h"
#include "Platforms\MyID3DX11FFT.h"

void printDevProp(cudaDeviceProp devProp);
void toFile(std::string name, std::vector<double> results, int ms);

std::vector<Platform *> getPlatforms(benchmarkArgument *args)
{
    std::vector<Platform *> platforms;
    // cuFFT only shipped for x64, FFTW x64 selected
#ifdef _WIN64
    if (args->platform_cufft)
        platforms.insert(platforms.begin(), new MyCuFFT(args->dimensions, args->number_of_lengths));
    if (args->platform_fftw)
        platforms.insert(platforms.begin(), new MyFFTW(args->dimensions, args->number_of_lengths));
#endif
    if (args->platform_cuda)
        platforms.insert(platforms.begin(), new MyCUDA(args->dimensions, args->number_of_lengths));
    if (args->platform_opencl)
        platforms.insert(platforms.begin(), new MyOpenCL(args->dimensions, args->number_of_lengths));
    if (args->platform_directx)
        platforms.insert(platforms.begin(), new MyDirectX(args->dimensions, args->number_of_lengths));
    if (args->platform_id3dx11)
        platforms.insert(platforms.begin(), new MyID3DX11FFT(args->dimensions, args->number_of_lengths));
    if (args->platform_openmp)
        platforms.insert(platforms.begin(), new MyOpenMP(args->dimensions, args->number_of_lengths));
    if (args->platform_c)
        platforms.insert(platforms.begin(), new MyC(args->dimensions, args->number_of_lengths));
    return platforms;
}

int testground()
{
    return 0;
}

int main(int argc, const char* argv[])
{
    benchmarkArgument args;
    if (!parseArguments(&args, argc, argv))
        return 0;
    if (args.run_testground) {
        std::cout << "Running Testground" << std::endl;
        testground();
    }
    if (args.profiler) {
        if (args.test_platform) {
            cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
            std::vector<Platform *> platforms = getPlatforms(&args);
            for (int i = args.start; i <= args.end; ++i) {
                int n = power2(i);
                for (Platform *platform : platforms)
                    platform->runPerformance(n);
            }
            platforms.clear();
        }
        return 0;
    }
    if (args.show_cuprop) {
        std::cout << "Control Show CUDA Properties" << std::endl;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printDevProp(prop);
    }
    if (args.test_platform) {
        std::cout << "Control Test Platforms" << std::endl;
        std::vector<Platform *> platforms = getPlatforms(&args);
        if (args.performance_metrics) {
            cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
            cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
            std::cout << "  Running Platform Performance Test (might take a while)";
            for (int i = args.start; i <= args.end; ++i) {
                int n = power2(i);
                for (Platform *platform : platforms)
                    platform->runPerformance(n);
                std::cout << '.';
            }
            std::cout << std::endl;
        }
        if (args.display) {
            int tabs = 1 + (int)log10(power2(args.start + args.number_of_lengths)) / 7;
            std::cout << std::string(tabs, '\t');
            for (Platform *platform : platforms)
                std::cout << platform->name << "\t";
            std::cout << std::endl;
            std::cout << std::fixed;
            std::cout << std::setprecision(0);
            for (int i = 0; i < args.number_of_lengths; ++i) {
                std::cout << power2(i + args.start) << std::string(tabs - (int)log10(power2(args.start + i)) / 7, '\t');
                for (Platform *platform : platforms) {
                    if (args.performance_metrics)
                        std::cout << platform->results.at(i);
                    if (!args.performance_metrics) {
                        if (args.validate && !platform->validate(power2(args.start + i), args.write_img))
                            std::cout << "FAIL";
                        else
                            std::cout << "OK";
                    }
                    else {
                        if (args.validate && !platform->validate(power2(args.start + i), args.write_img))
                            std::cout << "!";
                    }
                    std::cout << "\t";
                }
                std::cout << std::endl;
            }
        }
        if (args.write_file && args.performance_metrics) {
            for (Platform *platform : platforms)
                toFile(platform->name, platform->results, args.number_of_lengths);
        }
        std::cout << "  Test Platforms Complete" << std::endl;
        std::cout << "Press the any key to continue...";
        getchar();
        platforms.clear();
    }
    std::cout << "Benchmark FFT finished" << std::endl;
    return 0;
}

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

void toFile(std::string name, std::vector<double> results, int ms)
{
    std::string filename;
    FILE *f = getTextFilePointer(name, &filename);
    for (int i = 0; i < ms; ++i) {
        int val = (int)floor(results[i]);
        int dec = (int)floor((results[i] - val) * 10);
        fprintf_s(f, "%d,%1d\n", val, dec);
    }
    fclose(f);
    std::cout << "Wrote '" << filename << "'" << std::endl;
}