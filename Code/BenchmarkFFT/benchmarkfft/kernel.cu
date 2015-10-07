
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

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

void printDevProp(cudaDeviceProp devProp);
void toFile(std::string name, std::vector<double> results, int ms);

std::vector<Platform *> getPlatforms(benchmarkArgument *args)
{
    std::vector<Platform *> platforms;
    if (args->platform_cuda)
        platforms.insert(platforms.begin(), new MyCUDA(args->dimensions, args->number_of_lengths));
    if (args->platform_opencl)
        platforms.insert(platforms.begin(), new MyOpenCL(args->dimensions, args->number_of_lengths));
    if (args->platform_cufft)
        platforms.insert(platforms.begin(), new MyCuFFT(args->dimensions, args->number_of_lengths));
    if (args->platform_c)
        platforms.insert(platforms.begin(), new MyC(args->dimensions, args->number_of_lengths));
    if (args->platform_openmp)
        platforms.insert(platforms.begin(), new MyOpenMP(args->dimensions, args->number_of_lengths));
    return platforms;
}

int main(int argc, const char* argv[])
{
    benchmarkArgument args;
    std::cout << "Parsing arguments" << std::endl;
    if (!parseArguments(&args, argc, argv))
        return 0;
    std::cout << "Control Show CUDA Properties" << std::endl;
    if (args.show_cuprop) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printDevProp(prop);
    }
    std::cout << "Control Test Platforms" << std::endl;
    if (args.test_platform) {

        cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

        std::vector<Platform *> platforms = getPlatforms(&args);
        std::cout << "  Running Platforms (might take a while)..." << std::endl;
        if (args.profiler) cudaProfilerStart();
        for (int i = args.start; i < args.end; ++i) {
            int n = power2(i);
            for (Platform *platform : platforms)
                platform->runPerformance(n);
        }
        if (args.profiler) cudaProfilerStop();
        if (args.display) {
            std::cout << std::string(1 + (int)log10(power2(args.start + args.number_of_lengths)) / 8, '\t');
            for (Platform *platform : platforms)
                std::cout << platform->name << "\t";
            std::cout << std::endl << std::setprecision(0);
            for (int i = 0; i < args.number_of_lengths; ++i) {
                std::cout << power2(i + args.start) << std::string(2 - (int)log10(power2(args.start + i)) / 8, '\t');
                for (Platform *platform : platforms)
                    std::cout << platform->results.at(i) << (args.validate != 1 || platform->validate(power2(args.start + i)) ? "" : "!") << "\t";
                std::cout << std::endl;
            }
        }
        if (args.write_file)
            for (Platform *platform : platforms)
                toFile(platform->name, platform->results, args.number_of_lengths);
        std::cout << "  Test Platforms Complete" << std::endl;
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
    std::string filename = "out/" + name + ".txt";
    FILE *f;
    fopen_s(&f, filename.c_str(), "w");
    for (int i = 0; i < ms; ++i) {
        int val = (int)floor(results[i]);
        int dec = (int)floor((results[i] - val) * 10);
        fprintf_s(f, "%d,%1d\n", val, dec);
    }
    fclose(f);
    std::cout << "Wrote '" << filename << "'" << std::endl;
}