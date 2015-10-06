
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

void printDevProp(cudaDeviceProp devProp);
void toFile(std::string name, std::vector<double> results, int ms);

std::vector<Platform> getPlatforms(benchmarkArgument *args)
{
    std::vector<Platform> platforms;
    if (args->platform_cuda)
        platforms.insert(platforms.begin(), MyCUDA(args->dimensions, args->number_of_lengths));
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
        cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        std::vector<Platform> platforms = getPlatforms(&args);
        if (args.profiler) cudaProfilerStart();
        for (int n = power2(args.start); n < power2(args.end); ++n) {
            for (Platform platform : platforms) {
                platform.runPerformance(n);
            }
        }
        if (args.profiler) cudaProfilerStop();
        if (args.display) {
            std::cout << "n\t";
            for (Platform platform : platforms)
                std::cout << platform.name << "\t";
            std::cout << "\n";
            std::cout << std::setprecision(0);
            for (int i = 0; i < platforms[0].results.size(); ++i) {
                std::cout << power2(i + args.start) << "\t";
                for (Platform platform : platforms)
                    std::cout << platform.results.at(i) << (args.validate == 1 || platform.validate(power2(args.start + i)) ? "" : "!") << "\t";
                std::cout << "\n";
            }
            getchar();
        }
        if (args.write_file)
            for (Platform platform : platforms)
                toFile(platform.name, platform.results, args.number_of_lengths);
    }
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
    for (int i = 0; i < ms; ++i)
        fprintf_s(f, "%0.f\n", results[i]);
    fclose(f);
    printf("Wrote '%s'\n", filename.c_str());
}