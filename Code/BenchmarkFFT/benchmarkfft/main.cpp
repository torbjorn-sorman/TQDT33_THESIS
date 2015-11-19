#include <iostream>
#include <iomanip>
#include <functional>
#include <algorithm>
#include <vector>

#include "Definitions.h"
#include "Common\parsearg.h"
#include "Platforms\Platform.h"
#if defined(_NVIDIA)
#include "Platforms\MyCUDA.h"
#include "Platforms\MyCuFFT.h"
#endif
#include "Platforms\MyClFFT.h"
#include "Platforms\MyOpenCL.h"
#include "Platforms\MyOpenGL.h"
#include "Platforms\MyOpenMP.h"
#include "Platforms\MyC.h"
#include "Platforms\MyFFTW.h"
#include "Platforms\MyDirectX.h"
//#include "Platforms\MyID3DX11FFT.h"
#if defined(_NVIDIA)
void printDevProp(cudaDeviceProp devProp);
#endif
void toFile(std::string name, std::vector<double> results, int ms);
void toFile(std::string name, std::vector<Platform *> platforms, benchmarkArgument *a);

std::vector<Platform *> getPlatforms(benchmarkArgument *args)
{
    std::vector<Platform *> platforms;
    // cuFFT only shipped for x64, FFTW x64 selected
#if defined(_WIN64)
#if defined(_NVIDIA)
    if (args->platform_cufft) {
        platforms.push_back(new MyCuFFT(args->dimensions, args->test_runs));
    }
#elif defined(_AMD)
    if (args->platform_clfft) {
        platforms.push_back(new MyClFFT(args->dimensions, args->test_runs));
    }
#endif
    if (args->platform_fftw)
        platforms.push_back(new MyFFTW(args->dimensions, args->test_runs));
#endif
#if defined(_NVIDIA)
    if (args->platform_cuda) {
        platforms.push_back(new MyCUDA(args->dimensions, args->test_runs));
    }
#endif
    if (args->platform_directx)
        platforms.push_back(new MyDirectX(args->dimensions, args->test_runs));
    if (args->platform_opengl)
        platforms.push_back(new MyOpenGL(args->dimensions, args->test_runs));
    if (args->platform_opencl)
        platforms.push_back(new MyOpenCL(args->dimensions, args->test_runs));
    /*
    if (args->platform_id3dx11)
    platforms.push_back(new MyID3DX11FFT(args->dimensions, args->test_runs));
    */
    if (args->platform_openmp)
        platforms.push_back(new MyOpenMP(args->dimensions, args->test_runs));
    if (args->platform_c)
        platforms.push_back(new MyC(args->dimensions, args->test_runs));
    return platforms;
}

int testground()
{
    return 0;
}

int main(int argc, char* argv[])
{
    benchmarkArgument args;
    if (!parse_args(&args, argc, argv))
        return 0;
#if defined(_NVIDIA)
    vendor_gpu = VENDOR_NVIDIA;
#elif defined(_AMD)
    vendor_gpu = VENDOR_AMD;
#endif
    // Special setup, OpenGL need a window context!
    if (args.platform_opengl) {
        glutInit(&argc, argv);
        glutCreateWindow("GL Context");
    }
    if (args.profiler) {
        number_of_tests = 1;
        if (args.test_platform) {
#if defined(_NVIDIA)
            cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
#endif
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
#if defined(_NVIDIA)
    if (args.show_cuprop) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printDevProp(prop);
    }
#endif
    if (args.test_platform) {
        std::vector<Platform *> platforms = getPlatforms(&args);
        std::cout << "\n" << std::endl;
        if (args.performance_metrics) {
#if defined(_NVIDIA)
            cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
            cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
#endif
            std::cout << "  Running Platform Tests (might take a while)";
            for (int i = args.start; i <= args.end; ++i) {
                int n = power2(i);
                for (auto platform : platforms) {
                    platform->runPerformance(n);
                }
                std::cout << '.';
            }
            std::cout << std::endl;
        }
        if (args.display) {
            int tabs = 1 + (int)log10(power2(args.start + args.test_runs)) / 7;
            std::cout << std::string(tabs, '\t');
            for (auto platform : platforms) {
                std::cout << platform->name << "\t";
            }
            std::cout << std::endl;
            std::cout << std::fixed;
            std::cout << std::setprecision(0);
            for (int i = 0; i < args.test_runs; ++i) {
                std::cout << power2(i + args.start) << std::string(tabs - (int)log10(power2(args.start + i)) / 7, '\t');
                for (auto platform : platforms) {
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
            for (auto platform : platforms) {
                toFile(platform->name, platform->results, args.test_runs);
            }
            toFile("all", platforms, &args);
        }
        std::cout << "  Test Platforms Complete" << std::endl;
        std::cout << "Press the any key to continue...";
#pragma warning(suppress: 6031)
        getchar();
        platforms.clear();
    }
    std::cout << "Benchmark FFT finished" << std::endl;
    return 0;
}

#if defined(_NVIDIA)
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n", devProp.major);
    printf("Minor revision number:         %d\n", devProp.minor);
    printf("Name:                          %s\n", devProp.name);
    printf("Total global memory:           %Ix\n", devProp.totalGlobalMem);
    printf("Total shared memory per block: %Ix\n", devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n", devProp.regsPerBlock);
    printf("Warp size:                     %d\n", devProp.warpSize);
    printf("Maximum memory pitch:          %Ix\n", devProp.memPitch);
    printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n", devProp.clockRate);
    printf("Total constant memory:         %Ix\n", devProp.totalConstMem);
    printf("Texture alignment:             %Ix\n", devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    printf("\n");
    return;
}
#endif

void toFile(std::string name, std::vector<double> results, int ms)
{
    std::string filename;
    FILE *f = get_txt_file_pntr(name, &filename);
    for (int i = 0; i < ms; ++i) {
        int val = (int)floor(results[i]);
        int dec = (int)floor((results[i] - val) * 10);
        fprintf_s(f, "%d,%1d\n", val, dec);
    }
    fclose(f);
    std::cout << "Wrote '" << filename << "'" << std::endl;
}

void toFile(std::string name, std::vector<Platform *> platforms, benchmarkArgument *a)
{
    std::string filename;
    FILE *f = get_txt_file_pntr(name, &filename);
    fprintf_s(f, "\t");
    for (auto platform : platforms) {
        fprintf_s(f, "%s\t", platform->name.c_str());
    }
    fprintf_s(f, "\n");
    for (int i = 0; i < a->test_runs; ++i) {
        fprintf_s(f, "%d\t", power2(i) * power2(a->start));
        for (auto platform : platforms) {
            int val = (int)floor(platform->results[i]);
            int dec = (int)floor((platform->results[i] - val) * 10);
            fprintf_s(f, "%d,%1d\t", val, dec);
        }
        fprintf_s(f, "\n");
    }
    fclose(f);
    std::cout << "Wrote '" << filename << "'" << std::endl;
}