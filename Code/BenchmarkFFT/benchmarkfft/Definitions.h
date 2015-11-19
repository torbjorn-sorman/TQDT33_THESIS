#pragma once
#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#if defined(_NVIDIA)
#include "cuComplex.h"
#endif
//
// Typedefs to improve code readability and better semantics
//

#if defined(_AMD)
struct cpx
{
    float x, y;
};
struct dim3
{
    unsigned int x, y, z;
};
#else
typedef cuFloatComplex cpx;
#endif
typedef float transform_direction;

//
// Math & Algorithm defines
//
#define M_2_PI 6.28318530718f
#define M_PI 3.14159265359f
#define RELATIVE_ERROR_MARGIN 0.00001
#define FFT_FORWARD -1.0f
#define FFT_INVERSE 1.0f

//
// Test related defines
//
#define MEASURE_BY_TIMESTAMP
#define HIGHEST_EXP 26
#define HIGHEST_EXP_2D 13 // 8192 -> the limit of a 2GB primary mem device. Two buffers cover 8192 * 8192 * 8 * 2 = 1 073 741 824 bytes of memory.

//
// Vendor specific
//
#define VENDOR_NVIDIA   4318
#define VENDOR_AMD      4098
#define VENDOR_BASIC    5140

extern int vendor_gpu;
extern int number_of_tests;

struct benchmarkArgument
{
    int dimensions = 1;
    int start = 2;
    int end = 10;
    int test_runs = 8;
    unsigned int vendor = VENDOR_NVIDIA;
    bool test_platform = false;
    bool performance_metrics = false;
    bool platform_cufft = false;
    bool platform_cuda = false;
    bool platform_opencl = false;
    bool platform_clfft = false;
    bool platform_opengl = false;
    bool platform_directx = false;
    bool platform_id3dx11 = false;
    bool platform_c = false;
    bool platform_openmp = false;
    bool platform_fftw = false;
    bool profiler = false;
    bool show_cuprop = false;
    bool write_file = false;
    bool write_img = false;
    bool validate = false;
    bool display = false;
    bool run_testground = false;
};

#endif