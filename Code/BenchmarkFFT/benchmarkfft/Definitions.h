#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include "cuComplex.h"

//
// Typedefs to improve code readability and better semantics
//

typedef cuFloatComplex cpx;
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
// Hardware & Tweak defines, no need to poll this every time program runs.
//
#define MAX_BLOCK_SIZE 1024             
#define TILE_DIM 64
#define THREAD_TILE_DIM 32              
#define NO_STREAMING_MULTIPROCESSORS 1 // 7 // GTX670 specific, limits block synchronization
#define MAX_THREADS_PER_BLOCK 1024          // GTX670 specific
#define HW_LIMIT ((MAX_THREADS_PER_BLOCK / MAX_BLOCK_SIZE) * NO_STREAMING_MULTIPROCESSORS)
#define SHARED_MEM_SIZE 49152   // 48K assume: 49152 bytes. Total mem size is 65536, where 16384 is cache if

//
// Test related defines
//
#define MEASURE_BY_TIMESTAMP
#define NUM_PERFORMANCE 12
#define HIGHEST_EXP 26
#define HIGHEST_EXP_2D 14

struct benchmarkArgument {
    int dimensions = 1;
    int start = 2;
    int end = 10;
    int number_of_lengths = 8;
    bool test_platform = false;
    bool performance_metrics = false;
    bool platform_cufft = false;
    bool platform_cuda = false;
    bool platform_opencl = false;
    bool platform_opengl = false;
    bool platform_directx = false;
    bool platform_id3dx11 = false;
    bool platform_c = false;
    bool platform_openmp = false;
    bool platform_fftw = false;
    bool profiler = false;
    bool show_cuprop = false;
    bool write_file = false;
    bool validate = false;
    bool display = false;
    bool run_testground = false;
};

#endif