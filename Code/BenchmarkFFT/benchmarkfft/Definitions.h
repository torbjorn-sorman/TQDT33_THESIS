#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include "cuComplex.h"

//
// Typedefs to improve code readability and better semantics
//

typedef cuFloatComplex cpx;

typedef float transform_direction;
/*
typedef void(*fftFunction)(transform_direction direction, cpx **in, cpx **out, int n);
typedef void(*transposeFunction)(cpx **seq, int, int n);
typedef void(*twiddleFunction)(cpx *W, int, int n);
typedef void(*bitReverseFunction)(cpx *seq, double dir, int leading_bits, int n);
*/
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
#define MAX_BLOCK_SIZE 1024     // this limits tsCombineGPUSync!
#define TILE_DIM 64
#define THREAD_TILE_DIM 32      // This squared is the number of threads per block in the transpose kernel.
#define HW_LIMIT ((1024 / MAX_BLOCK_SIZE) * 7)
#define NO_STREAMING_MULTIPROCESSORS 7
#define SHARED_MEM_SIZE 49152   // 48K assume: 49152 bytes. Total mem size is 65536, where 16384 is cache if

//
// Test related defines
//
#define NUM_PERFORMANCE 16
#define HIGHEST_EXP 27
#define HIGHEST_EXP_2D 14

//
// CUDA compiler nvcc intrisics related defines.
//
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#define SYNC_THREADS __syncthreads()
#define BIT_REVERSE(x, l) ((__brev((x))) >> (l))
#define SIN_COS_F(a, x, y) __sincosf(a, x, y)
#define FIND_FIRST_BIT(v) (__ffs(v))
#define ATOMIC_CAS(a,c,v) (atomicCAS((int *)(a),(int)(c),(int)(v)))
#define THREAD_FENCE __threadfence()
#define ATOMIC_ADD(a, b) (atomicAdd((int *)(a), (int)(b)))
#define SURF2D_READ(d,s,x,y) (surf2Dread((d), (s), (x) * sizeof(cpx), (y)))
#define SURF2D_WRITE(d,s,x,y) (surf2Dwrite((d), (s), (x) * sizeof(cpx), (y)));
#else
#define SURF2D_READ(d,s,x,y) 1
#define SURF2D_WRITE(d,s,x,y) 1
#define ATOMIC_ADD(a, b) 1
#define ATOMIC_CAS(a, c, v) 1
#define THREAD_FENCE
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#define SYNC_THREADS
#define BIT_REVERSE(x, l) 0
#define SIN_COS_F(a, x, y)
#define FIND_FIRST_BIT(v)
#endif

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