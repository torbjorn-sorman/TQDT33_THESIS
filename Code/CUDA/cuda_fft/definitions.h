#ifndef TSDEFINITIONS_H
#define TSDEFINITIONS_H

#include "cuComplex.h"

// Save space for better overview when many params
typedef cuFloatComplex cpx;
typedef cudaSurfaceObject_t cuSurf;
typedef float fftDir;

typedef void(*fftFunction)(fftDir direction, cpx **in, cpx **out, int n);
typedef void(*transposeFunction)(cpx **seq, int, int n);
typedef void(*twiddleFunction)(cpx *W, int, int n);
typedef void(*bitReverseFunction)(cpx *seq, double dir, int lead, int n);

#define M_2_PI 6.28318530718f
#define M_PI 3.14159265359f

#define FFT_FORWARD -1.0f
#define FFT_INVERSE 1.0f

#define SHARED_MEM_SIZE 49152   // 48K assume: 49152 bytes. Total mem size is 65536, where 16384 is cache if 
#define MAX_BLOCK_SIZE 1024     // this limits tsCombineGPUSync!
#define TILE_DIM 64
#define THREAD_TILE_DIM 32      // This squared is the number of threads per block in the transpose kernel.

#define BIT_REVERSED_OUTPUT

// Test
#define NUM_PERFORMANCE 10
#define HIGHEST_EXP 25

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
    int test_platform = 0;
    int platform_cufft = 0;
    int platform_cuda = 0;
    int platform_opencl = 0;
    int platform_opengl = 0;
    int platform_directx = 0;
    int platform_c = 0;
    int platform_openmp = 0;
    int profiler = 0;
    int show_cuprop = 0;
    int write_file = 0;
    int validate = 0;
    int display = 0;
};

#endif