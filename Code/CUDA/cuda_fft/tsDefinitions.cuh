#ifndef TSDEFINITIONS_H
#define TSDEFINITIONS_H

#include "cuComplex.h"

typedef cuFloatComplex cpx;
typedef const float fftDirection;

typedef void(*fftFunction)(fftDirection direction, cpx **in, cpx **out, const int n);
typedef void(*transposeFunction)(cpx **seq, const int, const int n);
typedef void(*twiddleFunction)(cpx *W, const int, const int n);
typedef void(*bitReverseFunction)(cpx *seq, const double dir, const int lead, const int n);

#define M_2_PI 6.28318530718f
#define M_PI 3.14159265359f

#define FFT_FORWARD -1.0f
#define FFT_INVERSE 1.0f

#define SHARED_MEM_SIZE 49152 // 48K assume: 49152 bytes. Total mem size is 65536, where 16384 is cache if 
#define MAX_BLOCK_SIZE 4//1024

//#define PRECALC_TWIDDLE
#define BIT_REVERSED_OUTPUT

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#define SYNC_THREADS __syncthreads()
#define BIT_REVERSE(x, l) ((__brev((x))) >> (l))
#define SIN_COS_F(a, x, y) sincosf(a, x, y)
#define FIND_FIRST_BIT(v) (__ffs(v))
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#define SYNC_THREADS
#define BIT_REVERSE(x, l) 0
#define SIN_COS_F(a, x, y)
#define FIND_FIRST_BIT(v)
#endif

#endif