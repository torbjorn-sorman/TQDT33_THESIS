#ifndef FFT_HELPER_CU
#define FFT_HELPER_CU

#include "cuda_runtime.h"

#include "definitions.cuh"

// Fast bit-reversal
static int tab32[32] = { 
    0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30, 8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31 
};

// Host functions (launching kernel with already calculated values)
__host__ int log2_32(int value);

// Texture memory?
__host__ cudaTextureObject_t specifyTexture(cpx *dev_W);

__host__ void swap(cpx **in, cpx **out);

// Kernel functions
__global__ void twiddle_factors(cpx *W, const float angle, const int n);
__global__ void bit_reverse(cpx *in, cpx *out, const float scale, const int lead);
__global__ void bit_reverse(cpx *seq, fftDirection dir, const int lead, const int n);

// Device functions, callable by kernels
__device__ unsigned int bitReverse32(unsigned int x, const int l);

#endif