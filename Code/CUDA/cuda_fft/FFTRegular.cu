#include <stdio.h>
#include <device_launch_parameters.h>

#include "math.h"
#include "FFTRegular.cuh"
#include "fft_helper.cuh"
#include "fft_test.cuh"

__global__ void _FFTBody(cpx *in, cpx *out, cpx *W, const int dist, const int dist2, const int n2);
__host__ __inline void _setBlocksAndThreads(int *numBlocks, int *threadsPerBlock, const int size);


__host__ int FFTRegular_Validate(const size_t n)
{
    int result;
    cpx *in, *ref, *out, *dev_in, *dev_out, *dev_W;

    _fftTestSeq(n, &in, &ref, &out);
    _cudaMalloc(n, &dev_in, &dev_out, &dev_W);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);

    FFTRegular(FFT_FORWARD, &dev_in, &dev_out, dev_W, n);
    FFTRegular(FFT_INVERSE, &dev_out, &dev_in, dev_W, n);

    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);
    _cudaFree(&dev_in, &dev_out, &dev_W);
    cudaDeviceSynchronize();

    result = checkError(in, ref, n);
    _fftFreeSeq(&in, &out, &ref);
    return result != 1;
}

__host__ double FFTRegular_Performance(const size_t n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *out, *dev_in, *dev_out, *dev_W;

    _fftTestSeq(n, &in, &ref, &out);
    _cudaMalloc(n, &dev_in, &dev_out, &dev_W);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);

    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        FFTRegular(FFT_FORWARD, &dev_in, &dev_out, dev_W, n);        
        measures[i] = stopTimer();
    }

    _cudaFree(&dev_in, &dev_out, &dev_W);
    cudaDeviceSynchronize();
    _fftFreeSeq(&in, &out, &ref);
    return avg(measures, NUM_PERFORMANCE);
}

__host__ void FFTRegular(fftDirection dir, cpx **dev_in, cpx **dev_out, cpx *dev_W, const int n)
{
    int dist, dist2, threadsPerBlock, numBlocks;
    const int n2 = (n / 2);
    const float w_angle = dir * (M_2_PI / n);
    const float scale = dir == FFT_FORWARD ? 1.f : 1.f / n;
    dist2 = n;
    dist = n2;

    _setBlocksAndThreads(&numBlocks, &threadsPerBlock, n2);
    //twiddle_factors << < numBlocks, threadsPerBlock >> >(dev_W, w_angle, n);
    //cudaDeviceSynchronize();
    
    //_FFTBody << < numBlocks, threadsPerBlock >> >(*dev_in, *dev_out, dev_W, dist, dist2, n);
    //cudaDeviceSynchronize();
    while ((dist2 = dist) > 1) {
        dist = dist >> 1;
        //_FFTBody << < numBlocks, threadsPerBlock >> >(*dev_out, *dev_out, dev_W, dist, dist2, n);
        //cudaDeviceSynchronize();
    }
    
    _setBlocksAndThreads(&numBlocks, &threadsPerBlock, n);
    //bit_reverse << < numBlocks, threadsPerBlock >> >(*dev_out, *dev_in, scale, 32 - log2_32(n));
    //cudaDeviceSynchronize();
    swap(dev_in, dev_out);
}

__global__ void _FFTBody(cpx *in, cpx *out, cpx *W, const int dist, const int dist2, const int n2)
{
}

/*

__inline void _fft_inner_body(cpx *in, cpx *out, const cpx *W, const int lower, const int upper, const int dist, const int mul)
{
    int u, p;
    float tmp_r, tmp_i;
    for (int l = lower; l < upper; ++l) {
        u = l + dist;
        p = (l - lower) * mul;
        tmp_r = in[l].r - in[u].r;
        tmp_i = in[l].i - in[u].i;
        out[l].r = in[l].r + in[u].r;
        out[l].i = in[l].i + in[u].i;
        out[u].r = (W[p].r * tmp_r) - (W[p].i * tmp_i);
        out[u].i = (W[p].i * tmp_r) + (W[p].r * tmp_i);
    }
}

__inline void _fft_body(cpx *in, cpx *out, cpx *W, int dist, int dist2, const int n_threads, const int n)
{
    const int count = n / dist2;
#ifdef _OPENMP        
    if (count >= n_threads) {
#pragma omp parallel for schedule(static)              
        for (int lower = 0; lower < n; lower += dist2) {
            _fft_inner_body(in, out, W, lower, dist + lower, dist, count);
        }
    }
    else
    {
        int u, p, upper;
        float tmp_r, tmp_i;
        for (int lower = 0; lower < n; lower += dist2) {
            upper = dist + lower;
#pragma omp parallel for schedule(static) private(u, p, tmp_r, tmp_i)
            for (int l = lower; l < upper; ++l) {
                u = l + dist;
                p = (l - lower) * count;
                tmp_r = in[l].r - in[u].r;
                tmp_i = in[l].i - in[u].i;
                out[l].r = in[l].r + in[u].r;
                out[l].i = in[l].i + in[u].i;
                out[u].r = (W[p].r * tmp_r) - (W[p].i * tmp_i);
                out[u].i = (W[p].i * tmp_r) + (W[p].r * tmp_i);
            }
        }
    }
#else
    for (int lower = 0; lower < n; lower += dist2) {
        fft_inner_body(in, out, W, lower, dist + lower, dist, count);
    }
#endif
}

*/

__host__ __inline void _setBlocksAndThreads(int *numBlocks, int *threadsPerBlock, const int size)
{
    int v1 = 256;
    if (size > v1) {
        *numBlocks = size / v1;
        *threadsPerBlock = v1;
    }
    else {
        *numBlocks = 1;
        *threadsPerBlock = size;
    }
}

__host__ __inline void _swap(cpx **in, cpx **out)
{
    cpx *tmp = *in;
    *in = *out;
    *out = tmp;
}