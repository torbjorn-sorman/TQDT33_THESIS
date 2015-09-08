#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include <cufft.h>

#include "definitions.cuh"
#include "fft_test.cuh"
#include "fft_helper.cuh"

#include "FFTConstGeom.cuh"
#include "FFTRegular.cuh"
#include "FFTTobb.cuh"

__host__ double cuFFT_Performance(const size_t n);

int main()
{    
    printf("\tcuFFT\ttbFFT\ttbFFT\n");
    for (int n = power2(2); n < power2(16); n *= 2) {        
        printf("\n%d:", n);

        // cuFFT
        printf("\t%.0f", cuFFT_Performance(n));

        // Regular (not working, used as ref)
        printf("\t%.0f", FFTRegular_Performance(n));
        if (FFTRegular_Validate(n) == 0) printf("!");        

        // Regular
        printf("\t%.0f", FFTTobb_Performance(n));
        if (FFTTobb_Validate(n) == 0) printf("!");

        // Const geom
        printf("\t%.0f", FFTConstGeom_Performance(n));
        if (FFTConstGeom_Validate(n) == 0) printf("!");
    }
    printf("\nDone...");
    getchar();
    return 0;
}

__host__ double cuFFT_Performance(const size_t n)
{
    double measures[NUM_PERFORMANCE];
    cpx *dev_in = 0;
    cpx *dev_out = 0;
    cudaMalloc((void**)&dev_in, sizeof(cpx) * n);
    cudaMalloc((void**)&dev_out, sizeof(cpx) * n);
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, 1);
    for (int i = 0; i < 20; ++i) {
        startTimer();
        cufftExecC2C(plan, dev_in, dev_out, CUFFT_FORWARD);
        cudaDeviceSynchronize();
        measures[i] = stopTimer();
    }
    cufftDestroy(plan);
    cudaFree(dev_in);
    cudaFree(dev_out);
    return avg(measures, NUM_PERFORMANCE);
}

__host__ double cuFFT_2D_Performance(const size_t n)
{
    double measures[NUM_PERFORMANCE];
    cpx *dev_in = 0;
    cpx *dev_out = 0;
    cudaMalloc((void**)&dev_in, sizeof(cpx) * n * n);
    cudaMalloc((void**)&dev_out, sizeof(cpx) * n * n);
    cufftHandle plan;
    cufftPlan2d(&plan, n, n, CUFFT_C2C);
    for (int i = 0; i < 20; ++i) {
        startTimer();
        cufftExecC2C(plan, dev_in, dev_out, CUFFT_FORWARD);
        cudaDeviceSynchronize();
        measures[i] = stopTimer();
    }
    cufftDestroy(plan);
    cudaFree(dev_in);
    cudaFree(dev_out);
    return avg(measures, NUM_PERFORMANCE);
}