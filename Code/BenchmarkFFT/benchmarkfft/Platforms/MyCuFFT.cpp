#include "MyCuFFT.h"
#if defined(_NVIDIA)
MyCuFFT::MyCuFFT(const int dim, const int runs)
    : Platform(dim)
{
    name = "cuFFT";
}

MyCuFFT::~MyCuFFT()
{
}

bool MyCuFFT::validate(const int n, bool write_img)
{
    return true;
}

void MyCuFFT::runPerformance(const int n)
{
#if defined(_WIN64)
#ifndef MEASURE_BY_TIMESTAMP
    double measures[number_of_tests];
    cpx *dev_in, *dev_out;
    cufftHandle plan;
    cuda_setup_buffers(n, &dev_in, &dev_out, NULL, NULL, NULL);   
    if (dimensions == 1) {
        cufftPlan1d(&plan, n, CUFFT_C2C, 1);
        for (int i = 0; i < number_of_tests; ++i) {
            start_timer();
            cufftExecC2C(plan, dev_in, dev_out, CUFFT_FORWARD);
            cudaDeviceSynchronize();            
            measures[i] = stop_timer();
        }
    }
    else {
        cufftPlan2d(&plan, n, n, CUFFT_C2C);
        for (int i = 0; i < number_of_tests; ++i) {
            start_timer();
            cufftExecC2C(plan, dev_in, dev_out, CUFFT_FORWARD);
            measures[i] = stop_timer();
        }       
    }
    cufftDestroy(plan);
    cuda_shakedown(n, &dev_in, &dev_out, NULL, NULL, NULL);
    results.push_back(average_best(measures, number_of_tests));
    #else
    double measures[64];
    cpx *dev_in, *dev_out;
    cufftHandle plan;
    cuda_setup_buffers(n, &dev_in, &dev_out, NULL, NULL, NULL);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    if (dimensions == 1) {
        cufftPlan1d(&plan, n, CUFFT_C2C, 1);
        for (int i = 0; i < number_of_tests; ++i) {
            cudaEventRecord(start);
            cufftExecC2C(plan, dev_in, dev_out, CUFFT_FORWARD);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            measures[i] = milliseconds * 1000;
        }
    }
    else {
        cufftPlan2d(&plan, n, n, CUFFT_C2C);
        for (int i = 0; i < number_of_tests; ++i) {
            cudaEventRecord(start);
            cufftExecC2C(plan, dev_in, dev_out, CUFFT_FORWARD);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            measures[i] = milliseconds * 1000;
        }
    }
    cufftDestroy(plan);
    cuda_shakedown(n, &dev_in, &dev_out, NULL, NULL, NULL);
    results.push_back(average_best(measures, number_of_tests));
#endif
#endif
}
#endif