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
    double measures[64];
    cpx *dev_in, *dev_out;
    cufftHandle plan;
    cuda_setup_buffers(n, &dev_in, &dev_out, NULL, NULL, NULL);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    if (dimensions == 1) {
        int rank[1] = { n };
        cufftPlanMany(&plan, 1, rank, NULL, 1, 1024, NULL, 1, 1024, CUFFT_C2C, batch_count(n));
        for (int i = 0; i < number_of_tests; ++i) {
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            cufftExecC2C(plan, dev_in, dev_out, CUFFT_FORWARD);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            measures[i] = milliseconds * 1000;
        }
    }
    else {
        int rank[2] = { n, n };
        cufftPlanMany(&plan, 2, rank, NULL, 1, 1024, NULL, 1, 1024, CUFFT_C2C, batch_count(n * n));

        for (int i = 0; i < number_of_tests; ++i) {
            cudaDeviceSynchronize();
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
}
#endif