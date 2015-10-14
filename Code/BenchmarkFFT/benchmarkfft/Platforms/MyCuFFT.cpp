#include "MyCuFFT.h"

MyCuFFT::MyCuFFT(const int dim, const int runs)
    : Platform(dim)
{
    name = "cuFFT";
}

MyCuFFT::~MyCuFFT()
{
}

bool MyCuFFT::validate(const int n)
{
    return true;
}

void MyCuFFT::runPerformance(const int n)
{
#ifdef _WIN64
    double measures[NUM_PERFORMANCE];
    cpx *dev_in;
    cufftHandle plan;
<<<<<<< HEAD
    fftMalloc(n, &dev_in, NULL, NULL, NULL, NULL);    
=======
    fftMalloc(n, &dev_in, &dev_out, NULL, NULL, NULL, NULL);    
>>>>>>> parent of f544fdb... blaj
    if (dimensions == 1) {
        cufftPlan1d(&plan, n, CUFFT_C2C, 1);
        for (int i = 0; i < NUM_PERFORMANCE; ++i) {
            startTimer();
            cufftExecC2C(plan, dev_in, dev_in, CUFFT_FORWARD);
            cudaDeviceSynchronize();
            measures[i] = stopTimer();
        }
    }
    else {
        cufftPlan2d(&plan, n, n, CUFFT_C2C);
        for (int i = 0; i < NUM_PERFORMANCE; ++i) {
            startTimer();
            cufftExecC2C(plan, dev_in, dev_in, CUFFT_FORWARD);
            cudaDeviceSynchronize();
            measures[i] = stopTimer();
        }       
    }
    cufftDestroy(plan);
<<<<<<< HEAD
    fftResultAndFree(n, &dev_in, NULL, NULL, NULL, NULL);
=======
    fftResultAndFree(n, &dev_in, &dev_out, NULL, NULL, NULL, NULL);
>>>>>>> parent of f544fdb... blaj
    results.push_back(avg(measures, NUM_PERFORMANCE));
#endif
}