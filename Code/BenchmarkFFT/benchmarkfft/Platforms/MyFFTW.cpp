#include "MyFFTW.h"

MyFFTW::MyFFTW(const int dim, const int runs)
    : Platform(dim)
{
    name = "FFTW";
}

MyFFTW::~MyFFTW()
{
}

bool MyFFTW::validate(const int n)
{   
    return true;
}

void MyFFTW::runPerformance(const int n)
{
#ifdef _WIN64
    bool dim1 = (dimensions == 1);
    double measures[NUM_PERFORMANCE];    
    fftwf_plan p;
    fftwf_complex *fftw_in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (dim1 ? n : n * n));
    fftwf_complex *fftw_out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (dim1 ? n : n * n));
    
    if (dim1) {
        p = fftwf_plan_dft_1d(n, fftw_in, fftw_out, FFTW_FORWARD, FFTW_MEASURE);
        for (int i = 0; i < NUM_PERFORMANCE; ++i) {
            startTimer();
            fftwf_execute(p);
            measures[i] = stopTimer();
        }
    }
    else {
        p = fftwf_plan_dft_2d(n, n, fftw_in, fftw_out, FFTW_FORWARD, FFTW_MEASURE);
        for (int i = 0; i < NUM_PERFORMANCE; ++i) {
            startTimer();
            fftwf_execute(p);
            measures[i] = stopTimer();
        }
    }
    
    fftwf_destroy_plan(p);
    fftwf_free(fftw_in);
    fftwf_free(fftw_out);
    results.push_back(average_best(measures, NUM_PERFORMANCE));    
#endif
}
