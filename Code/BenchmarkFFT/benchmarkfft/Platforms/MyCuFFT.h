#ifndef MYCUFFT_H
#define MYCUFFT_H

#include "Platform.h"
#include <cufft.h>
#include "CUDA/cuda_helper.cuh"
#include "../Common/mymath.h"
#include "../Common/mytimer.h"

class MyCuFFT : public Platform
{
public:
    MyCuFFT::MyCuFFT(const int dim, const int runs);
    ~MyCuFFT();
    virtual bool MyCuFFT::validate(const int n);
    virtual void MyCuFFT::runPerformance(const int n);
};

#endif