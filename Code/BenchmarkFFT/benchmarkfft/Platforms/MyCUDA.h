#ifndef MYCUDA_H
#define MYCUDA_H

#include "Platform.h"
#include "CUDA/cuda_fft.cuh"

class MyCUDA : public Platform
{
public:
    MyCUDA::MyCUDA(const int dim, const int runs);
    ~MyCUDA();
    virtual bool MyCUDA::validate(const int n);
    virtual void MyCUDA::runPerformance(const int n);
};

#endif