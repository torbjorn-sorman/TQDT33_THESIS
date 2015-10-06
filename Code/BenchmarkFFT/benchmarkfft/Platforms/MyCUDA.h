#pragma once
#include "Platform.h"
#include "CUDA/MyFFTCUDA.cuh"
class MyCUDA : public Platform
{
public:
    MyCUDA::MyCUDA(const int dim, const int runs);
    ~MyCUDA();
    virtual bool MyCUDA::validate(const int n);
    virtual void MyCUDA::runPerformance(const int n);
};

