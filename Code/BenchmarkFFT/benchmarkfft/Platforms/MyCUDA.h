#pragma once
#ifndef MYCUDA_H
#define MYCUDA_H
#if defined(_NVIDIA)
#include "cuda_profiler_api.h"
#include "Platform.h"
#include "CUDA/cuda_fft.cuh"
#include "CUDA/cuda_fft_surface.cuh"

class MyCUDA : public Platform
{
public:
    MyCUDA::MyCUDA(const int dim, const int runs);
    ~MyCUDA();
    virtual bool MyCUDA::validate(const int n, bool write_img);
    virtual void MyCUDA::runPerformance(const int n);
};

#endif
#endif