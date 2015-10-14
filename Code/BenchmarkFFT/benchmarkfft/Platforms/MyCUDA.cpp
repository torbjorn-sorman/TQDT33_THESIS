#include "MyCUDA.h"

MyCUDA::MyCUDA(const int dim, const int runs)
    : Platform(dim)
{
    name = "CUDA";
}

MyCUDA::~MyCUDA()
{
}

bool MyCUDA::validate(const int n)
{   
    if (dimensions == 1)
        return CUDA_validate(n) == 1;
    return CUDA2D_validate(n) == 1;
}

void MyCUDA::runPerformance(const int n)
{
    cudaProfilerStart();
    double time = ((dimensions == 1) ? CUDA_performance(n) : CUDA2D_performance(n));
    results.push_back(time);
    cudaProfilerStop();
}
