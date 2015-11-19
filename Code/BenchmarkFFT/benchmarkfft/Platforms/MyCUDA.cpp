#include "MyCUDA.h"
#if defined(_NVIDIA)
MyCUDA::MyCUDA(const int dim, const int runs)
    : Platform(dim)
{
    name = "CUDA";
    int v;
    cudaDriverGetVersion(&v);
    printf("CUDA:\t\t%d.%d\n", (v/1000), (v - ((v/1000)*1000)));
}

MyCUDA::~MyCUDA()
{
}

bool MyCUDA::validate(const int n, bool write_img)
{
    if (dimensions == 1)
        return cuda_validate(n) == 1;
    return cuda_2d_validate(n, write_img) == 1;
}

void MyCUDA::runPerformance(const int n)
{
    cudaProfilerStart();
    double time = ((dimensions == 1) ? cuda_performance(n) : cuda_2d_performance(n));
    results.push_back(time);
    cudaProfilerStop();
}
#endif