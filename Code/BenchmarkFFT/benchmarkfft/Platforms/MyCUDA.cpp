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
        return tsCombine_Validate(n) == 1;
    return tsCombine2D_Validate(n) == 1;
}

void MyCUDA::runPerformance(const int n)
{
    double time = ((dimensions == 1) ? tsCombine_Performance(n) : tsCombine2D_Performance(n));
    results.push_back(time);
}
