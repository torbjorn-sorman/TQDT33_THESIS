#include "MyCUDA.h"

MyCUDA::MyCUDA(const int dimensions)
{
    MyCUDA::name = "CUDA";
    MyCUDA::dimensions = dimensions;
}

MyCUDA::~MyCUDA()
{
}

bool MyCUDA::validate(const int n)
{
    if (MyCUDA::dimensions == 1)
        return tsCombine_Validate(n);
    return tsCombine2D_Validate(n);
}

void MyCUDA::performance(const int n)
{
    if (MyCUDA::dimensions == 1)
        MyCUDA::results.push_back(tsCombine_Performance(n));
    else
        MyCUDA::results.push_back(tsCombine2D_Performance(n));
}