#include "MyCUDA.h"

MyCUDA::MyCUDA(const int dim, const int runs)
{
    name = "CUDA";
    dimensions = dim;
    results = std::vector<double>(runs);
}

MyCUDA::~MyCUDA()
{
}

bool MyCUDA::validate(const int n)
{   
    if (dimensions == 1)
        return tsCombine_Validate(n);
    return tsCombine2D_Validate(n);
}

void MyCUDA::runPerformance(const int n)
{
    printf("RUN!");
    if (dimensions == 1) {
        printf("RUN!");
        results.push_back(tsCombine_Performance(n));
    }
    else {
        results.push_back(tsCombine2D_Performance(n));
    }
}
