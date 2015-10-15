#include "MyOpenMP.h"

MyOpenMP::MyOpenMP(const int dim, const int runs)
    : Platform(dim)
{
    name = "OpenMP";
}

MyOpenMP::~MyOpenMP()
{
}

//#define OMP_FAST_INDEX

bool MyOpenMP::validate(const int n)
{   
#ifndef OMP_FAST_INDEX
    bool valid = ((dimensions == 1) ? openmp_validate(n) == 1 : openmp_2d_validate(n) == 1);
#else
    bool valid = ((dimensions == 1) ? openmp_fast_index_validate(n) == 1 : openmp_fast_index_2d_validate(n) == 1);
#endif
    return valid;
}

void MyOpenMP::runPerformance(const int n)
{
#ifndef OMP_FAST_INDEX
    double time = ((dimensions == 1) ? openmp_performance(n) : openmp_2d_performance(n));    
#else
    double time = ((dimensions == 1) ? openmp_fast_index_performance(n) : openmp_fast_index_2d_performance(n));    
#endif
    results.push_back(time);
}
