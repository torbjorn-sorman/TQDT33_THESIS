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
    bool valid = ((dimensions == 1) ? ompConstantGeometry_validate(n) == 1 : ompConstantGeometry2D_validate(n) == 1);
#else
    bool valid = ((dimensions == 1) ? ompFastIndex_validate(n) == 1 : ompFastIndex2D_validate(n) == 1);
#endif
    return valid;
}

void MyOpenMP::runPerformance(const int n)
{
#ifndef OMP_FAST_INDEX
    double time = ((dimensions == 1) ? ompConstantGeometry_runPerformance(n) : ompConstantGeometry2D_runPerformance(n));    
#else
    double time = ((dimensions == 1) ? ompFastIndex_runPerformance(n) : ompFastIndex2D_runPerformance(n));    
#endif
    results.push_back(time);
}
