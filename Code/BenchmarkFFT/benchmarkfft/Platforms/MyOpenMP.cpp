#include "MyOpenMP.h"

MyOpenMP::MyOpenMP(const int dim, const int runs)
    : Platform(dim)
{
    name = "OpenMP";
}

MyOpenMP::~MyOpenMP()
{
}

bool MyOpenMP::validate(const int n)
{   
    if (dimensions == 1)
        return ompConstantGeometry_validate(n) == 1;
    return ompConstantGeometry2D_validate(n) == 1;
}

void MyOpenMP::runPerformance(const int n)
{
    double time = ((dimensions == 1) ? ompConstantGeometry_runPerformance(n) : ompConstantGeometry2D_runPerformance(n));
    results.push_back(time);
}
