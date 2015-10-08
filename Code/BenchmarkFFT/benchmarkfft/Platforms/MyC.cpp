#include "MyC.h"

MyC::MyC(const int dim, const int runs)
    : Platform(dim)
{
    name = "C_C++";
}

MyC::~MyC()
{
}

bool MyC::validate(const int n)
{   
    
    if (dimensions == 1)
        return cConstantGeometry_validate(n) == 1;
    return cConstantGeometry2D_validate(n) == 1;
}

void MyC::runPerformance(const int n)
{
    double time = ((dimensions == 1) ? cConstantGeometry_runPerformance(n) : cConstantGeometry2D_runPerformance(n));    
    results.push_back(time);
}
