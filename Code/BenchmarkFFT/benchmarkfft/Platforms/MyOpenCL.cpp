#include "MyOpenCL.h"

MyOpenCL::MyOpenCL(const int dim, const int runs)
    : Platform(dim)
{
    name = "OpenCL";
}

MyOpenCL::~MyOpenCL()
{
}

bool MyOpenCL::validate(const int n)
{   
    if (dimensions == 1)
        return opencl_validate(n) == 1;
    return opencl_2d_validate(n) == 1;
}

void MyOpenCL::runPerformance(const int n)
{
    double time = ((dimensions == 1) ? opencl_performance(n) : opencl_2d_performance(n));
    results.push_back(time);
}
