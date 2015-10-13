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
        return OCL_validate(n) == 1;
    return OCL2D_validate(n) == 1;
}

void MyOpenCL::runPerformance(const int n)
{
    double time = ((dimensions == 1) ? OCL_performance(n) : -1);//OCL2D_performance(n));
    results.push_back(time);
}
