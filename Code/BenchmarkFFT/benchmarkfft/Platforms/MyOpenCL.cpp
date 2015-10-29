#include "MyOpenCL.h"

MyOpenCL::MyOpenCL(const int dim, const int runs)
    : Platform(dim)
{
    name = "OpenCL";
}

MyOpenCL::~MyOpenCL()
{
}

bool MyOpenCL::validate(const int n, bool write_img)
{   
    if (dimensions == 1)
        return ocl_validate(n) == 1;
    return ocl_2d_validate(n, write_img) == 1;
}

void MyOpenCL::runPerformance(const int n)
{
    double time = ((dimensions == 1) ? ocl_performance(n) : ocl_2d_performance(n));
    results.push_back(time);
}
