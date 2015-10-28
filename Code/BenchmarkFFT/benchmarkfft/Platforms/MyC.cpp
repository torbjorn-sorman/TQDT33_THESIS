#include "MyC.h"

MyC::MyC(const int dim, const int runs)
    : Platform(dim)
{
    name = "C_C++";
}

MyC::~MyC()
{
}

bool MyC::validate(const int n, bool write_img)
{   
    
    if (dimensions == 1)
        return c_validate(n) == 1;
    return c_2d_validate(n, write_img) == 1;
}

void MyC::runPerformance(const int n)
{
    double time = ((dimensions == 1) ? c_performance(n) : c_2d_performance(n));    
    results.push_back(time);
}
