#ifndef MYOPENCL_H
#define MYOPENCL_H

#include "Platform.h"
#include "OpenCL/ocl_fft.h"

class MyOpenCL : public Platform
{
public:
    MyOpenCL::MyOpenCL(const int dim, const int runs);
    ~MyOpenCL();
    virtual bool MyOpenCL::validate(const int n);
    virtual void MyOpenCL::runPerformance(const int n);
};

#endif