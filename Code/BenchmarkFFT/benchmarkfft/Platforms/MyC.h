#ifndef MYC_H
#define MYC_H

#include "Platform.h"
#include "C\c_fft.h"

class MyC : public Platform
{
public:
    MyC::MyC(const int dim, const int runs);
    ~MyC();
    virtual bool MyC::validate(const int n);
    virtual void MyC::runPerformance(const int n);
};

#endif