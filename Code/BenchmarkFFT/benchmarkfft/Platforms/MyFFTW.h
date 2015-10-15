#ifndef MYFFTW_H
#define MYFFTW_H

#include "Platform.h"
#include "../Definitions.h"
#include "../Common/mymath.h"
#include "../Common/mytimer.h"
#include "FFTW\fftw3.h"

class MyFFTW : public Platform
{
public:
    MyFFTW::MyFFTW(const int dim, const int runs);
    ~MyFFTW();
    virtual bool MyFFTW::validate(const int n);
    virtual void MyFFTW::runPerformance(const int n);
};

#endif