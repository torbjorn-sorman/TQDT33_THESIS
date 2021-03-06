#pragma once
#ifndef MYDIRECTX_H
#define MYDIRECTX_H

#include "Platform.h"
#include "DirectX\dx_fft.h"

class MyDirectX : public Platform
{
private:
    double performanceTime;    
public:
    MyDirectX::MyDirectX(const int dim, const int runs);
    ~MyDirectX();
    virtual bool MyDirectX::validate(const int n, bool write_img);
    virtual void MyDirectX::runPerformance(const int n);
};

#endif