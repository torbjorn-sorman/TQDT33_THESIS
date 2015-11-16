#pragma once
#ifndef MYCLFFT_H
#define MYCLFFT_H

#include "Platform.h"

class MyClFFT : public Platform
{
public:
    MyClFFT::MyClFFT(const int dim, const int runs);
    ~MyClFFT();
    virtual bool MyClFFT::validate(const int n, bool write_img);
    virtual void MyClFFT::runPerformance(const int n);
};

#endif