#ifndef MYID3DX11FFT_H
#define MYID3DX11FFT_H

#include <iostream>
#include <d3d11.h>
#include <D3dcsx.h>

#include "Platform.h"
#include "DirectX\dx_helper.h"


class MyID3DX11FFT : public Platform
{
public:
    MyID3DX11FFT::MyID3DX11FFT(const int dim, const int runs);
    ~MyID3DX11FFT();
    virtual bool MyID3DX11FFT::validate(const int n);
    virtual void MyID3DX11FFT::runPerformance(const int n);
};

#endif