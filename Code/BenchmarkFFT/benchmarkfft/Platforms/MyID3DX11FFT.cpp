#include "MyID3DX11FFT.h"

MyID3DX11FFT::MyID3DX11FFT(const int dim, const int runs)
    : Platform(dim)
{
    name = "ID3DX11FFT";
}

MyID3DX11FFT::~MyID3DX11FFT()
{
}

bool MyID3DX11FFT::validate(const int n, bool write_img)
{       
    return false;
}

void MyID3DX11FFT::runPerformance(const int n)
{
    double time = -1.0;
    results.push_back(time);
}
