#include "MyDirectX.h"

MyDirectX::MyDirectX(const int dim, const int runs)
    : Platform(dim)
{
    name = "DirectX";
}

MyDirectX::~MyDirectX()
{
}

bool MyDirectX::validate(const int n)
{   
    if (dimensions == 1)
        return dx_validate(n) == 1;
    return dx_2d_validate(n) == 1; 
}

void MyDirectX::runPerformance(const int n)
{
    double time = ((dimensions == 1) ? dx_performance(n) : dx_2d_performance(n));    
    results.push_back(time);
}