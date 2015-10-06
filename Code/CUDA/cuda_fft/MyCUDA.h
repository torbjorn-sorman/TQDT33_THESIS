#pragma once
#include "Platform.h"
#include "tsCombine.cuh"
#include <vector>
class MyCUDA :
    public Platform
{
public:
    MyCUDA(const int dimensions);
    ~MyCUDA();
    bool validate(const int n);
    void performance(const int n);
};

