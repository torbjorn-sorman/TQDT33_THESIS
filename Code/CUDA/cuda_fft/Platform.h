#pragma once
#include <string>
#include <Windows.h>
#include <vector>
class Platform
{
public:
    std::string name;
    std::vector<double> results;
    int dimensions;
    Platform()
    {
    }
    virtual ~Platform()
    {
    }
    virtual bool validate(const int n)
    {
        return 1;
    }
    virtual void performance(const int n)
    {
    }
};

