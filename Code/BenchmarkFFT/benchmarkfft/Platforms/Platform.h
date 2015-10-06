#pragma once
#include <string>
#include <vector>
class Platform
{
public:
    std::string name;
    std::vector<double> results;

    Platform()
    {        
    }

    virtual ~Platform()
    {
    }

    virtual bool validate(const int n)
    {
        return false;
    }

    virtual void runPerformance(const int n)
    {
    }

protected:
    int dimensions;
};

