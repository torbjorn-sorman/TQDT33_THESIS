#ifndef PLATFORM_H
#define PLATFORM_H

#include <string>
#include <vector>
class Platform
{
public:
    std::string name;
    std::vector<double> results;

    Platform()
    {
        results = std::vector<double>();
    }

    Platform(const int dim)
    {
        dimensions = dim;
    }

    virtual ~Platform()
    {
    }

    virtual bool validate(const int n, bool write_img)
    {
        return false;
    }

    virtual void runPerformance(const int n)
    {
    }

protected:
    int dimensions;
};

#endif