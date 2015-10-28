#ifndef MYOPENMP_H
#define MYOPENMP_H

#include "Platform.h"
#include "OpenMP/omp_constant_geometry.h"
#include "OpenMP/omp_fast_index.h"

class MyOpenMP : public Platform
{
public:
    MyOpenMP::MyOpenMP(const int dim, const int runs);
    ~MyOpenMP();
    virtual bool MyOpenMP::validate(const int n, bool write_img);
    virtual void MyOpenMP::runPerformance(const int n);
};

#endif