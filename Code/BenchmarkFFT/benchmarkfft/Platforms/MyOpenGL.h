#pragma once
#ifndef MYOPENCL_H
#define MYOPENCL_H

#include "Platform.h"
#include "OpenGL/ogl_fft.h"

class MyOpenGL : public Platform
{
public:
    MyOpenGL::MyOpenGL(const int dim, const int runs);
    ~MyOpenGL();
    virtual bool MyOpenGL::validate(const int n, bool write_img);
    virtual void MyOpenGL::runPerformance(const int n);
};

#endif