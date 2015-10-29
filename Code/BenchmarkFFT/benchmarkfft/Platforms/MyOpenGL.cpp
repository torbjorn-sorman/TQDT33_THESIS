#include "MyOpenGL.h"

MyOpenGL::MyOpenGL(const int dim, const int runs)
    : Platform(dim)
{
    name = "OpenCL";
}

MyOpenGL::~MyOpenGL()
{
}

bool MyOpenGL::validate(const int n, bool write_img)
{   
    if (dimensions == 1)
        return ogl_validate(n) == 1;
    return ogl_2d_validate(n, write_img) == 1;
}

void MyOpenGL::runPerformance(const int n)
{
    double time = ((dimensions == 1) ? ogl_performance(n) : ogl_2d_performance(n));
    results.push_back(time);
}
