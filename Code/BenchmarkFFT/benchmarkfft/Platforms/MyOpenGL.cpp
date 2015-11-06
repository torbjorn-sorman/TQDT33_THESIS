#include "MyOpenGL.h"
#include <iostream>

MyOpenGL::MyOpenGL(const int dim, const int runs)
    : Platform(dim)
{
    name = "OpenGL";    
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    }
    fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
    
}

MyOpenGL::~MyOpenGL()
{
}

bool MyOpenGL::validate(const int n, bool write_img)
{   
    if (dimensions == 1)
        return gl_validate(n) == 1;
    return gl_2d_validate(n, write_img) == 1;
}

void MyOpenGL::runPerformance(const int n)
{
    double time = ((dimensions == 1) ? gl_performance(n) : gl_2d_performance(n));
    results.push_back(time);
}