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
    fprintf(stdout, "OpenGL:\t\t%s\n", glGetString(GL_VERSION));
}

MyOpenGL::~MyOpenGL()
{
}

bool MyOpenGL::validate(const int n, bool write_img)
{
    if (dimensions == 1) {
#if defined(_AMD)
        if (n > power2(23))
            return false;
#endif
        return gl_validate(n) == 1;
    }
    else {
#if defined(_AMD)
        if (n > power2(11))
            return false;
#endif
        return gl_2d_validate(n, write_img) == 1;
    }
}

void MyOpenGL::runPerformance(const int n)
{
    double time;
    if (dimensions == 1) {
#if defined(_AMD)
        if (n > power2(23)){
            results.push_back(-1);
            return;
        }
#endif
        time = gl_performance(n);
    }
    else {
#if defined(_AMD)
        if (n > power2(12)){
            results.push_back(-1);
            return;
        }
#endif
        time = gl_2d_performance(n);
    }
    results.push_back(time);
}