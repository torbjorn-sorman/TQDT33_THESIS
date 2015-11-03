#pragma once
#ifndef GL_HELPER_H
#define GL_HELPER_H


#include "glew/glew.h"
#include "glew/wglew.h"

#include "freeglut/freeglut.h"
#include "freeglut/freeglut_ext.h"
#include "freeglut/freeglut_std.h"

#include "../../Definitions.h"
#include "../../Common/myfile.h"
#include "../../Common/mycomplex.h"

struct gl_args {
    GLuint program;
    GLuint buf_in;
    GLuint buf_out;
    GLuint number_of_buffers = 2;
    char *shader_src;
};

void gl_swap_io(gl_args* a);
void gl_setup(gl_args* args, cpx* data, const int n);
void gl_read_buffer(GLuint buffer, cpx* data, const int n);
void gl_shakedown(gl_args *a);

#endif