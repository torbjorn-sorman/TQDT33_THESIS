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
    dim3 groups = { 1, 1, 1 };
    dim3 threads = { 1, 1, 1 };
    int number_of_blocks = 1;
    char *shader_src;
};

static __inline void gl_swap_buffers(gl_args *a_l, gl_args *a_g)
{
    GLuint buf_i = a_l->buf_in;
    GLuint buf_o = a_l->buf_out;
    a_g->buf_in = a_l->buf_in = buf_o;
    a_g->buf_out = a_l->buf_out = buf_i;
}

void gl_load_buffer(GLuint buffer, cpx* data, const int binding, const int n);
void gl_swap_io(gl_args* a);
double gl_query_time(unsigned int q[NUM_TESTS][2]);
void gl_setup(gl_args* a_dev, gl_args* a_host, cpx* in, cpx* out, const int groups_size, const int n);
void gl_read_buffer(GLuint buffer, cpx** data, const int n);
void gl_shakedown(gl_args *a);

#endif