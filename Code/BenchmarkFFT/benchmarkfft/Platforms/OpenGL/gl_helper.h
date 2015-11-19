#pragma once
#ifndef GL_HELPER_H
#define GL_HELPER_H

#include "glew/glew.h"
#include "freeglut/freeglut.h"
#include "../../Definitions.h"
#include "../../Common/myfile.h"
#include "../../Common/mycomplex.h"

struct gl_args {
    GLuint program;
    GLuint shader;
    GLuint buf_in;
    GLuint buf_out;
    dim3 groups = dim3{ 1, 1, 1 };
    dim3 threads = dim3{ 1, 1, 1 };
    int number_of_blocks = 1;
    int tile_dim = 32;
    char *shader_src;
};

static __inline void gl_swap_buffers(gl_args *a_l, gl_args *a_g, gl_args *a_t)
{
    GLuint buf_i = a_l->buf_in;
    GLuint buf_o = a_l->buf_out;
    a_t->buf_in = a_g->buf_in = a_l->buf_in = buf_o;
    a_t->buf_out = a_g->buf_out = a_l->buf_out = buf_i;
}

static __inline void gl_swap_buffers(gl_args *a_l, gl_args *a_g)
{
    GLuint buf_i = a_l->buf_in;
    GLuint buf_o = a_l->buf_out;
    a_g->buf_in = a_l->buf_in = buf_o;
    a_g->buf_out = a_l->buf_out = buf_i;
}

void gl_check_errors();
void gl_load_buffer(GLuint buffer, cpx* data, const int binding, const int n);
void gl_swap_io(gl_args* a);
double gl_query_time(unsigned int q[64][2]);
void gl_setup(gl_args* a_dev, gl_args* a_host, cpx* in, cpx* out, const int groups_size, const int n);
void gl_setup_2d(gl_args* a_dev, gl_args* a_host, gl_args* a_trans, cpx* in, cpx* out, int group_size, int tile_dim, const int n);
void gl_read_buffer(cpx* dst, GLuint buffer, const int n);
void gl_shakedown(gl_args *a);

template<typename... Args>
void gl_shakedown(gl_args *a, Args... args)
{
    gl_shakedown(a);
    gl_shakedown(args...);
}

void gl_get_adapter(int gpuIndex);

#endif