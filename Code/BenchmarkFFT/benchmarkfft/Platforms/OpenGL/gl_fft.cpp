#include "gl_fft.h"

#include <stdio.h>
#include <string>
#include "../../Common/cpx_debug.h"

__inline void gl_fft(transform_direction dir, gl_args *args, const int n);
__inline void gl_fft_2d(transform_direction dir, gl_args *args, const int n);


#define GL_GROUP_SIZE 1024

//
// Testing
//

int gl_validate(const int n)
{
    cpx *in = get_seq(n, 1);
    cpx *out = 0;
    cpx *ref = get_seq(n, in);
    gl_args args;
    gl_setup(&args, in, NULL, GL_GROUP_SIZE, n);

    gl_fft(FFT_FORWARD, &args, n);
    gl_read_buffer(args.buf_out, &out, n);
    double forward_diff = diff_forward_sinus(out, n);

#ifdef SHOW_BLOCKING_DEBUG
    cpx_to_console(in, "Buf In", n);
    cpx_to_console(out, "Buf Out", n);
    getchar();
#endif

    gl_swap_io(&args);
    gl_fft(FFT_INVERSE, &args, n);
    gl_read_buffer(args.buf_out, &out, n);
    double inverse_diff = diff_seq(out, ref, n);

    gl_shakedown(&args);
    free(in);
    free(ref);
    return forward_diff < RELATIVE_ERROR_MARGIN && inverse_diff < RELATIVE_ERROR_MARGIN;
}

int gl_2d_validate(const int n, bool write_img)
{
    /*
    cpx *data, *ref;
    setup_seq2D(&data, NULL, &ref, n);
    gl_args args;
    gl_setup_2d(&args, data, n);

    gl_fft_2d(FFT_FORWARD, &args, n);
    if (write_img) {
    gl_read_buffer(&args, args.buf_output, data, n * n);
    write_normalized_image("DirectX", "freq", data, n, true);
    }
    swap_io(&args);

    gl_fft_2d(FFT_INVERSE, &args, n);
    gl_read_buffer(&args, args.buf_output, data, n);
    if (write_img)
    write_image("DirectX", "spat", data, n);

    gl_shakedown(&args);
    double diff = diff_seq(data, ref, n);
    free(data);
    free(ref);
    return diff < RELATIVE_ERROR_MARGIN;
    */
    return 0;
}

#ifndef MEASURE_BY_TIMESTAMP
double gl_performance(const int n)
{
    double measures[NUM_TESTS];
    gl_args args;
    for (int i = 0; i < NUM_TESTS; ++i) {
        gl_setup(&args, NULL, n);
        startTimer();
        gl_fft(FFT_FORWARD, &args, n);
        measures[i] = stopTimer();
        gl_shakedown(&args);
    }
    return average_best(measures, NUM_TESTS);
}
double gl_2d_performance(const int n)
{
    double measures[NUM_TESTS];
    gl_args args;
    for (int i = 0; i < NUM_TESTS; ++i) {
        gl_setup_2d(&args, NULL, n);
        startTimer();
        gl_fft_2d(FFT_FORWARD, &args, n);
        measures[i] = stopTimer();
        gl_shakedown(&args);
    }
    return average_best(measures, NUM_TESTS);
}
#else
double gl_performance(const int n)
{
    gl_args args;    
    GLuint queries[NUM_TESTS][2];
    
    //profiler_data profiler[NUM_TESTS];
    gl_setup(&args, NULL, NULL, GL_GROUP_SIZE, n);
    for (int i = 0; i < NUM_TESTS; ++i) {
        glGenQueries(2, queries[i]);
        glQueryCounter(queries[i][0], GL_TIMESTAMP);

        gl_fft(FFT_FORWARD, &args, n);

        glQueryCounter(queries[i][1], GL_TIMESTAMP);
    }
    gl_shakedown(&args);
    return gl_query_time(queries);
}
double gl_2d_performance(const int n)
{
    /*
    gl_args args;
    gl_setup_2d(&args, NULL, n);
    profiler_data profiler[NUM_TESTS];
    for (int i = 0; i < NUM_TESTS; ++i) {
    profiler_data p;
    gl_start_profiling(&args, &p);

    gl_fft_2d(FFT_FORWARD, &args, n);

    gl_end_profiling(&args, &p);
    profiler[i] = p;
    }
    gl_shakedown(&args);
    return gl_avg(profiler, &args);
    */
    return -1;
}
#endif

//
// Algorithm
//
/*
__inline void gl_set_buffers(gl_args *a)
{
a->context->CSSetShaderResources(0, 1, &a->view_nullptr);
a->context->CSSetUnorderedAccessViews(0, 1, &a->buf_output_uav, &a->init_cnts);
a->context->CSSetShaderResources(0, 1, &a->buf_input_srv);
}
*/
__inline void gl_set_local_args(gl_args *a, float local_angle, unsigned int steps_left, unsigned int leading_bits, float scalar, unsigned int block_range_half)
{
    glUniform1f(glGetUniformLocation(a->program, "local_angle"), local_angle);
    glUniform1ui(glGetUniformLocation(a->program, "steps_left"), steps_left);
    glUniform1f(glGetUniformLocation(a->program, "scalar"), scalar);
    GLuint loc = glGetUniformLocation(a->program, "leading_bits");
    glUniform1ui(loc, leading_bits);
    glUniform1ui(glGetUniformLocation(a->program, "block_range_half"), block_range_half);
}
/*
__inline void gl_set_global_args(gl_args *a, float angle, int dist, unsigned int lmask, int steps)
{
gl_cs_args cb = { angle, 0.f, 0.f, 0, 0, 0, 0, 0, steps, lmask, dist };
gl_map_args<gl_cs_args>(a->context, a->buf_constant, &cb);
gl_set_buffers(a);
a->context->CSSetShader(a->cs_global, nullptr, 0);
}
*/

static int tab32[32] = {
    0, 9, 1, 10, 13, 21, 2, 29,
    11, 14, 16, 18, 22, 25, 3, 30,
    8, 12, 20, 28, 15, 17, 24, 7,
    19, 27, 23, 6, 26, 5, 4, 31
};

// Integer, base 2 log
static __inline int log2_32(int value)
{
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return tab32[(unsigned int)(value * 0x07C4ACDD) >> 27];
}

__inline void gl_fft(transform_direction dir, gl_args *args, const int n)
{
    int n_half = (n >> 1);
    int steps_left = log2_32(n);
    int leading_bits = 32 - steps_left;
    int steps_gpu = log2_32(GL_GROUP_SIZE);
    float scalar = (dir == FFT_FORWARD ? 1.f : 1.f / n);
    int number_of_blocks = n_half > GL_GROUP_SIZE ? (n_half / GL_GROUP_SIZE) : 1;;
    int n_per_block = n / number_of_blocks;
    float global_angle = dir * (M_2_PI / n);
    float local_angle = dir * (M_2_PI / n_per_block);
    int block_range_half = n_half;
    /*
    if (number_of_blocks > 1) {
    int steps = 0;
    int dist = n;
    while (--steps_left > steps_gpu) {
    gl_set_global_args(args, global_angle, dist >>= 1, 0xFFFFFFFF << steps_left, steps++);
    args->context->Dispatch(args->n_groups.x, 1, 1);
    //swap_io(args);
    }
    ++steps_left;
    block_range_half = n_per_block >> 1;
    }
    */
    glUseProgram(args->program);
    gl_set_local_args(args, local_angle, steps_left, leading_bits, scalar, block_range_half);
    glDispatchCompute(args->groups.x, 1, 1);
}
/*
__inline void gl_fft_2d_helper(transform_direction dir, gl_args *args, const int n)
{
int steps_left = log2_32(n);
int leading_bits = 32 - steps_left;
float scalar = (dir == FFT_FORWARD ? 1.f : 1.f / n);
const int n_per_block = n / args->n_groups.y;
const float global_angle = dir * (M_2_PI / n);
const float local_angle = dir * (M_2_PI / n_per_block);
int block_range = n;
if (args->n_groups.y > 1) {
const int steps_gpu = log2_32(MAX_BLOCK_SIZE);
gl_cs_args cb = { global_angle, 0.f, 0.f, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF << steps_left, n };
args->context->CSSetShader(args->cs_global, nullptr, 0);
while (--steps_left > steps_gpu) {
cb.dist >>= 1;
cb.lmask = 0xFFFFFFFF << steps_left;
gl_map_args<gl_cs_args>(args->context, args->buf_constant, &cb);
gl_set_buffers(args);
args->context->Dispatch(args->n_groups.x, args->n_groups.y, 1);
swap_io(args);
++cb.steps;
}
++steps_left;
block_range = n_per_block;
}
gl_set_local_args(args, global_angle, local_angle, steps_left, leading_bits, 0, scalar, 0, block_range >> 1);
args->context->Dispatch(args->n_groups.x, args->n_groups.y, 1);
}

__inline void gl_fft_2d(transform_direction dir, gl_args *args, const int n)
{
UINT width = n > TILE_DIM ? (n / TILE_DIM) : 1;

gl_fft_2d_helper(dir, args, n);

swap_io(args);
gl_set_buffers(args);
args->context->CSSetShader(args->cs_transpose, nullptr, 0);
args->context->Dispatch(width, width, 1);

swap_io(args);
gl_fft_2d_helper(dir, args, n);

swap_io(args);
gl_set_buffers(args);
args->context->CSSetShader(args->cs_transpose, nullptr, 0);
args->context->Dispatch(width, width, 1);
}
*/