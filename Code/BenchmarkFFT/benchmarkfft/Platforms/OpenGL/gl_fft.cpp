#include "gl_fft.h"

__inline void gl_fft(transform_direction dir, gl_args *args_local, gl_args *args_global, const int n);
__inline void gl_fft_2d(transform_direction dir, gl_args *args, const int n);

#define GL_GROUP_SIZE 1024

//
// Testing
//

int gl_validate(const int n)
{
    cpx *in = get_seq(n, 1);
    cpx *out = 0,
        *tmp = 0;
    cpx *ref = get_seq(n, in);
    gl_args args_local, args_global;
    gl_setup(&args_local, &args_global, in, NULL, GL_GROUP_SIZE, n);

    gl_fft(FFT_FORWARD, &args_local, &args_global, n);
    gl_read_buffer(args_local.buf_out, &out, n);
    double forward_diff = diff_forward_sinus(out, n);

    gl_swap_buffers(&args_local, &args_global);
    gl_fft(FFT_INVERSE, &args_local, &args_global, n);
    out = 0;
    gl_read_buffer(args_local.buf_out, &out, n);
    double inverse_diff = diff_seq(out, ref, n);

#if 0 && defined(SHOW_BLOCKING_DEBUG)
    cpx_to_console(in, "GL In", 32);
    cpx_to_console(out, "GL Out", 32);
    printf("%f and %f\n", forward_diff, inverse_diff);
    getchar();
#endif

    gl_shakedown(&args_local);
    gl_shakedown(&args_global);
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
    gl_args a_local, a_global;
    GLuint queries[NUM_TESTS][2];

    //profiler_data profiler[NUM_TESTS];
    gl_setup(&a_local, &a_global, NULL, NULL, GL_GROUP_SIZE, n);
    for (int i = 0; i < NUM_TESTS; ++i) {
        glGenQueries(2, queries[i]);
        glQueryCounter(queries[i][0], GL_TIMESTAMP);

        gl_fft(FFT_FORWARD, &a_local, &a_global, n);

        glQueryCounter(queries[i][1], GL_TIMESTAMP);
    }
    gl_shakedown(&a_local);
    gl_shakedown(&a_global);
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

__inline void gl_bind_io_buffers(gl_args *a)
{
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, a->buf_in);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, a->buf_out);
}

__inline void gl_set_local_args(gl_args *a, float local_angle, unsigned int steps_left, unsigned int leading_bits, float scalar, unsigned int block_range_half)
{
    glUniform1f(glGetUniformLocation(a->program, "local_angle"), local_angle);
    glUniform1ui(glGetUniformLocation(a->program, "steps_left"), steps_left);
    glUniform1f(glGetUniformLocation(a->program, "scalar"), scalar);
    glUniform1ui(glGetUniformLocation(a->program, "leading_bits"), leading_bits);
    glUniform1ui(glGetUniformLocation(a->program, "block_range_half"), block_range_half);
}

__inline void gl_set_global_args(gl_args *a, float global_angle, unsigned int dist, unsigned int lmask, unsigned int steps)
{
    glUniform1f(glGetUniformLocation(a->program, "global_angle"), global_angle);
    glUniform1ui(glGetUniformLocation(a->program, "dist"), dist);
    glUniform1ui(glGetUniformLocation(a->program, "lmask"), lmask);
    glUniform1ui(glGetUniformLocation(a->program, "steps"), steps);
}

__inline void gl_fft(transform_direction dir, gl_args *a_local, gl_args* a_global, const int n)
{
    fft_args args;
    int n_half = (n >> 1);
    int number_of_blocks = n_half > GL_GROUP_SIZE ? (n_half / GL_GROUP_SIZE) : 1;
    set_fft_arguments(&args, dir, number_of_blocks, GL_GROUP_SIZE, n);
    if (number_of_blocks > 1) {
        glUseProgram(a_global->program);
        while (--args.steps_left > args.steps_gpu) {
            gl_set_global_args(a_global, args.global_angle, args.dist >>= 1, 0xFFFFFFFF << args.steps_left, args.steps++);
            glDispatchCompute(a_global->groups.x, a_global->groups.y, a_global->groups.z);
        }
        ++args.steps_left;
    }
    glUseProgram(a_local->program);
    gl_set_local_args(a_local, args.local_angle, args.steps_left, args.leading_bits, args.scalar, args.block_range_half);
    glDispatchCompute(a_local->groups.x, a_local->groups.y, a_local->groups.z);
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