#include "gl_fft.h"

__inline void gl_fft(transform_direction dir, gl_args *a_dev, gl_args *a_host, const int n);
__inline void gl_fft_2d(transform_direction dir, gl_args *args, const int n);

#define GL_GROUP_SIZE 1024
#define GL_TILE_DIM 64

//
// Testing
//

int gl_validate(const int n)
{
    cpx *in = get_seq(n, 1);
    cpx *out = 0;
    cpx *ref = get_seq(n, in);
    gl_args a_dev, a_host;
    gl_setup(&a_dev, &a_host, in, NULL, GL_GROUP_SIZE, n);

    gl_fft(FFT_FORWARD, &a_dev, &a_host, n);
    gl_read_buffer(a_dev.buf_out, &out, n);
    double forward_diff = diff_forward_sinus(out, n);
    
#if defined(SHOW_BLOCKING_DEBUG)
    cpx_to_console(out, "GL FFT Out", 32);
    getchar();
#endif

    gl_swap_buffers(&a_dev, &a_host);
    gl_fft(FFT_INVERSE, &a_dev, &a_host, n);
    out = 0;
    gl_read_buffer(a_dev.buf_out, &out, n);
    double inverse_diff = diff_seq(out, ref, n);

#if defined(SHOW_BLOCKING_DEBUG)
    cpx_to_console(ref, "GL Ref", 32);
    cpx_to_console(out, "GL Out", 32);
    printf("%f and %f\n", forward_diff, inverse_diff);
    getchar();
#endif

    gl_shakedown(&a_dev);
    gl_shakedown(&a_host);
    free(in);
    free(ref);
    return forward_diff < RELATIVE_ERROR_MARGIN && inverse_diff < RELATIVE_ERROR_MARGIN;
}

int gl_2d_validate(const int n, bool write_img)
{
    /*
    cpx *data, *ref;
    setup_seq2D(&data, NULL, &ref, n);
    gl_args a_dev, a_host, a_trans;
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
    gl_args a_dev, a_host;
    GLuint queries[NUM_TESTS][2];

    //profiler_data profiler[NUM_TESTS];
    gl_setup(&a_dev, &a_host, NULL, NULL, GL_GROUP_SIZE, n);
    for (int i = 0; i < NUM_TESTS; ++i) {
        glGenQueries(2, queries[i]);
        glQueryCounter(queries[i][0], GL_TIMESTAMP);

        gl_fft(FFT_FORWARD, &a_dev, &a_host, n);

        glQueryCounter(queries[i][1], GL_TIMESTAMP);
    }
    gl_shakedown(&a_dev);
    gl_shakedown(&a_host);
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

__inline void gl_set_trans_args(gl_args *a, unsigned int n)
{    
    glUniform1ui(glGetUniformLocation(a->program, "n"), n);
}

__inline void gl_fft(transform_direction dir, gl_args *a_dev, gl_args* a_host, const int n)
{
    fft_args args;
    set_fft_arguments(&args, dir, a_dev->number_of_blocks, GL_GROUP_SIZE, n);    
    if (a_dev->number_of_blocks > 1) {
        glUseProgram(a_host->program);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, a_host->buf_in);
        while (--args.steps_left > args.steps_gpu) {
            gl_set_global_args(a_host, args.global_angle, args.dist >>= 1, 0xFFFFFFFF << args.steps_left, args.steps++);
            glDispatchCompute(a_host->groups.x, a_host->groups.y, a_host->groups.z);
        }
        ++args.steps_left;
    }
    glUseProgram(a_dev->program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, a_dev->buf_in);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, a_dev->buf_out);    
    gl_set_local_args(a_dev, args.local_angle, args.steps_left, args.leading_bits, args.scalar, args.block_range_half);
    glDispatchCompute(a_dev->groups.x, a_dev->groups.y, a_dev->groups.z);
}

__inline void gl_fft_2d(transform_direction dir, gl_args *a_dev, gl_args* a_host, gl_args* a_trans, const int n)
{
    UINT width = n > GL_TILE_DIM ? (n / GL_TILE_DIM) : 1;

    gl_fft(dir, a_dev, a_host, n);    
    gl_swap_io(a_trans);
    gl_bind_io_buffers(a_trans);
    glUseProgram(a_trans->program);
    gl_set_trans_args(a_trans, n);
    glDispatchCompute(width, width, 1);

    gl_fft(dir, a_dev, a_host, n);
    gl_swap_io(a_trans);
    gl_bind_io_buffers(a_trans);
    glUseProgram(a_trans->program);
    gl_set_trans_args(a_trans, n);
    glDispatchCompute(width, width, 1);
}