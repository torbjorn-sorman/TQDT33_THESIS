#include "gl_fft.h"
#include "../../Common/mytimer.h"

__inline void gl_fft(transform_direction dir, gl_args *a_dev, gl_args *a_host, const int n);
__inline void gl_fft_2d(transform_direction dir, gl_args *a_dev, gl_args* a_host, gl_args* a_trans, const int n);

int gl_group_size()
{
    switch (vendor_gpu)
    {
    case VENDOR_NVIDIA: return 1024;
    case VENDOR_AMD:    return 256;
    default:            return 256;
    }
}

int gl_tile_dim()
{
    switch (vendor_gpu)
    {
    case VENDOR_NVIDIA: return 32;
    case VENDOR_AMD:    return 64;
    default:            return 32;
    }
}

//
// Testing
//

int gl_validate(const int n)
{
    cpx *in = get_seq(n, 1);
    cpx *out = get_seq(n);
    cpx *ref = get_seq(n, in);
    double forward_diff, inverse_diff;
    {
        gl_args a_dev, a_host;
        gl_setup(&a_dev, &a_host, in, NULL, gl_group_size(), n);
        gl_fft(FFT_FORWARD, &a_dev, &a_host, n);
        gl_read_buffer(out, a_dev.buf_out, n);
        forward_diff = diff_forward_sinus(out, n);
        gl_shakedown(&a_dev, &a_host);
    }
    {
        gl_args a_dev, a_host;
        gl_setup(&a_dev, &a_host, out, NULL, gl_group_size(), n);
        gl_fft(FFT_INVERSE, &a_dev, &a_host, n);
        gl_read_buffer(out, a_dev.buf_out, n);
        inverse_diff = diff_seq(out, ref, n);
        gl_shakedown(&a_dev, &a_host);
    }
    free_all(in, out, ref);    
    return forward_diff < RELATIVE_ERROR_MARGIN && inverse_diff < RELATIVE_ERROR_MARGIN;
}

int gl_2d_validate(const int n, bool write_img)
{
    cpx *data, *tmp, *ref;
    setup_seq_2d(&data, &tmp, &ref, n);
    {
        gl_args a_dev, a_host, a_trans;
        gl_setup_2d(&a_dev, &a_host, &a_trans, data, tmp, gl_group_size(), gl_tile_dim(), n);
        memset(data, 0, sizeof(cpx) * n * n);
        gl_fft_2d(FFT_FORWARD, &a_dev, &a_host, &a_trans, n);
        gl_read_buffer(data, a_dev.buf_out, n * n);
        if (write_img) {
            gl_read_buffer(tmp, a_dev.buf_in, n * n);            
            write_normalized_image("OpenGL", "freq", data, n, true);                 
        }
        gl_shakedown(&a_dev, &a_host, &a_trans);
    }
    {
        gl_args a_dev, a_host, a_trans;
        gl_setup_2d(&a_dev, &a_host, &a_trans, data, tmp, gl_group_size(), gl_tile_dim(), n);
        memset(data, 0, sizeof(cpx) * n * n);
        gl_fft_2d(FFT_INVERSE, &a_dev, &a_host, &a_trans, n);
        gl_read_buffer(data, a_dev.buf_out, n * n);
        if (write_img) {
            write_image("OpenGL", "spat", data, n);
        }
        gl_shakedown(&a_dev, &a_host, &a_trans);
    }
    double diff = diff_seq(data, ref, n);
    free_all(data, tmp, ref);
    return diff < RELATIVE_ERROR_MARGIN;
}

#ifndef MEASURE_BY_TIMESTAMP
double gl_performance(const int n)
{
    double measures[number_of_tests];
    gl_args a_dev, a_host;    
    for (int i = 0; i < number_of_tests; ++i) {
        gl_setup(&a_dev, &a_host, NULL, NULL, gl_group_size(), n);
        start_timer();
        gl_fft(FFT_FORWARD, &a_dev, &a_host, n);
        glFinish();
        measures[i] = stop_timer();
        gl_shakedown(&a_dev, &a_host);       
    }
    return average_best(measures, number_of_tests);
}
double gl_2d_performance(const int n)
{
    double measures[number_of_tests];
    gl_args a_dev, a_host, a_trans;
    for (int i = 0; i < number_of_tests; ++i) {
        gl_setup_2d(&a_dev, &a_host, &a_trans, NULL, NULL, gl_group_size(), gl_tile_dim(), n);
        start_timer();
        gl_fft_2d(FFT_INVERSE, &a_dev, &a_host, &a_trans, n);
        glFinish();
        measures[i] = stop_timer();
        gl_shakedown(&a_dev, &a_host);
    }
    return average_best(measures, number_of_tests);
}
#else
double gl_performance(const int n)
{
    gl_args a_dev, a_host;
    GLuint queries[64][2];
    gl_setup(&a_dev, &a_host, NULL, NULL, gl_group_size(), n);
    for (int i = 0; i < number_of_tests; ++i) {
        glGenQueries(2, queries[i]);
        glQueryCounter(queries[i][0], GL_TIMESTAMP);
        gl_fft(FFT_FORWARD, &a_dev, &a_host, n);
        glQueryCounter(queries[i][1], GL_TIMESTAMP);
    }
    gl_shakedown(&a_dev, &a_host);
    return gl_query_time(queries);
}
double gl_2d_performance(const int n)
{
    gl_args a_dev, a_host, a_trans;
    GLuint queries[64][2];
    gl_setup_2d(&a_dev, &a_host, &a_trans, NULL, NULL, gl_group_size(), gl_tile_dim(), n);
    for (int i = 0; i < number_of_tests; ++i) {
        glGenQueries(2, queries[i]);
        glQueryCounter(queries[i][0], GL_TIMESTAMP);
        gl_fft_2d(FFT_FORWARD, &a_dev, &a_host, &a_trans, n);
        glQueryCounter(queries[i][1], GL_TIMESTAMP);
    }
    gl_shakedown(&a_dev, &a_host, &a_trans);
    return gl_query_time(queries);
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

__inline void gl_set_local_args(gl_args *a, float local_angle, unsigned int steps_left, unsigned int leading_bits, float scalar, unsigned int block_range)
{
    glUniform1f(glGetUniformLocation(a->program, "local_angle"), local_angle);
    glUniform1ui(glGetUniformLocation(a->program, "steps_left"), steps_left);
    glUniform1f(glGetUniformLocation(a->program, "scalar"), scalar);
    glUniform1ui(glGetUniformLocation(a->program, "leading_bits"), leading_bits);
    glUniform1ui(glGetUniformLocation(a->program, "block_range"), block_range);
}

__inline void gl_set_global_args(gl_args *a, float global_angle, unsigned int dist, unsigned int lmask, unsigned int steps)
{
    glUniform1f(glGetUniformLocation(a->program, "global_angle"), global_angle);
    glUniform1ui(glGetUniformLocation(a->program, "dist"), dist);
    glUniform1ui(glGetUniformLocation(a->program, "lmask"), lmask);
    glUniform1ui(glGetUniformLocation(a->program, "steps"), steps);
}

__inline void gl_fft(transform_direction dir, gl_args *a_dev, gl_args* a_host, const int n)
{
    fft_args args;
    set_fft_arguments(&args, dir, a_dev->number_of_blocks, gl_group_size(), n);
    if (a_dev->number_of_blocks > 1) {
        glUseProgram(a_host->program);
        gl_bind_io_buffers(a_host);
        while (--args.steps_left > args.steps_gpu) {   
            gl_set_global_args(a_host, args.global_angle, args.dist >>= 1, 0xFFFFFFFF << args.steps_left, args.steps++);
            glDispatchCompute(a_host->groups.x, a_host->groups.y, a_host->groups.z);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
        ++args.steps_left;
    }
    glUseProgram(a_dev->program);
    gl_bind_io_buffers(a_dev);
    gl_set_local_args(a_dev, args.local_angle, args.steps_left, args.leading_bits, args.scalar, args.block_range);
    glDispatchCompute(a_dev->groups.x, a_dev->groups.y, a_dev->groups.z);
}

__inline void gl_fft_2d(transform_direction dir, gl_args *a_dev, gl_args* a_host, gl_args* a_trans, const int n)
{
    UINT group_dim = n > gl_tile_dim() ? (n / gl_tile_dim()) : 1;

    gl_fft(dir, a_dev, a_host, n);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    //return;
    gl_swap_buffers(a_dev, a_host, a_trans);
    glUseProgram(a_trans->program);
    gl_bind_io_buffers(a_trans);
    glDispatchCompute(group_dim, group_dim, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    gl_swap_buffers(a_dev, a_host, a_trans);
    gl_fft(dir, a_dev, a_host, n);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    gl_swap_buffers(a_dev, a_host, a_trans);
    glUseProgram(a_trans->program);
    gl_bind_io_buffers(a_trans);
    glDispatchCompute(group_dim, group_dim, 1);
}