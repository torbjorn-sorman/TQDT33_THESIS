#include "dx_fft.h"

__inline void dx_fft(transform_direction dir, dx_args *args, const int n);
__inline void dx_fft_2d(transform_direction dir, dx_args *args, const int n);

//
// Testing
//

int dx_validate(const int n)
{
    cpx *in = get_seq(n, 1);
    cpx *out = get_seq(n);
    cpx *ref = get_seq(n, in);
    dx_args args;
    dx_setup(&args, in, n);

    dx_fft(FFT_FORWARD, &args, n);
    dx_read_buffer(&args, args.buf_output, out, n);
    double forward_diff = diff_forward_sinus(out, n);

    swap_io(&args);
    dx_fft(FFT_INVERSE, &args, n);
    dx_read_buffer(&args, args.buf_output, out, n);
    double inverse_diff = diff_seq(out, ref, n);

    dx_shakedown(&args);
    free(in);
    free(out);
    free(ref);
    return forward_diff < RELATIVE_ERROR_MARGIN && inverse_diff < RELATIVE_ERROR_MARGIN;
}

int dx_2d_validate(const int n, bool write_img)
{
    cpx *data, *ref;
    setup_seq2D(&data, NULL, &ref, n);
    dx_args args;
    dx_setup_2d(&args, data, n);

    dx_fft_2d(FFT_FORWARD, &args, n);    
    if (write_img) {
        dx_read_buffer(&args, args.buf_output, data, n * n);
        write_normalized_image("DirectX", "freq", data, n, true);
    }
    swap_io(&args);

    dx_fft_2d(FFT_INVERSE, &args, n);
    dx_read_buffer(&args, args.buf_output, data, n);
    if (write_img)
        write_image("DirectX", "spat", data, n);

    dx_shakedown(&args);
    double diff = diff_seq(data, ref, n);
    free(data);
    free(ref);
    return diff < RELATIVE_ERROR_MARGIN;
}

#ifndef MEASURE_BY_TIMESTAMP
double dx_performance(const int n)
{
    double measures[NUM_TESTS];
    dx_args args;
    for (int i = 0; i < NUM_TESTS; ++i) {
        dx_setup(&args, NULL, n);
        startTimer();
        dx_fft(FFT_FORWARD, &args, n);
        measures[i] = stopTimer();
        dx_shakedown(&args);
    }
    return average_best(measures, NUM_TESTS);
}
double dx_2d_performance(const int n)
{
    double measures[NUM_TESTS];
    dx_args args;
    for (int i = 0; i < NUM_TESTS; ++i) {
        dx_setup_2d(&args, NULL, n);
        startTimer();
        dx_fft_2d(FFT_FORWARD, &args, n);
        measures[i] = stopTimer();
        dx_shakedown(&args);
    }
    return average_best(measures, NUM_TESTS);
}
#else
double dx_performance(const int n)
{
    dx_args args;
    profiler_data profiler[NUM_TESTS];
    dx_setup(&args, NULL, n);
    for (int i = 0; i < NUM_TESTS; ++i) {
        profiler_data p;
        dx_start_profiling(&args, &p);

        dx_fft(FFT_FORWARD, &args, n);

        dx_end_profiling(&args, &p);        
        profiler[i] = p;
    }
    dx_shakedown(&args);
    return dx_avg(profiler, &args);
}
double dx_2d_performance(const int n)
{
    dx_args args;
    dx_setup_2d(&args, NULL, n);
    profiler_data profiler[NUM_TESTS];
    for (int i = 0; i < NUM_TESTS; ++i) {
        profiler_data p;
        dx_start_profiling(&args, &p);

        dx_fft_2d(FFT_FORWARD, &args, n);

        dx_end_profiling(&args, &p);
        profiler[i] = p;
    }
    dx_shakedown(&args);
    return dx_avg(profiler, &args);
}
#endif

//
// Algorithm
//

__inline void dx_set_buffers(dx_args *a)
{
    a->context->CSSetShaderResources(0, 1, &a->view_nullptr);
    a->context->CSSetUnorderedAccessViews(0, 1, &a->buf_output_uav, &a->init_cnts);
    a->context->CSSetShaderResources(0, 1, &a->buf_input_srv);
}

__inline void dx_set_local_args(dx_args *a, float global_angle, float local_angle, int steps_left, int leading_bits, int steps_gpu, float scalar, int number_of_blocks, int block_range_half)
{
    dx_cs_args cb = { global_angle, local_angle, scalar, steps_left, leading_bits, steps_gpu, number_of_blocks, block_range_half, 0, 0, 0 };
    dx_map_args<dx_cs_args>(a->context, a->buf_constant, &cb);
    dx_set_buffers(a);
    a->context->CSSetShader(a->cs_local, nullptr, 0);
}

__inline void dx_set_global_args(dx_args *a, float angle, int dist, unsigned int lmask, int steps)
{
    dx_cs_args cb = { angle, 0.f, 0.f, 0, 0, 0, 0, 0, steps, lmask, dist };
    dx_map_args<dx_cs_args>(a->context, a->buf_constant, &cb);
    dx_set_buffers(a);
    a->context->CSSetShader(a->cs_global, nullptr, 0);
}

__inline void dx_fft(transform_direction dir, dx_args *args, const int n)
{
    int n_half = (n >> 1);
    int steps_left = log2_32(n);
    int leading_bits = 32 - steps_left;
    int steps_gpu = log2_32(MAX_BLOCK_SIZE);
    float scalar = (dir == FFT_FORWARD ? 1.f : 1.f / n);
    int number_of_blocks = n_half > MAX_BLOCK_SIZE ? (n_half / MAX_BLOCK_SIZE) : 1;;
    int n_per_block = n / number_of_blocks;
    float global_angle = dir * (M_2_PI / n);
    float local_angle = dir * (M_2_PI / n_per_block);
    int block_range_half = n_half;
    if (number_of_blocks > 1) {
        int steps = 0;
        int dist = n;
        while (--steps_left > steps_gpu) {
            dx_set_global_args(args, global_angle, dist >>= 1, 0xFFFFFFFF << steps_left, steps++);
            args->context->Dispatch(args->n_groups.x, 1, 1);
            swap_io(args);
        }
        ++steps_left;
        block_range_half = n_per_block >> 1;
    }
    dx_set_local_args(args, global_angle, local_angle, steps_left, leading_bits, steps_gpu, scalar, 1, block_range_half);
    args->context->Dispatch(args->n_groups.x, 1, 1);
}

__inline void dx_fft_2d_helper(transform_direction dir, dx_args *args, const int n)
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
        dx_cs_args cb = { global_angle, 0.f, 0.f, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF << steps_left, n };
        args->context->CSSetShader(args->cs_global, nullptr, 0);
        while (--steps_left > steps_gpu) {
            cb.dist >>= 1;
            cb.lmask = 0xFFFFFFFF << steps_left;
            dx_map_args<dx_cs_args>(args->context, args->buf_constant, &cb);
            dx_set_buffers(args);
            args->context->Dispatch(args->n_groups.x, args->n_groups.y, 1);
            swap_io(args);
            ++cb.steps;
        }
        ++steps_left;
        block_range = n_per_block;
    }
    dx_set_local_args(args, global_angle, local_angle, steps_left, leading_bits, 0, scalar, 0, block_range >> 1);
    args->context->Dispatch(args->n_groups.x, args->n_groups.y, 1);
}

__inline void dx_fft_2d(transform_direction dir, dx_args *args, const int n)
{
    UINT width = n > TILE_DIM ? (n / TILE_DIM) : 1;

    dx_fft_2d_helper(dir, args, n);

    swap_io(args);
    dx_set_buffers(args);
    args->context->CSSetShader(args->cs_transpose, nullptr, 0);
    args->context->Dispatch(width, width, 1);

    swap_io(args);
    dx_fft_2d_helper(dir, args, n);

    swap_io(args);
    dx_set_buffers(args);
    args->context->CSSetShader(args->cs_transpose, nullptr, 0);
    args->context->Dispatch(width, width, 1);
}