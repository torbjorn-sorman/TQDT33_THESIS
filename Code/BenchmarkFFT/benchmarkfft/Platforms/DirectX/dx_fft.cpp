#include "dx_fft.h"

__inline void dx_fft(transform_direction dir, dx_args *args, const int n);
__inline void dx_fft_2d(transform_direction dir, cpx **in, cpx **out, const int n);

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

#ifdef SHOW_BLOCKING_DEBUG  
    dx_read_buffer(&args, args.buf_input, in, n);
    cpx_to_console(in, "DX In", 8);
    cpx_to_console(out, "DX Out", 8);
    getchar();
#endif

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

int dx_2d_validate(const int n)
{
    cpx *in, *buf, *ref;
    setup_seq2D(&in, &buf, &ref, n);

    dx_fft_2d(FFT_FORWARD, &in, &buf, n);
    write_normalized_image("DirectX", "freq", in, n, true);
    dx_fft_2d(FFT_INVERSE, &in, &buf, n);
    write_image("DirectX", "spat", in, n);

    double diff = diff_seq(in, ref, n);
    free(in);
    free(buf);
    free(ref);
    return diff < RELATIVE_ERROR_MARGIN;
}

//#define DX_TIMESTAMP

#ifndef DX_TIMESTAMP
double dx_performance(const int n)
{
    double measures[NUM_PERFORMANCE];
    dx_args args;
    profiler_data profiler[NUM_PERFORMANCE];
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        dx_setup(&args, NULL, n);
        startTimer();
        dx_fft(FFT_FORWARD, &args, n);
        measures[i] = stopTimer();
        dx_shakedown(&args);
    }
    return average_best(measures, NUM_PERFORMANCE);
}
#else
double dx_performance(const int n)
{
    dx_args args;
    profiler_data profiler[NUM_PERFORMANCE];
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        profiler_data p;
        dx_setup(&args, NULL, n);
        dx_start_profiling(&args, &p);
        dx_fft(FFT_FORWARD, &args, n);
        dx_end_profiling(&args, &p);
        dx_shakedown(&args);
        profiler[i] = p;
    }
    return dx_avg(profiler, &args);
}
#endif

double dx_2d_performance(const int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in = get_seq(n * n);
    cpx *out = get_seq(n * n);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        dx_fft_2d(FFT_FORWARD, &in, &out, n);
        measures[i] = stopTimer();
    }
    free(in);
    free(out);
    return average_best(measures, NUM_PERFORMANCE);
}

//
// Algorithm
//

__inline void dx_set_local_args(dx_args *a, float global_angle, float local_angle, int steps_left, int leading_bits, int steps_gpu, float scalar, int number_of_blocks, int block_range_half)
{
    dx_cs_args cb = { global_angle, local_angle, scalar, steps_left, leading_bits, steps_gpu, number_of_blocks, block_range_half, 0, 0, 0 };
    dx_map_args<dx_cs_args>(a->context, a->buf_constant, &cb);
    a->context->CSSetShaderResources(0, 1, &a->view_nullptr);
    a->context->CSSetUnorderedAccessViews(0, 1, &a->buf_output_uav, &a->init_cnts);
    /*
    if (number_of_blocks > 1 && number_of_blocks <= HW_LIMIT) {
    a->context->CSSetUnorderedAccessViews(1, 1, &a->buf_sin_uav, &a->init_cnts);
    a->context->CSSetUnorderedAccessViews(2, 1, &a->buf_sout_uav, &a->init_cnts);
    }
    */
    a->context->CSSetShaderResources(0, 1, &a->buf_input_srv);
    a->context->CSSetShader(a->cs_local, nullptr, 0);

}

__inline void dx_set_global_args(dx_args *a, float angle, int dist, unsigned int lmask, int steps)
{
    dx_cs_args cb = { angle, 0.f, 0.f, 0, 0, 0, 0, 0, steps, lmask, dist };
    dx_map_args<dx_cs_args>(a->context, a->buf_constant, &cb);
    a->context->CSSetShaderResources(0, 1, &a->view_nullptr);
    a->context->CSSetUnorderedAccessViews(0, 1, &a->buf_output_uav, &a->init_cnts);
    a->context->CSSetShaderResources(0, 1, &a->buf_input_srv);
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
    if (number_of_blocks > HW_LIMIT) {
        --steps_left;
        int steps = 0;
        int dist = n_half;
        dx_set_global_args(args, global_angle, dist, 0xFFFFFFFF << steps_left, steps);
        args->context->Dispatch(args->n_groups, 1, 1);
        swap_io(args);
        while (--steps_left > steps_gpu) {
            dist >>= 1;
            ++steps;
            dx_set_global_args(args, global_angle, dist, 0xFFFFFFFF << steps_left, steps);
            args->context->Dispatch(args->n_groups, 1, 1);
            swap_io(args);
        }
        ++steps_left;
        number_of_blocks = 1;
        block_range_half = n_per_block >> 1;
    }
    dx_set_local_args(args, global_angle, local_angle, steps_left, leading_bits, steps_gpu, scalar, number_of_blocks, block_range_half);
    args->context->Dispatch(args->n_groups, 1, 1);
}

__inline void dx_fft_2d(transform_direction dir, cpx **in, cpx **out, const int n)
{
    return;
}