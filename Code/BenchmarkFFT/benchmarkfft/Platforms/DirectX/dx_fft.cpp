#include "dx_fft.h"
#include "../../Common/cpx_debug.h"

__inline void dx_fft(transform_direction dir, dx_args *args, const int n);
__inline void dx_fft_2d(transform_direction dir, dx_args *args, const int n);

int dx_block_size()
{
    switch (vendor_gpu)
    {
    case VENDOR_AMD:    return 256;
    case VENDOR_NVIDIA: return 1024;
    default:            return 1;
    }
}

int dx_tile_dim()
{
    switch (vendor_gpu)
    {
    case VENDOR_AMD:    return 64;
    case VENDOR_NVIDIA: return 32;
    default:            return 1;
    }
}

int dx_block_dim()
{
    switch (vendor_gpu)
    {
    case VENDOR_AMD:
    case VENDOR_NVIDIA:
    default:            return 16;
    }
}

//
// Testing
//

int dx_validate(const int n)
{
    cpx *in, *out, *ref;
    dx_args args;
    setup_seq(&in, &out, &ref, batch_count(n), n);    
    dx_setup(&args, in, dx_block_size(), n);

    dx_fft(FFT_FORWARD, &args, n);
    dx_read_buffer(&args, args.buf_output, out, batch_size(n));
    double forward_diff = diff_forward_sinus(out, batch_count(n), n);

    swap_io(&args);
    dx_fft(FFT_INVERSE, &args, n);
    dx_read_buffer(&args, args.buf_output, out, batch_size(n));
    double inverse_diff = diff_seq(out, ref, batch_size(n));

    dx_shakedown(&args);
    free_all(in, out, ref);
    return forward_diff < RELATIVE_ERROR_MARGIN && inverse_diff < RELATIVE_ERROR_MARGIN;
}

int dx_2d_validate(const int n, bool write_img)
{
    cpx *data, *ref;
    setup_seq_2d(&data, NULL, &ref, n);
    dx_args args;
    dx_setup_2d(&args, data, dx_block_size(), dx_tile_dim(), n);
    dx_fft_2d(FFT_FORWARD, &args, n);
    if (write_img) {
        dx_read_buffer(&args, args.buf_output, data, n * n);
        write_normalized_image("DirectX", "freq", data, n, true);
    }

    swap_io(&args);

    dx_fft_2d(FFT_INVERSE, &args, n);
    dx_read_buffer(&args, args.buf_output, data, n * n);
    if (write_img) {
        write_image("DirectX", "spat", data, n);
    }

    dx_shakedown(&args);
    double diff = diff_seq(data, ref, n * n);
    free_all(data, ref);
    return diff < RELATIVE_ERROR_MARGIN;
}

#ifndef MEASURE_BY_TIMESTAMP
double dx_performance(const int n)
{
    double measures[number_of_tests];
    dx_args args;
    for (int i = 0; i < number_of_tests; ++i) {
        dx_setup(&args, NULL, n);
        start_timer();
        dx_fft(FFT_FORWARD, &args, n);
        measures[i] = stop_timer();
        dx_shakedown(&args);
    }
    return average_best(measures, number_of_tests);
}
double dx_2d_performance(const int n)
{
    double measures[number_of_tests];
    dx_args args;
    for (int i = 0; i < number_of_tests; ++i) {
        dx_setup_2d(&args, NULL, n);
        start_timer();
        dx_fft_2d(FFT_FORWARD, &args, n);
        measures[i] = stop_timer();
        dx_shakedown(&args);
    }
    return average_best(measures, number_of_tests);
}
#else
double dx_perf(void(*fft_fn)(transform_direction, dx_args*, const int), dx_args* args, const int n)
{
#if defined(_AMD)
    double measurements[64];
    for (int i = 0; i < number_of_tests; ++i) {
        profiler_data p;
        dx_start_profiling(args, &p);
        fft_fn(FFT_FORWARD, args, n);
        dx_end_profiling(args, &p);
        measurements[i] = dx_time_elapsed(&p, args->context);
    }
    return average_best(measurements, number_of_tests);
#elif defined(_NVIDIA)
    profiler_data p_data[64];
    for (int i = 0; i < number_of_tests; ++i) {
        profiler_data p;
        dx_start_profiling(args, &p);
        fft_fn(FFT_FORWARD, args, n);
        dx_end_profiling(args, &p);
        p_data[i] = p;        
    }
    return dx_avg(p_data, args);
#endif
}
double dx_performance(const int n)
{
    dx_args args;
    dx_setup(&args, NULL, dx_block_size(), n);
    double m = dx_perf(dx_fft, &args, n);
    dx_shakedown(&args);
    return m;
}
double dx_2d_performance(const int n)
{
    dx_args args;
    dx_setup_2d(&args, NULL, dx_block_size(), dx_tile_dim(), n);
    double m = dx_perf(dx_fft_2d, &args, n);
    dx_shakedown(&args);
    return m;
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

__inline void dx_set_args(dx_args *a, float global_angle, float local_angle, float scalar, int steps_left, int leading_bits, int steps_gpu, int number_of_blocks, int steps, unsigned int lmask, int dist)
{
    dx_cs_args cb = { global_angle, local_angle, scalar, steps_left, leading_bits, steps_gpu, number_of_blocks, steps, lmask, dist };
    dx_map_args<dx_cs_args>(a->context, a->buf_constant, &cb);
}

__inline void dx_fft(transform_direction dir, dx_args *a, const int n)
{
    fft_args args;
    set_fft_arguments(&args, dir, a->number_of_blocks, dx_block_size(), n);
    dx_set_buffers(a);
    if (a->number_of_blocks > 1) {
        a->context->CSSetShader(a->cs_global, nullptr, 0);
        while (--args.steps_left > args.steps_gpu) {
            dx_set_args(a, args.global_angle, 0, 0, 0, 0, 0, 0, args.steps++, 0xFFFFFFFF << args.steps_left, args.dist >>= 1);
            a->context->Dispatch(a->n_groups.x, a->n_groups.y, a->n_groups.z);
            swap_io(a);
            dx_set_buffers(a);
        }
        ++args.steps_left;
    }
    a->context->CSSetShader(a->cs_local, nullptr, 0);
    dx_set_args(a, args.global_angle, args.local_angle, args.scalar, args.steps_left, args.leading_bits, args.steps_gpu, 1, 0, 0, 0);
    a->context->Dispatch(a->n_groups.x, a->n_groups.y, a->n_groups.z);
}

__inline void dx_fft_2d(transform_direction dir, dx_args *args, const int n)
{
    UINT width = n > dx_tile_dim() ? (n / dx_tile_dim()) : 1;

    dx_fft(dir, args, n);

    swap_io(args);
    dx_set_buffers(args);
    args->context->CSSetShader(args->cs_transpose, nullptr, 0);
    args->context->Dispatch(width, width, 1);

    swap_io(args);
    dx_fft(dir, args, n);

    swap_io(args);
    dx_set_buffers(args);
    args->context->CSSetShader(args->cs_transpose, nullptr, 0);
    args->context->Dispatch(width, width, 1);
}