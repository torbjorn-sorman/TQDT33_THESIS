#include "dx_fft.h"

__inline void dx_fft(fftDir dir, cpx **in, cpx **out, const int n);
__inline void dx_fft_2d(fftDir dir, cpx **in, cpx **out, const int n);
static __inline void dx_execute(dx_args *args, fftDir dir, bool swap_uav, const int n);

//
// Testing
//

int dx_validate(const int n)
{
    cpx *in = get_seq(n, 1);
    cpx *out = get_seq(n);
    cpx *ref = get_seq(n, in);

    // Setup Buffers and Shaders.
    dx_args args;
    LPCWSTR shader_file = L"Platforms/DirectX/dx_compute_shader.hlsl";
    dx_set_dim(shader_file, n);
    dx_setup(&args, shader_file, n);
    dx_write_buffer(args.context, args.buffer_cpu_input, in, n);
    args.context->CSSetShader(args.compute_shader_fft_local, NULL, 0);

    dx_execute(&args, FFT_FORWARD, false, n);
    dx_execute(&args, FFT_INVERSE, true, n);

    // Read back result.
    args.context->CopyResource(args.buffer_staging, args.buffer_gpu_out);
    dx_read_buffer(args.context, args.buffer_staging, out, n);

    double diff = diff_seq(out, ref, n);
    dx_shakedown(&args);
    free(in);
    free(out);
    free(ref);
    return diff < RELATIVE_ERROR_MARGIN;
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

double dx_performance(const int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in = get_seq(n, 1);

    // Setup Buffers and Shaders.
    dx_args args;
    LPCWSTR shader_file = L"Platforms/DirectX/dx_compute_shader.hlsl";
    dx_set_dim(shader_file, n);
    dx_setup(&args, shader_file, n);
    dx_write_buffer(args.context, args.buffer_cpu_input, in, n);
    args.context->CSSetShader(args.compute_shader_fft_local, NULL, 0);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        dx_execute(&args, FFT_FORWARD, false, n);
        measures[i] = stopTimer();
    }
    dx_shakedown(&args);
    free(in);
    return average_best(measures, NUM_PERFORMANCE);
}

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

__inline void dx_setup_arguments(dx_args *a, dx_cs_args *p, fftDir dir, const int n)
{
    const int n2 = (n >> 1);
    p->block_range_half = n2;
    p->steps_left = log2_32(n);
    p->leading_bits = 32 - p->steps_left;
    p->steps_gpu = log2_32(MAX_BLOCK_SIZE);
    p->scalar = (dir == FFT_FORWARD ? 1.f : 1.f / n);
    p->number_of_blocks = n2 > MAX_BLOCK_SIZE ? (n2 / MAX_BLOCK_SIZE) : 1;
    p->angle = dir * (M_2_PI / n);
    p->local_angle = dir * (M_2_PI / (n / p->number_of_blocks));
    // Associate UAVs with correct RWStructuredBuffer
    a->context->CSSetUnorderedAccessViews(0, 1, &a->buffer_gpu_in_uav, &a->init_counts);
    a->context->CSSetUnorderedAccessViews(1, 1, &a->buffer_gpu_out_uav, &a->init_counts);
}

static __inline void dx_execute(dx_args *args, fftDir dir, bool swap_uav, const int n)
{
    // Prepare algorithm parameters.
    if (swap_uav)
        swap_device_buffers(args);
    dx_cs_args params;
    dx_setup_arguments(args, &params, dir, n);
    dx_map_args(args->context, args->buffer_constant, &params);
    // Execute
    args->context->Dispatch(args->number_of_groups, 1, 1);
}

__inline void dx_fft(fftDir dir, cpx **in, cpx **out, const int n)
{
    return;
}

__inline void dx_fft_2d(fftDir dir, cpx **in, cpx **out, const int n)
{
    return;
}



bool dx_fft(const int n)
{
    // Prepare test sequences.
    cpx *in = get_seq(n, 1);
    cpx *out = get_seq(n);
    cpx *ref = get_seq(n, in);

    // Setup Buffers and Shaders.
    dx_args args;
    LPCWSTR shader_file = L"Platforms/DirectX/dx_compute_shader.hlsl";
    dx_set_dim(shader_file, n);
    dx_setup(&args, shader_file, n);
    dx_write_buffer(args.context, args.buffer_cpu_input, in, n);

    //
    // FFT Forward
    //
    args.context->CSSetShader(args.compute_shader_fft_local, NULL, 0);
    dx_execute(&args, FFT_FORWARD, false, n);

    //
    // FFT Inverse
    //        
    dx_execute(&args, FFT_INVERSE, true, n);

    // Read back result.
    args.context->CopyResource(args.buffer_staging, args.buffer_gpu_out);
    dx_read_buffer(args.context, args.buffer_staging, out, n);

    // Validate result
    printf("DX\tExpected\t\tActual\n");
    for (int i = 0; i < n; ++i) {
        printf("\t%.3f\t%.3f\t\t%.3f\t%.3f\n", ref[i].x, ref[i].y, out[i].x, out[i].y);
    }

    dx_shakedown(&args);
    free(in);
    free(out);
    free(ref);
    return 0;
}