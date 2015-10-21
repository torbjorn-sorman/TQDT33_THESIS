#include "dx_fft.h"

__inline void dx_fft(fftDir dir, cpx **in, cpx **out, const int n);
__inline void dx_fft_2d(fftDir dir, cpx **in, cpx **out, const int n);

//
// Testing
//

int dx_validate(const int n)
{
    cpx *in = get_seq(n, 1);
    cpx *out = get_seq(n);
    cpx *ref = get_seq(n, in);
    dx_fft(FFT_FORWARD, &in, &out, n);
    dx_fft(FFT_INVERSE, &out, &in, n);
    double diff = diff_seq(in, ref, n);
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
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        dx_fft(FFT_FORWARD, &in, &in, n);
        measures[i] = stopTimer();
    }
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

__inline void dx_fft(fftDir dir, cpx **in, cpx **out, const int n)
{
    return;
}

__inline void dx_fft_2d(fftDir dir, cpx **in, cpx **out, const int n)
{
    return;
}

__inline void dx_setup_arguments(dx_cs_args *args, fftDir dir, const int n)
{
    const int n2 = (n >> 1);
    args->block_range_half = n2;
    args->steps_left = log2_32(n);
    args->leading_bits = 32 - args->steps_left;
    args->steps_gpu = log2_32(MAX_BLOCK_SIZE);
    args->scalar = (dir == FFT_FORWARD ? 1.f : 1.f / n);
    args->number_of_blocks = n2 > MAX_BLOCK_SIZE ? (n2 / MAX_BLOCK_SIZE) : 1;
    args->angle = dir * (M_2_PI / n);
    args->local_angle = dir * (M_2_PI / (n / args->number_of_blocks));
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

    // Prepare algorithm constant parameters.
    dx_cs_args a;
    dx_setup_arguments(&a, FFT_FORWARD, n);
    dx_map_args(args.context, args.buffer_constant, &a);

    // Execute
    args.context->Dispatch(args.number_of_groups, 1, 1);
    /*
    // Read back result.
    args.context->CopyResource(args.buffer_staging, args.buffer_gpu_out);
    dx_read_buffer(args.context, args.buffer_staging, out, n);

    // Check result
    printf("DX\tExpected\t\tActual\n");
    for(int i = 0; i < n; ++i) {
        printf("\t%.3f\t%.3f\t\t%.3f\t%.3f\n", ref[i].x, ref[i].y, out[i].x, out[i].y);
    }
    */
    //
    // FFT Inverse
    //

    // Attach the out buffer to the output via its unordered access view. (SWAPPED)
    args.context->CSSetUnorderedAccessViews(0, 1, &args.buffer_gpu_out_uav, &args.init_counts);
    args.context->CSSetUnorderedAccessViews(1, 1, &args.buffer_gpu_in_uav, &args.init_counts);

    dx_setup_arguments(&a, FFT_INVERSE, n);
    dx_map_args(args.context, args.buffer_constant, &a);

    // Execute
    args.context->Dispatch(args.number_of_groups, 1, 1);

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