#include "dx_fft.h"

__inline void dx_fft(transform_direction dir, dx_args *args, const int n);
__inline void dx_fft_2d(transform_direction dir, cpx **in, cpx **out, const int n);

//
// Testing
//

void dx_prepare(dx_args *args, cpx *in, const int n)
{
    LPCWSTR cs_file_l = L"Platforms/DirectX/dx_cs_local.hlsl";
    LPCWSTR cs_file_g = L"Platforms/DirectX/dx_cs_global.hlsl";
    dx_set_dim(cs_file_l, n);
    dx_set_dim(cs_file_g, n);
    dx_setup(args, cs_file_l, cs_file_g, n);
    if (in != NULL)
        dx_write_buffer(args->context, args->buffer_cpu_input, in, n);
}

int dx_validate(const int n)
{
    cpx *in = get_seq(n, 1);
    cpx *out = get_seq(n);
    cpx *ref = get_seq(n, in);
    dx_args args;
    dx_prepare(&args, in, n);

    dx_fft(FFT_FORWARD, &args, n);
    dx_read_buffer(&args, args.buffer_gpu_out, out, n);
<<<<<<< HEAD

#ifdef SHOW_BLOCKING_DEBUG    
    dx_read_buffer(&args, args.buffer_gpu_in, in, n);
    cpx_to_console(out, "DX Out", 8);
    getchar();
#endif

=======
>>>>>>> parent of b3d7254... all is crap
    double forward_diff = diff_forward_sinus(out, n);
    swap_device_buffers(&args);
    dx_fft(FFT_INVERSE, &args, n);
    dx_read_buffer(&args, args.buffer_gpu_out, out, n);
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

#define DX_TIMESTAMP

#ifndef DX_TIMESTAMP
double dx_performance(const int n)
{
    double measures[NUM_PERFORMANCE];
    dx_args args;
    profiler_data profiler[NUM_PERFORMANCE];
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        dx_prepare(&args, NULL, n);
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

<<<<<<< HEAD
/*float           angle;
    float           local_angle;
    float           scalar;
    int             steps_left;
    int             leading_bits;
    int             steps_gpu;
    int             number_of_blocks;
    int             block_range_half;
    bool            load_input;
    int             steps;
    unsigned int    lmask;
    int             dist;*/

__inline void dx_set_local_args(dx_args *a, float global_angle, float local_angle, int steps_left, int leading_bits, int steps_gpu, float scalar, int number_of_blocks, int block_range_half, bool load_input)
{    
    /*
    dx_cs_args p;
=======
__inline void dx_set_local_args(dx_args *a, float global_angle, float local_angle, int steps_left, int leading_bits, int steps_gpu, float scalar, int number_of_blocks, int block_range_half, int load_input)
{
    dx_cs_args_local p;
    a->context->CSSetShader(a->cs_local, NULL, 0);
>>>>>>> parent of b3d7254... all is crap
    p.angle = global_angle;
    p.local_angle = local_angle;
    p.block_range_half = block_range_half;
    p.steps_left = steps_left;
    p.leading_bits = leading_bits;
    p.steps_gpu = steps_gpu;
    p.scalar = scalar;
    p.number_of_blocks = number_of_blocks;
    p.load_input = load_input;
    dx_map_args<dx_cs_args>(a->context, a->buffer_constant, &p);
    */
    
    dx_cs_args cb = { global_angle, local_angle, scalar, steps_left, leading_bits, steps_gpu, number_of_blocks, block_range_half, load_input, 0, 0, 0 };
    a->context->UpdateSubresource(a->buffer_constant, 0, nullptr, &cb, 0, 0);
    a->context->CSSetConstantBuffers(0, 1, &a->buffer_constant);
    
    a->context->CSSetUnorderedAccessViews(0, 1, &a->buffer_gpu_in_uav, &a->init_counts);
    a->context->CSSetUnorderedAccessViews(1, 1, &a->buffer_gpu_out_uav, &a->init_counts);
<<<<<<< HEAD
    a->context->CSSetShader(a->cs_local, nullptr, 0);

}

__inline void dx_set_global_args(dx_args *a, float angle, int dist, unsigned int lmask, int steps, bool load_input)
{   
    dx_cs_args p;
=======
    dx_map_args<dx_cs_args_local>(a->context, a->buffer_constant, &p);
}

__inline void dx_set_global_args(dx_args *a, float angle, int dist, unsigned int lmask, int steps, int load_input, bool use_in_buffer)
{
    dx_cs_args_global p;
    a->context->CSSetShader(a->cs_global, NULL, 0);
>>>>>>> parent of b3d7254... all is crap
    p.angle = angle;
    p.dist = dist;
    p.lmask = lmask;
    p.steps = steps;
    p.load_input = load_input;
<<<<<<< HEAD
    dx_map_args<dx_cs_args>(a->context, a->buffer_constant, &p);
    /*
    dx_cs_args cb = { angle, 0.f, 0.f, 0, 0, 0, 0, 0, load_input, steps, lmask, dist };
    a->context->UpdateSubresource(a->buffer_constant, 0, nullptr, &cb, 0, 0);
    a->context->CSSetConstantBuffers(0, 1, &a->buffer_constant);
    */
    a->context->CSSetUnorderedAccessViews(0, 1, &a->buffer_gpu_in_uav, &a->init_counts);
    a->context->CSSetShader(a->cs_global, nullptr, 0);
=======
    a->context->CSSetUnorderedAccessViews(0, 1, use_in_buffer ? &a->buffer_gpu_in_uav : &a->buffer_gpu_out_uav, &a->init_counts);
    a->context->CSSetUnorderedAccessViews(1, 1, &a->buffer_gpu_out_uav, &a->init_counts);
    dx_map_args<dx_cs_args_global>(a->context, a->buffer_constant, &p);
>>>>>>> parent of b3d7254... all is crap
}

__inline void _dx_fft(transform_direction dir, dx_args *args, const int n)
{

    int n_half = (n >> 1);
    int steps_left = log2_32(n);
    int leading_bits = 32 - steps_left;
    int steps_gpu = log2_32(MAX_BLOCK_SIZE);
    float scalar = (dir == FFT_FORWARD ? 1.f : 1.f / n);
    int load_input = 1;
    int number_of_blocks = n_half > MAX_BLOCK_SIZE ? (n_half / MAX_BLOCK_SIZE) : 1;;
    int n_per_block = n / number_of_blocks;
    float global_angle = dir * (M_2_PI / n);
    float local_angle = dir * (M_2_PI / n_per_block);
    int block_range_half = n_half;
    if (number_of_blocks >= HW_LIMIT) {


        --steps_left;
        int steps = 0;
        int dist = n_half;
        dx_set_global_args(args, global_angle, dist, 0xFFFFFFFF << steps_left, steps, load_input);
        args->context->Dispatch(args->number_of_groups, 1, 1);
        load_input = 0;

        while (--steps_left > steps_gpu) {
            dist >>= 1;
            ++steps;
            dx_set_global_args(args, global_angle, dist, 0xFFFFFFFF << steps_left, steps, load_input);
            args->context->Dispatch(args->number_of_groups, 1, 1);
        }
<<<<<<< HEAD
=======
        swap_device_buffers(args);
>>>>>>> parent of b3d7254... all is crap
        ++steps_left;
        number_of_blocks = 1;
        block_range_half = n_per_block >> 1;
    }

    dx_set_local_args(args, global_angle, local_angle, steps_left, leading_bits, steps_gpu, scalar, number_of_blocks, block_range_half, load_input);
    args->context->Dispatch(args->number_of_groups, 1, 1);
}

__inline void dx_fft_2d(transform_direction dir, cpx **in, cpx **out, const int n)
{
    return;
}