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

void dx_setup(dx_args *args)
{
    D3D_FEATURE_LEVEL featureLevel;
    dx_check_error(D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, NULL, NULL, 0, D3D11_SDK_VERSION, &args->device, &featureLevel, &args->context), "D3D11CreateDevice");

    // Input only buffer.
    D3D11_BUFFER_DESC input_buffer_desc = get_input_buffer_description(args->dimension);
    dx_check_error(args->device->CreateBuffer(&input_buffer_desc, NULL, &args->buffer_cpu_input), "Create CPU Buffer");
    dx_check_error(args->device->CreateShaderResourceView(args->buffer_cpu_input, NULL, &args->buffer_cpu_input_srv), "Create CPU Buffer ShaderResourceView");

    // GPU read/write accessible buffers.
    D3D11_BUFFER_DESC output_buffer_desc = get_output_buffer_description(args->dimension);
    dx_check_error(args->device->CreateBuffer(&output_buffer_desc, NULL, &args->buffer_gpu_in), "Create GPU In Buffer ");
    dx_check_error(args->device->CreateBuffer(&output_buffer_desc, NULL, &args->buffer_gpu_out), "Create GPU Out Buffer ");

    // Create an unordered access view for the GPU buffers.  
    D3D11_UNORDERED_ACCESS_VIEW_DESC output_uav_desc = get_unordered_access_view_description(args->dimension);
    dx_check_error(args->device->CreateUnorderedAccessView(args->buffer_gpu_in, &output_uav_desc, &args->buffer_gpu_in_uav), "Create GPU In UnorderedAccessView");
    dx_check_error(args->device->CreateUnorderedAccessView(args->buffer_gpu_out, &output_uav_desc, &args->buffer_gpu_out_uav), "Create GPU Out UnorderedAccessView");

    // Create a staging buffer, which will be used to copy back from the GPU out buffer.
    D3D11_BUFFER_DESC staging_buffer_desc = get_staging_buffer_description(args->dimension);
    dx_check_error(args->device->CreateBuffer(&staging_buffer_desc, NULL, &args->buffer_staging), "Create Staging Buffer");

    // Create a constant buffer (this buffer is used to pass the constant value 'a' to the kernel as cbuffer Constants).
    D3D11_BUFFER_DESC constant_buffer_desc = get_constant_buffer_description();
    dx_check_error(args->device->CreateBuffer(&constant_buffer_desc, NULL, &args->buffer_constant), "Create Constant Buffer");

    // Compile the compute shader into a blob.
    ID3DBlob* errorBlob = nullptr;
    dx_check_error(D3DCompileFromFile(L"Platforms/DirectX/dx_compute_shader.hlsl", NULL, NULL, "dx_fft", "cs_5_0", D3D10_SHADER_ENABLE_STRICTNESS, 0, &args->shader_blob, &errorBlob), "D3DCompileFromFile", errorBlob);

    // Create a shader object from the compiled blob.
    dx_check_error(args->device->CreateComputeShader(args->shader_blob->GetBufferPointer(), args->shader_blob->GetBufferSize(), NULL, &args->compute_shader), "CreateComputeShader");

    // Make the shader active.
    args->context->CSSetShader(args->compute_shader, NULL, 0);

    // Attach the out buffer to the output via its unordered access view.
    args->context->CSSetUnorderedAccessViews(0, 1, &args->buffer_gpu_in_uav, &args->init_counts);
    args->context->CSSetUnorderedAccessViews(1, 1, &args->buffer_gpu_out_uav, &args->init_counts);

    // Attach the input buffers via their shader resource views.
    args->context->CSSetShaderResources(0, 1, &args->buffer_cpu_input_srv);

    // Attach the constant buffer
    args->context->CSSetConstantBuffers(0, 1, &args->buffer_constant);
}


bool dx_fft(const int n)
{
    dx_args args;
    dx_cs_args params;
    params.n_half = n / 2;
    cpx *in = get_seq(n, 1);
    cpx *out = get_seq(n);
    cpx *ref = get_seq(n, in);

    args.number_of_groups = n > MAX_BLOCK_SIZE ? n / MAX_BLOCK_SIZE : 1;
    args.dimension = args.number_of_groups * args.group_size;

    

    //
    // Setup done(?)
    //

    // Write data to the input buffer.
    dx_write_buffer(args.context, args.buffer_cpu_input, in, n);

    // Map the constant arguments.
    dx_map_parameters(args.context, args.buffer_constant, &params);

    // Execute the shader, in 'numGroups' groups of 'groupSize' threads each.
    args.context->Dispatch(args.number_of_groups, 1, 1);

    // Copy the out buffer to the staging buffer so that we can 
    // retrieve the data for accesss by the CPU.
    args.context->CopyResource(args.buffer_staging, args.buffer_gpu_in);

    dx_read_buffer(args.context, args.buffer_staging, out, n);

    // Now compare the GPU results against expected values.
    bool resultOK = true;
    for (size_t i = 0; i < n; ++i)
    {
        // NOTE: This comparison assumes the GPU produces *exactly* the 
        // same result as the CPU.  In general, this will not be the case
        // with floating-point calculations.
        cpx expected;
        expected.x = params.n_half * in[i].x;
        expected.y = params.n_half * in[i].y;
        if (out[i].x != expected.x || out[i].y != expected.y)
        {
            printf("Unexpected result at position %lu\n", i);
            resultOK = false;
        }
    }

    if (!resultOK)
    {
        printf("GPU results differed from the CPU results.\n");
        OutputDebugStringA("GPU results differed from the CPU results.\n");
        return 1;
    }

    printf("GPU output matched the CPU results.\n");
    OutputDebugStringA("GPU output matched the CPU results.\n");

    dx_shakedown(&args);

    return 0;
}