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

__inline void dx_create_buffer(ID3D11Device* device, ID3D11Buffer* buffer, bool is_staging_buffer, const int n)
{
    D3D11_BUFFER_DESC buffer_description;
    buffer_description.BindFlags = is_staging_buffer ? 0 : D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    buffer_description.Usage = is_staging_buffer ? D3D11_USAGE_STAGING : D3D11_USAGE_DEFAULT;
    buffer_description.CPUAccessFlags = is_staging_buffer ? D3D11_CPU_ACCESS_READ : 0;
    buffer_description.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    buffer_description.StructureByteStride = sizeof(cpx);
    buffer_description.ByteWidth = sizeof(cpx) * n;
    dx_check_error(device->CreateBuffer(&buffer_description, NULL, &buffer), "CreateBuffer");
}

__inline void dx_create_unordered_access_view(ID3D11Device* device, ID3D11Buffer* buffer, ID3D11UnorderedAccessView *buffer_uav, const int n)
{
    D3D11_UNORDERED_ACCESS_VIEW_DESC outputUAVDesc;
    outputUAVDesc.Buffer.FirstElement = 0;
    outputUAVDesc.Buffer.Flags = 0;
    outputUAVDesc.Buffer.NumElements = n;
    outputUAVDesc.Format = DXGI_FORMAT_UNKNOWN;
    outputUAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    dx_check_error(device->CreateUnorderedAccessView(buffer, &outputUAVDesc, &buffer_uav), "CreateUnorderedAccessView");
}

__inline void dx_create_parameter_buffer(ID3D11Device* device, ID3D11Buffer* arg_buffer)
{
    D3D11_BUFFER_DESC cbDesc;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbDesc.Usage = D3D11_USAGE_DYNAMIC;
    cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    cbDesc.MiscFlags = 0;
    cbDesc.ByteWidth = sizeof(dx_cs_args) + ((16 - (sizeof(dx_cs_args) & 0xFF)) % 16); // Must be multiple of 4 floats (128 bit)           35 1
    dx_check_error(device->CreateBuffer(&cbDesc, NULL, &arg_buffer), "CreateBuffer");
}

__inline void dx_write_buffer(ID3D11DeviceContext* context, ID3D11Buffer* buffer, cpx *in, const int n)
{
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    context->Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
    cpx* in_values = reinterpret_cast<cpx*>(mappedResource.pData);
    memcpy(in_values, &in[0], sizeof(cpx) * n);
    in_values = nullptr;
    context->Unmap(buffer, 0);
}

__inline void dx_read_buffer(ID3D11DeviceContext* context, ID3D11Buffer* buffer, cpx *out, const int n)
{
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    context->Map(buffer, 0, D3D11_MAP_READ, 0, &mappedResource);
    cpx* zData = reinterpret_cast<cpx*>(mappedResource.pData);
    memcpy(&out[0], zData, sizeof(cpx)*n);
    zData = nullptr;
    context->Unmap(buffer, 0);
}

void dx_setup(dx_args *args, cpx *in, const int n)
{
    float const a = 2.0f;

    D3D_FEATURE_LEVEL feat_lvl;
    dx_check_error(D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, NULL, NULL, 0, D3D11_SDK_VERSION, &args->device, &feat_lvl, &args->context), "D3D11CreateDevice");

    dx_create_buffer(args->device, args->input_buffer, false, n);
    dx_create_buffer(args->device, args->output_buffer, false, n);
    dx_create_buffer(args->device, args->staging_buffer, true, n);

    dx_check_error(args->device->CreateShaderResourceView(args->input_buffer, NULL, &args->input_srv), "CreateShaderResourceView");
    dx_check_error(args->device->CreateShaderResourceView(args->output_buffer, NULL, &args->output_srv), "CreateShaderResourceView");
    dx_create_unordered_access_view(args->device, args->input_buffer, args->input_buffer_uav, n);    
    dx_create_unordered_access_view(args->device, args->output_buffer, args->output_buffer_uav, n);

    dx_create_parameter_buffer(args->device, args->argument_buffer);
    dx_write_buffer(args->context, args->input_buffer, in, n);

    // Compile the compute shader into a blob.
    ID3DBlob* errorBlob = nullptr;
    dx_check_error(D3DCompileFromFile(L"Platforms/DirectX/dx_compute_shader.hlsl", NULL, NULL, "dx_fft", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, &args->shader_blob, &errorBlob), "D3DCompileFromFile", errorBlob);

    // Create a shader object from the compiled blob.
    dx_check_error(args->device->CreateComputeShader(args->shader_blob->GetBufferPointer(), args->shader_blob->GetBufferSize(), NULL, &args->compute_shader), "CreateComputeShader");

    // Make the shader active.
    args->context->CSSetShader(args->compute_shader, NULL, 0);

    // Attach the input buffers via their shader resource views.
    args->context->CSSetShaderResources(0, 1, &args->input_srv);
    args->context->CSSetShaderResources(0, 1, &args->output_srv);

    // Attach the constant buffer
    args->context->CSSetConstantBuffers(0, 1, &args->argument_buffer);
}

__inline void dx_map_parameters(ID3D11DeviceContext* context, ID3D11Buffer* arg_buffer, dx_cs_args *params)
{
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    context->Map(arg_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
    dx_cs_args* constants = reinterpret_cast<dx_cs_args *>(mappedResource.pData);
    constants[0] = *params;
    constants = nullptr;
    context->Unmap(arg_buffer, 0);
}

bool dx_fft(const int n)
{
    cpx *in = get_seq(n, 1);
    cpx *out = get_seq(n);
    cpx *ref = get_seq(n, in);
    dx_args args;

    UINT number_of_groups = (UINT)(n > MAX_BLOCK_SIZE ? n / MAX_BLOCK_SIZE : 1);
    dx_setup(&args, in, n);

    // Run (?)
    args.context->Dispatch(number_of_groups, 1, 1);
    args.context->CopyResource(args.staging_buffer, args.output_buffer);
    // Done (?)

    dx_read_buffer(args.context, args.staging_buffer, out, n);
    bool result = diff_seq(in, ref, n) > RELATIVE_ERROR_MARGIN;
    dx_shakedown(&args);
    return result;
}