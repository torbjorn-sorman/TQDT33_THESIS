#ifndef DX_HELPER_H
#define DX_HELPER_H

#include <stdio.h>
#include "../../Definitions.h"

struct dx_args
{
    const int                   group_size = MAX_BLOCK_SIZE;
    int                         number_of_groups = 1;
    int                         dimension = MAX_BLOCK_SIZE;
    UINT                        init_counts = 0xFFFFFFFF;
    ID3D11DeviceContext*        context = nullptr;
    ID3D11Device*               device = nullptr;
    ID3D11Buffer*               argument_buffer = nullptr;
    ID3D11Buffer*               staging_buffer = nullptr;
    ID3D11Buffer*               output_buffer = nullptr;
    ID3D11Buffer*               input_buffer = nullptr;
    ID3D11ShaderResourceView*   input_srv = nullptr;
    ID3D11ShaderResourceView*   output_srv = nullptr;
    ID3D11UnorderedAccessView*  input_buffer_uav = nullptr;
    ID3D11UnorderedAccessView*  output_buffer_uav = nullptr;
    ID3DBlob*                   shader_blob = nullptr;
    ID3D11ComputeShader*        compute_shader = nullptr;    
};

struct dx_cs_args
{
    float   angle;
    float   local_angle;
    float   scalar;
    int     steps_left;
    int     leading_bits;
    int     steps_gpu;
    int     number_of_blocks;
    int     n_half;
};

__inline void dx_check_error(HRESULT hr, char *method, ID3DBlob* error_blob)
{
    if (FAILED(hr)) {
        if (error_blob) {
            char const* message = (char*)error_blob->GetBufferPointer();
            printf("kernel.hlsl failed to compile; error message:\n");
            printf("%s\n", message);
            error_blob->Release();
        }
        printf("%s failed with return code %x\n", method, hr);        
        printf("Press the any key to continue...");
        getchar();
        exit(-1);
    }
}

__inline void dx_check_error(HRESULT hr, char *method)
{
    if (FAILED(hr)) {
        printf("%s failed with return code %x\n", method, hr);
        printf("Press the any key to continue...");
        getchar();
        exit(-1);
    }
}

__inline void dx_shakedown(dx_args *args)
{
    ID3D11UnorderedAccessView* nullUAV = nullptr;
    args->context->CSSetUnorderedAccessViews(0, 1, &nullUAV, &args->init_counts);
    args->context->CSSetUnorderedAccessViews(1, 1, &nullUAV, &args->init_counts);
    ID3D11ShaderResourceView* nullSRV = nullptr;
    args->context->CSSetShaderResources(0, 1, &nullSRV);
    args->context->CSSetShaderResources(1, 1, &nullSRV);
    ID3D11Buffer* nullBuffer = nullptr;
    args->context->CSSetConstantBuffers(0, 1, &nullBuffer);
    args->context->CSSetConstantBuffers(1, 1, &nullBuffer);

    if (args->compute_shader)
        args->compute_shader->Release();
    if (args->shader_blob)
        args->shader_blob->Release();
    if (args->argument_buffer)
        args->argument_buffer->Release();
    if (args->input_buffer)
        args->input_buffer->Release();
    if (args->output_buffer)
        args->output_buffer->Release();
    if (args->input_buffer_uav)
        args->input_buffer_uav->Release();
    if (args->output_buffer_uav)
        args->output_buffer_uav->Release();
    if (args->input_srv)
        args->input_srv->Release();
    if (args->output_srv)
        args->output_srv->Release();
    args->context->Release();
    args->device->Release();
}

#endif