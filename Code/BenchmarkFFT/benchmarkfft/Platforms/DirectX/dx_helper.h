#ifndef DX_HELPER_H
#define DX_HELPER_H

#include <stdio.h>
#include "../../Definitions.h"

struct dx_args
{
    const int group_size = MAX_BLOCK_SIZE;
    int number_of_groups = 1;
    int dimension = MAX_BLOCK_SIZE;
    ID3D11DeviceContext* context = nullptr;
    ID3D11Device* device = nullptr;
    ID3D11Buffer* argument_buffer = nullptr;
    ID3D11Buffer* staging_buffer = nullptr;
    ID3D11Buffer* output_buffer = nullptr;
    ID3D11Buffer* input_buffer = nullptr;
    ID3D11ShaderResourceView *input_srv = nullptr;
    ID3D11ShaderResourceView *output_srv = nullptr;
    ID3D11UnorderedAccessView* input_buffer_uav = nullptr;
    ID3D11UnorderedAccessView* output_buffer_uav = nullptr;
    ID3DBlob* shader_blob = nullptr;
    ID3D11ComputeShader* compute_shader = nullptr;
    UINT init_counts = 0xFFFFFFFF;
};

struct dx_cs_args
{
    float angle;
    float local_angle;
    int steps_left;
    int leading_bits;
    int steps_gpu;
    float scalar;
    int number_of_blocks;
    int n_half;
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
    // Disconnect everything from the pipeline.
    ID3D11UnorderedAccessView* nullUAV = nullptr;
    args->context->CSSetUnorderedAccessViews(0, 1, &nullUAV, &args->init_counts);
    ID3D11ShaderResourceView* nullSRV = nullptr;
    args->context->CSSetShaderResources(0, 1, &nullSRV);
    args->context->CSSetShaderResources(1, 1, &nullSRV);
    ID3D11Buffer* nullBuffer = nullptr;
    args->context->CSSetConstantBuffers(0, 1, &nullBuffer);

    args->compute_shader->Release();
    args->shader_blob->Release();
    args->argument_buffer->Release();
    args->staging_buffer->Release();
    args->input_buffer->Release();
    args->output_buffer->Release();
    args->input_buffer_uav->Release();
    args->output_buffer_uav->Release();
    args->input_srv->Release();
    args->output_srv->Release();
    args->context->Release();
    args->device->Release();
}

#endif