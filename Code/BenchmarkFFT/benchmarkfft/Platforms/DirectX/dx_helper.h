#ifndef DX_HELPER_H
#define DX_HELPER_H

#include <stdio.h>
#include "../../Definitions.h"
/*
struct dx_args
{
    const int                   group_size = MAX_BLOCK_SIZE;
    int                         number_of_groups = 1;
    int                         dimension = MAX_BLOCK_SIZE;
    UINT                        byte_width = 0;
    UINT                        init_counts = 0xFFFFFFFF;
    ID3D11DeviceContext*        context = nullptr;
    ID3D11Device*               device = nullptr;
    ID3D11Buffer*               argument_buffer = nullptr;
    ID3D11Buffer*               staging_buffer_read = nullptr;
    ID3D11Buffer*               out_buffer = nullptr;
    ID3D11Buffer*               in_buffer = nullptr;
    ID3D11Buffer*               input_buffer = nullptr;
    ID3D11ShaderResourceView*   in_srv = nullptr;
    ID3D11ShaderResourceView*   out_srv = nullptr;
    ID3D11UnorderedAccessView*  in_buffer_uav = nullptr;
    ID3D11UnorderedAccessView*  out_buffer_uav = nullptr;
    ID3DBlob*                   shader_blob = nullptr;
    ID3D11ComputeShader*        compute_shader = nullptr;    
};
*/
struct dx_args
{
    const int                   group_size = MAX_BLOCK_SIZE;
    int                         number_of_groups = 1;
    int                         dimension = MAX_BLOCK_SIZE;
    UINT                        init_counts = 0xFFFFFFFF;
    ID3D11DeviceContext*        context = nullptr;
    ID3D11Device*               device = nullptr;
    ID3D11Buffer*               buffer_cpu_input = nullptr;
    ID3D11ShaderResourceView*   buffer_cpu_input_srv = nullptr;
    ID3D11Buffer*               buffer_gpu_in = nullptr;
    ID3D11Buffer*               buffer_gpu_out = nullptr;
    ID3D11UnorderedAccessView*  buffer_gpu_in_uav = nullptr;
    ID3D11UnorderedAccessView*  buffer_gpu_out_uav = nullptr;
    ID3D11Buffer*               buffer_staging = nullptr;
    ID3D11Buffer*               buffer_constant = nullptr;
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

static __inline size_t padded_size(size_t sz, size_t width)
{
    return sz + ((width - (sz % width)) % width);
}

static __inline size_t padded_size(size_t sz)
{
    return padded_size(sz, 16); // 128 -bit registers
}

static __inline void dx_check_error(HRESULT hr, char *method, ID3DBlob* error_blob)
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

static __inline void dx_check_error(HRESULT hr, char *method)
{
    if (FAILED(hr)) {
        printf("%s failed with return code %x\n", method, hr);
        printf("Press the any key to continue...");
        getchar();
        exit(-1);
    }
}

__inline D3D11_BUFFER_DESC get_input_buffer_description(const int dimension)
{
    D3D11_BUFFER_DESC desc;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.Usage = D3D11_USAGE_DYNAMIC;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(cpx);
    desc.ByteWidth = sizeof(cpx) * dimension;
    return desc;
}

__inline D3D11_BUFFER_DESC get_output_buffer_description(const int dimension)
{
    D3D11_BUFFER_DESC desc;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(cpx);
    desc.ByteWidth = sizeof(cpx) * dimension;
    return desc;
}

__inline D3D11_UNORDERED_ACCESS_VIEW_DESC get_unordered_access_view_description(const int dimension)
{
    D3D11_UNORDERED_ACCESS_VIEW_DESC desc;
    desc.Buffer.FirstElement = 0;
    desc.Buffer.Flags = 0;
    desc.Buffer.NumElements = dimension;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    return desc;
}

__inline D3D11_BUFFER_DESC get_staging_buffer_description(const int dimension)
{
    D3D11_BUFFER_DESC desc;
    desc.BindFlags = 0;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(cpx);
    desc.ByteWidth = sizeof(cpx) * dimension;
    return desc;
}

__inline D3D11_BUFFER_DESC get_constant_buffer_description()
{
    D3D11_BUFFER_DESC desc;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    desc.Usage = D3D11_USAGE_DYNAMIC;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    desc.MiscFlags = 0;
    desc.ByteWidth = (UINT)padded_size(sizeof(cpx));
    return desc;
}

__inline void dx_write_buffer(ID3D11DeviceContext* context, ID3D11Buffer* buffer, cpx* in, const int n)
{
    D3D11_MAPPED_SUBRESOURCE mapped_resource;
    context->Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped_resource);
    cpx* data = reinterpret_cast<cpx*>(mapped_resource.pData);
    memcpy(data, in, padded_size(sizeof(cpx) * n, 16));
    data = nullptr;
    context->Unmap(buffer, 0);
}

__inline void dx_read_buffer(ID3D11DeviceContext* context, ID3D11Buffer* buffer, cpx *out, const int n)
{
    D3D11_MAPPED_SUBRESOURCE mapped_resource;
    context->Map(buffer, 0, D3D11_MAP_READ, 0, &mapped_resource);
    cpx* data = reinterpret_cast<cpx*>(mapped_resource.pData);
    memcpy(out, data, sizeof(cpx) * n);
    data = nullptr;
    context->Unmap(buffer, 0);
}

__inline void dx_map_parameters(ID3D11DeviceContext* context, ID3D11Buffer* arg_buffer, dx_cs_args *params)
{
    D3D11_MAPPED_SUBRESOURCE mapped_resource;
    context->Map(arg_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped_resource);
    dx_cs_args* constants = reinterpret_cast<dx_cs_args *>(mapped_resource.pData);
    constants[0] = *params;
    constants = nullptr;
    context->Unmap(arg_buffer, 0);
}

static __inline void dx_shakedown(dx_args *args)
{
    // Disconnect everything from the pipeline.
    ID3D11UnorderedAccessView* nullUAV = nullptr;
    args->context->CSSetUnorderedAccessViews(0, 1, &nullUAV, &args->init_counts);
    args->context->CSSetUnorderedAccessViews(1, 1, &nullUAV, &args->init_counts);
    ID3D11ShaderResourceView* nullSRV = nullptr;
    args->context->CSSetShaderResources(0, 1, &nullSRV);
    ID3D11Buffer* nullBuffer = nullptr;
    args->context->CSSetConstantBuffers(0, 1, &nullBuffer);

    args->compute_shader->Release();
    args->shader_blob->Release();
    args->buffer_constant->Release();
    args->buffer_staging->Release();
    args->buffer_gpu_out_uav->Release();
    args->buffer_gpu_out->Release();
    args->buffer_gpu_in_uav->Release();
    args->buffer_gpu_in->Release();
    args->buffer_cpu_input_srv->Release();
    args->buffer_cpu_input->Release();
    args->context->Release();
    args->device->Release();
}

#endif