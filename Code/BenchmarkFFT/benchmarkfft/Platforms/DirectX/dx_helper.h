#ifndef DX_HELPER_H
#define DX_HELPER_H

#include <stdio.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <regex>
#include "../../Definitions.h"

struct dx_args
{
    int                         number_of_groups = 1;
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
    ID3D11ComputeShader*        compute_shader_fft_local = nullptr;
    ID3D11ComputeShader*        compute_shader_fft_local_inverse = nullptr;
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
    int     block_range_half;
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

static __inline D3D11_BUFFER_DESC get_input_buffer_description(const int dimension)
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

static __inline D3D11_BUFFER_DESC get_output_buffer_description(const int dimension)
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

static __inline D3D11_UNORDERED_ACCESS_VIEW_DESC get_unordered_access_view_description(const int dimension)
{
    D3D11_UNORDERED_ACCESS_VIEW_DESC desc;
    desc.Buffer.FirstElement = 0;
    desc.Buffer.Flags = 0;
    desc.Buffer.NumElements = dimension;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    return desc;
}

static __inline D3D11_BUFFER_DESC get_staging_buffer_description(const int dimension)
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

static __inline D3D11_BUFFER_DESC get_constant_buffer_description()
{
    D3D11_BUFFER_DESC desc;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    desc.Usage = D3D11_USAGE_DYNAMIC;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    desc.MiscFlags = 0;
    desc.ByteWidth = (UINT)padded_size(sizeof(cpx));
    return desc;
}

static __inline void dx_write_buffer(ID3D11DeviceContext* context, ID3D11Buffer* buffer, cpx* in, const int n)
{
    D3D11_MAPPED_SUBRESOURCE mapped_resource;
    context->Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped_resource);
    cpx* data = reinterpret_cast<cpx*>(mapped_resource.pData);
    memcpy(data, in, padded_size(sizeof(cpx) * n, 16));
    data = nullptr;
    context->Unmap(buffer, 0);
}

static __inline void dx_read_buffer(ID3D11DeviceContext* context, ID3D11Buffer* buffer, cpx *out, const int n)
{
    D3D11_MAPPED_SUBRESOURCE mapped_resource;
    context->Map(buffer, 0, D3D11_MAP_READ, 0, &mapped_resource);
    cpx* data = reinterpret_cast<cpx*>(mapped_resource.pData);
    memcpy(out, data, sizeof(cpx) * n);
    data = nullptr;
    context->Unmap(buffer, 0);
}

static __inline void dx_map_args(ID3D11DeviceContext* context, ID3D11Buffer* arg_buffer, dx_cs_args *params)
{
    D3D11_MAPPED_SUBRESOURCE mapped_resource;
    context->Map(arg_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped_resource);
    dx_cs_args* constants = reinterpret_cast<dx_cs_args *>(mapped_resource.pData);
    constants[0] = *params;
    constants = nullptr;
    context->Unmap(arg_buffer, 0);
}

static __inline void dx_setup(dx_args *args, LPCWSTR shader_file, const int n)
{
    args->number_of_groups = (n >> 1) > MAX_BLOCK_SIZE ? ((n >> 1) / MAX_BLOCK_SIZE) : 1;
    D3D_FEATURE_LEVEL featureLevel;
    D3D11_BUFFER_DESC input_buffer_desc = get_input_buffer_description(n);
    D3D11_BUFFER_DESC output_buffer_desc = get_output_buffer_description(n);
    D3D11_UNORDERED_ACCESS_VIEW_DESC output_uav_desc = get_unordered_access_view_description(n);
    D3D11_BUFFER_DESC staging_buffer_desc = get_staging_buffer_description(n);
    D3D11_BUFFER_DESC constant_buffer_desc = get_constant_buffer_description();
    ID3DBlob* errorBlob = nullptr;

    dx_check_error(D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, NULL, NULL, 0, D3D11_SDK_VERSION, &args->device, &featureLevel, &args->context), "D3D11CreateDevice");

    // Input only buffer.
    dx_check_error(args->device->CreateBuffer(&input_buffer_desc, NULL, &args->buffer_cpu_input), "Create CPU Buffer");
    dx_check_error(args->device->CreateShaderResourceView(args->buffer_cpu_input, NULL, &args->buffer_cpu_input_srv), "Create CPU Buffer ShaderResourceView");

    // GPU read/write accessible buffers.
    dx_check_error(args->device->CreateBuffer(&output_buffer_desc, NULL, &args->buffer_gpu_in), "Create GPU In Buffer ");
    dx_check_error(args->device->CreateBuffer(&output_buffer_desc, NULL, &args->buffer_gpu_out), "Create GPU Out Buffer ");

    // Create an unordered access view for the GPU buffers.  
    dx_check_error(args->device->CreateUnorderedAccessView(args->buffer_gpu_in, &output_uav_desc, &args->buffer_gpu_in_uav), "Create GPU In UnorderedAccessView");
    dx_check_error(args->device->CreateUnorderedAccessView(args->buffer_gpu_out, &output_uav_desc, &args->buffer_gpu_out_uav), "Create GPU Out UnorderedAccessView");

    // Create a staging buffer, which will be used to copy back from the GPU out buffer.
    dx_check_error(args->device->CreateBuffer(&staging_buffer_desc, NULL, &args->buffer_staging), "Create Staging Buffer");

    // Create a constant buffer (this buffer is used to pass the constant value 'a' to the kernel as cbuffer Constants).
    dx_check_error(args->device->CreateBuffer(&constant_buffer_desc, NULL, &args->buffer_constant), "Create Constant Buffer");

    // Compile the compute shader into a blob.
    dx_check_error(D3DCompileFromFile(shader_file, NULL, NULL, "dx_fft", "cs_5_0", D3D10_SHADER_ENABLE_STRICTNESS, 0, &args->shader_blob, &errorBlob), "D3DCompileFromFile", errorBlob);
    
    // Create a shader object from the compiled blob.
    dx_check_error(args->device->CreateComputeShader(args->shader_blob->GetBufferPointer(), args->shader_blob->GetBufferSize(), NULL, &args->compute_shader_fft_local), "CreateComputeShader");
    /*
    // Compile the compute shader into a blob.
    dx_check_error(D3DCompileFromFile(shader_file, NULL, NULL, "dx_fft", "cs_5_0", D3D10_SHADER_ENABLE_STRICTNESS, 0, &args->shader_blob, &errorBlob), "D3DCompileFromFile", errorBlob);

    // Create a shader object from the compiled blob.
    dx_check_error(args->device->CreateComputeShader(args->shader_blob->GetBufferPointer(), args->shader_blob->GetBufferSize(), NULL, &args->compute_shader_fft_local_inverse), "CreateComputeShader");
    */
    // Attach the out buffer to the output via its unordered access view.
    args->context->CSSetUnorderedAccessViews(0, 1, &args->buffer_gpu_in_uav, &args->init_counts);
    args->context->CSSetUnorderedAccessViews(1, 1, &args->buffer_gpu_out_uav, &args->init_counts);

    // Attach the input buffers via their shader resource views.
    args->context->CSSetShaderResources(0, 1, &args->buffer_cpu_input_srv);

    // Attach the constant buffer
    args->context->CSSetConstantBuffers(0, 1, &args->buffer_constant);
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

    args->compute_shader_fft_local->Release();
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

static __inline void dx_set_dim(LPCWSTR shader_file, const int n)
{
    std::ifstream in_file(shader_file);
    std::stringstream buffer;
    buffer << in_file.rdbuf();
    in_file.close();

    std::regex e("(#define\\s*GROUP_SIZE_X)\\s*\\d*$");
    std::ofstream out_file(shader_file);
    std::stringstream fmt;
    fmt << "$1 " << std::to_string(n >> 1);
    out_file << std::regex_replace(buffer.str(), e, fmt.str());
    out_file.close();
}

#endif