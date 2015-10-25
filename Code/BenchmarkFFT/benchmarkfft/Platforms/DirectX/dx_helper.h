#ifndef DX_HELPER_H
#define DX_HELPER_H

#include <stdio.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <regex>
#include "../../Definitions.h"
#include <d3d11.h>
#include <d3dcompiler.h>
#include "../../Common/mymath.h"

#include <comdef.h>
#include <comip.h>

_COM_SMARTPTR_TYPEDEF(ID3D11Query, __uuidof(ID3D11Query));

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
    ID3D11Buffer*               buffer_sync_in = nullptr;
    ID3D11Buffer*               buffer_sync_out = nullptr;
    ID3D11UnorderedAccessView*  buffer_gpu_in_uav = nullptr;
    ID3D11UnorderedAccessView*  buffer_gpu_out_uav = nullptr;
    ID3D11Buffer*               buffer_staging = nullptr;
    ID3D11Buffer*               buffer_constant = nullptr;
    ID3DBlob*                   cs_blob_local = nullptr;
    ID3DBlob*                   cs_blob_global = nullptr;
    ID3D11ComputeShader*        cs_local = nullptr;
    ID3D11ComputeShader*        cs_global = nullptr;
};

struct dx_cs_args
{
    float           angle;
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
    int             dist;
};

struct profiler_data
{
    ID3D11QueryPtr disjoint_query;
    ID3D11QueryPtr q_start;
    ID3D11QueryPtr q_end;
};

static __inline void dx_start_profiling(dx_args *args, profiler_data *p_data)
{
    if (p_data->disjoint_query == NULL)
    {
        D3D11_QUERY_DESC desc;
        desc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
        desc.MiscFlags = 0;
        args->device->CreateQuery(&desc, &p_data->disjoint_query);
        desc.Query = D3D11_QUERY_TIMESTAMP;
        args->device->CreateQuery(&desc, &p_data->q_start);
        args->device->CreateQuery(&desc, &p_data->q_end);
    }
    args->context->Begin(p_data->disjoint_query);
    args->context->End(p_data->q_start);
}

static __inline void dx_end_profiling(dx_args *args, profiler_data *p_data)
{
    args->context->End(p_data->q_end);
    args->context->End(p_data->disjoint_query);
}

static __inline double dx_avg(profiler_data profiler[], dx_args *args)
{
    double m[NUM_PERFORMANCE];
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        profiler_data p = profiler[i];
        UINT64 ts_start, ts_end;
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT q_freq;
        while (S_OK != args->context->GetData(p.q_start, &ts_start, sizeof(UINT64), 0)){};
        while (S_OK != args->context->GetData(p.q_end, &ts_end, sizeof(UINT64), 0)){};
        while (S_OK != args->context->GetData(p.disjoint_query, &q_freq, sizeof(D3D11_QUERY_DATA_TIMESTAMP_DISJOINT), 0)){};
        m[i] = (((double)(ts_end - ts_start)) / ((double)q_freq.Frequency)) * 1000000.0;
    }
    return average_best(m, NUM_PERFORMANCE);
}

static __inline double dx_time_elapsed(profiler_data *p, dx_args *args)
{
    UINT64 ts_start, ts_end;
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT q_freq;
    while (S_OK != args->context->GetData(p->q_start, &ts_start, sizeof(UINT64), 0)){};
    while (S_OK != args->context->GetData(p->q_end, &ts_end, sizeof(UINT64), 0)){};
    while (S_OK != args->context->GetData(p->disjoint_query, &q_freq, sizeof(D3D11_QUERY_DATA_TIMESTAMP_DISJOINT), 0)){};
    return (((double)(ts_end - ts_start)) / ((double)q_freq.Frequency)) * 1000000.0;
}

template<typename T> static __inline void swap(T **a, T **b)
{
    T *c = *a;
    *a = *b;
    *b = c;
}

static __inline void swap_device_buffers(dx_args *a)
{
    swap<ID3D11Buffer>(&a->buffer_gpu_in, &a->buffer_gpu_out);
}

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

static __inline void dx_read_buffer(dx_args* args, ID3D11Buffer* src, cpx* dst, const int n)
{
    args->context->CopyResource(args->buffer_staging, src);
    D3D11_MAPPED_SUBRESOURCE mapped_resource;
    args->context->Map(args->buffer_staging, 0, D3D11_MAP_READ, 0, &mapped_resource);
    cpx* data = reinterpret_cast<cpx*>(mapped_resource.pData);
    memcpy(dst, data, sizeof(cpx) * n);
    data = nullptr;
    args->context->Unmap(args->buffer_staging, 0);
}

template<typename T> static __inline void dx_map_args(ID3D11DeviceContext* context, ID3D11Buffer* arg_buffer, T *params)
{
    D3D11_MAPPED_SUBRESOURCE mapped_resource;
    context->Map(arg_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped_resource);
    T* constants = reinterpret_cast<T *>(mapped_resource.pData);
    constants[0] = *params;
    constants = nullptr;
    context->Unmap(arg_buffer, 0);
}

static __inline void dx_set_dim(LPCWSTR shader_file, const int n)
{ 
    std::ifstream in_file(shader_file);
    std::stringstream buffer;
    buffer << in_file.rdbuf();
    std::string new_content = buffer.str();
    in_file.close();

    std::regex e_grp_sz("(#define\\s*GROUP_SIZE_X)\\s*\\d*$");
    std::regex e_num_blk("(#define\\s*NUMBER_OF_BLOCKS)\\s*\\d*$");

    std::ofstream out_file(shader_file);
    std::stringstream fmt1, fmt2;
    const int n2 = n >> 1;
    if (n2 > MAX_BLOCK_SIZE) {
        fmt1 << "$1 " << std::to_string(MAX_BLOCK_SIZE);
        if ((n2 / MAX_BLOCK_SIZE) > HW_LIMIT)
            fmt2 << "$1 " << std::to_string(1);
        else
            fmt2 << "$1 " << std::to_string((n2 / MAX_BLOCK_SIZE));
    }
    else {
        fmt1 << "$1 " << std::to_string(n2);
        fmt2 << "$1 " << std::to_string(1);
    }
    new_content = std::regex_replace(new_content, e_grp_sz, fmt1.str());
    new_content = std::regex_replace(new_content, e_num_blk, fmt2.str());

    out_file << new_content;
    out_file.close();
}

static __inline void dx_setup(dx_args* args, cpx* in, const int n)
{
    LPCWSTR cs_file = L"Platforms/DirectX/dx_cs.hlsl";
    dx_set_dim(cs_file, n);

    args->number_of_groups = (n >> 1) > MAX_BLOCK_SIZE ? ((n >> 1) / MAX_BLOCK_SIZE) : 1;
    D3D_FEATURE_LEVEL featureLevel;
    D3D11_BUFFER_DESC input_buffer_desc = get_input_buffer_description(n);
    D3D11_BUFFER_DESC output_buffer_desc = get_output_buffer_description(n);
    D3D11_BUFFER_DESC staging_buffer_desc = get_staging_buffer_description(n);
    D3D11_BUFFER_DESC constant_buffer_desc = get_constant_buffer_description();
    D3D11_UNORDERED_ACCESS_VIEW_DESC output_uav_desc = get_unordered_access_view_description(n);
    ID3DBlob* errorBlob = nullptr;
    /*
    ID3D11Buffer *sync_in = nullptr;
    ID3D11Buffer *sync_out = nullptr;
    ID3D11UnorderedAccessView *uav_sync_in = nullptr;
    ID3D11UnorderedAccessView *uav_sync_out = nullptr;
    */
    dx_check_error(D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, NULL, NULL, 0, D3D11_SDK_VERSION, &args->device, &featureLevel, &args->context), "D3D11CreateDevice");

    // GPU read/write accessible buffers.
    dx_check_error(args->device->CreateBuffer(&output_buffer_desc, NULL, &args->buffer_gpu_in), "Create GPU In Buffer ");
    dx_check_error(args->device->CreateBuffer(&output_buffer_desc, NULL, &args->buffer_gpu_out), "Create GPU Out Buffer ");
    /*
    dx_check_error(args->device->CreateBuffer(&output_buffer_desc, NULL, &sync_in), "Create GPU Out Buffer ");
    dx_check_error(args->device->CreateBuffer(&output_buffer_desc, NULL, &sync_out), "Create GPU Out Buffer ");
    */

    // Input only buffer.
    dx_check_error(args->device->CreateBuffer(&input_buffer_desc, NULL, &args->buffer_cpu_input), "Create CPU Buffer");
    dx_check_error(args->device->CreateShaderResourceView(args->buffer_cpu_input, NULL, &args->buffer_cpu_input_srv), "Create CPU Buffer ShaderResourceView");

    // Create an unordered access view for the GPU buffers.  
    dx_check_error(args->device->CreateUnorderedAccessView(args->buffer_gpu_in, &output_uav_desc, &args->buffer_gpu_in_uav), "Create GPU In UnorderedAccessView");
    dx_check_error(args->device->CreateUnorderedAccessView(args->buffer_gpu_out, &output_uav_desc, &args->buffer_gpu_out_uav), "Create GPU Out UnorderedAccessView");
    /*
    dx_check_error(args->device->CreateUnorderedAccessView(sync_in, &output_uav_desc, &uav_sync_in), "Create GPU Out UnorderedAccessView");
    dx_check_error(args->device->CreateUnorderedAccessView(sync_out, &output_uav_desc, &uav_sync_out), "Create GPU Out UnorderedAccessView");
    */    

    // Create a staging buffer, which will be used to copy back from the GPU out buffer.
    dx_check_error(args->device->CreateBuffer(&staging_buffer_desc, NULL, &args->buffer_staging), "Create Staging Buffer");

    // Create a constant buffer (this buffer is used to pass the constant value 'a' to the kernel as cbuffer Constants).
    dx_check_error(args->device->CreateBuffer(&constant_buffer_desc, NULL, &args->buffer_constant), "Create Constant Buffer");

    // Compile the compute shader into a blob.
    dx_check_error(D3DCompileFromFile(cs_file, NULL, NULL, "dx_local", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &args->cs_blob_local, &errorBlob), "D3DCompileFromFile", errorBlob);
    //dx_check_error(D3DCompileFromFile(cs_file, NULL, NULL, "dx_global", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &args->cs_blob_global, &errorBlob), "D3DCompileFromFile", errorBlob);

    // Create a shader object from the compiled blob.
    dx_check_error(args->device->CreateComputeShader(args->cs_blob_local->GetBufferPointer(), args->cs_blob_local->GetBufferSize(), NULL, &args->cs_local), "CreateComputeShader");
    //dx_check_error(args->device->CreateComputeShader(args->cs_blob_global->GetBufferPointer(), args->cs_blob_global->GetBufferSize(), NULL, &args->cs_global), "CreateComputeShader");

    // Attach the input buffers via their shader resource views.
    args->context->CSSetShaderResources(0, 1, &args->buffer_cpu_input_srv);

    // Attach the constant buffer
    args->context->CSSetConstantBuffers(0, 1, &args->buffer_constant);

    if (in != NULL)
        dx_write_buffer(args->context, args->buffer_cpu_input, in, n);
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

    args->cs_local->Release();
    if (args->cs_global)
        args->cs_global->Release();
    args->cs_blob_local->Release();
    if (args->cs_blob_global)
        args->cs_blob_global->Release();
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