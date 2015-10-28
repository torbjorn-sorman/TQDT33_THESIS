#ifndef DX_HELPER_H
#define DX_HELPER_H

#include <stdio.h>
#include <string>
#include <iostream>
#include "../../Definitions.h"
#include <d3d11.h>
#include <d3dcompiler.h>
#include "../../Common/mymath.h"
#include "dx_csmanip.h"


_COM_SMARTPTR_TYPEDEF(ID3D11Query, __uuidof(ID3D11Query));

struct dx_args
{
    dim3                        n_groups = { 1, 1, 1 };
    UINT                        init_cnts = 0xFFFFFFFF;
    ID3D11DeviceContext*        context = nullptr;
    ID3D11Device*               device = nullptr;
    // IO buffers
    ID3D11Buffer*               buf_input = nullptr;
    ID3D11ShaderResourceView*   buf_input_srv = nullptr;
    ID3D11UnorderedAccessView*  buf_input_uav = nullptr;
    ID3D11Buffer*               buf_output = nullptr;
    ID3D11ShaderResourceView*   buf_output_srv = nullptr;
    ID3D11UnorderedAccessView*  buf_output_uav = nullptr;
    // Group/Block sync buffers (Limited by HW, "experimental")
    ID3D11Buffer*               buf_sin = nullptr;
    ID3D11UnorderedAccessView*  buf_sin_uav = nullptr;
    ID3D11Buffer*               buf_sout = nullptr;
    ID3D11UnorderedAccessView*  buf_sout_uav = nullptr;

    ID3D11Buffer*               buf_staging = nullptr;
    ID3D11Buffer*               buf_constant = nullptr;
    ID3DBlob*                   blob_local = nullptr;
    ID3DBlob*                   blob_local_col = nullptr;
    ID3DBlob*                   blob_global = nullptr;
    ID3DBlob*                   blob_transpose = nullptr;
    ID3D11ComputeShader*        cs_local = nullptr;
    ID3D11ComputeShader*        cs_local_col = nullptr;
    ID3D11ComputeShader*        cs_global = nullptr;
    ID3D11ComputeShader*        cs_transpose = nullptr;
    ID3D11ShaderResourceView*   view_nullptr = nullptr;
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

static __inline void swap_io(dx_args *a)
{
    swap<ID3D11Buffer>(&a->buf_input, &a->buf_output);
    swap<ID3D11ShaderResourceView>(&a->buf_input_srv, &a->buf_output_srv);
    swap<ID3D11UnorderedAccessView>(&a->buf_input_uav, &a->buf_output_uav);
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
    memcpy(data, in, padded_size(sizeof(cpx) * n));
    data = nullptr;
    context->Unmap(buffer, 0);
}

static __inline void dx_read_buffer(dx_args* args, ID3D11Buffer* src, cpx* dst, const int n)
{
    args->context->CopyResource(args->buf_staging, src);
    D3D11_MAPPED_SUBRESOURCE mapped_resource;
    args->context->Map(args->buf_staging, 0, D3D11_MAP_READ, 0, &mapped_resource);
    cpx* data = reinterpret_cast<cpx*>(mapped_resource.pData);
    memcpy(dst, data, sizeof(cpx) * n);
    data = nullptr;
    args->context->Unmap(args->buf_staging, 0);
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

static __inline void dx_setup(dx_args* a, cpx* in, const int n)
{
    LPCWSTR cs_file = L"Platforms/DirectX/dx_cs.hlsl";
    dx_set_dim(cs_file, n);

    a->n_groups.x = (n >> 1) > MAX_BLOCK_SIZE ? ((n >> 1) / MAX_BLOCK_SIZE) : 1;
    D3D_FEATURE_LEVEL featureLevel;
    D3D11_BUFFER_DESC rw_buffer_desc = get_output_buffer_description(n);
    D3D11_BUFFER_DESC staging_buffer_desc = get_staging_buffer_description(n);
    D3D11_BUFFER_DESC constant_buffer_desc = get_constant_buffer_description();
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = get_unordered_access_view_description(n);
    ID3DBlob* errorBlob = nullptr;

    dx_check_error(D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, NULL, NULL, 0, D3D11_SDK_VERSION, &a->device, &featureLevel, &a->context), "D3D11CreateDevice");

    // GPU read/write accessible buffers.
    dx_check_error(a->device->CreateBuffer(&rw_buffer_desc, NULL, &a->buf_input), "Create GPU In Buffer ");
    dx_check_error(a->device->CreateBuffer(&rw_buffer_desc, NULL, &a->buf_output), "Create GPU Out Buffer ");

    // Create shader resource view for IO buffers.  
    dx_check_error(a->device->CreateShaderResourceView(a->buf_input, NULL, &a->buf_input_srv), "Create CPU Buffer ShaderResourceView");
    dx_check_error(a->device->CreateShaderResourceView(a->buf_output, NULL, &a->buf_output_srv), "Create CPU Buffer ShaderResourceView");

    // Create unordered access view for IO buffers.  
    dx_check_error(a->device->CreateUnorderedAccessView(a->buf_input, &uav_desc, &a->buf_input_uav), "Create GPU In UnorderedAccessView");
    dx_check_error(a->device->CreateUnorderedAccessView(a->buf_output, &uav_desc, &a->buf_output_uav), "Create GPU Out UnorderedAccessView");

    // Create a staging buffer, which will be used to copy back from the GPU out buffer.
    dx_check_error(a->device->CreateBuffer(&staging_buffer_desc, NULL, &a->buf_staging), "Create Staging Buffer");

    // Create a constant buffer (this buffer is used to pass the constant value 'a' to the kernel as cbuffer Constants).
    dx_check_error(a->device->CreateBuffer(&constant_buffer_desc, NULL, &a->buf_constant), "Create Constant Buffer");

    // Compile the compute shader into a blob.
    dx_check_error(D3DCompileFromFile(cs_file, NULL, NULL, "dx_local", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &a->blob_local, &errorBlob), "D3DCompileFromFile", errorBlob);
    dx_check_error(D3DCompileFromFile(cs_file, NULL, NULL, "dx_global", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &a->blob_global, &errorBlob), "D3DCompileFromFile", errorBlob);

    // Create a shader object from the compiled blob.
    dx_check_error(a->device->CreateComputeShader(a->blob_local->GetBufferPointer(), a->blob_local->GetBufferSize(), NULL, &a->cs_local), "CreateComputeShader");
    dx_check_error(a->device->CreateComputeShader(a->blob_global->GetBufferPointer(), a->blob_global->GetBufferSize(), NULL, &a->cs_global), "CreateComputeShader");
    
    // Attach the constant buffer
    a->context->CSSetConstantBuffers(0, 1, &a->buf_constant);

    if (a->n_groups.x > 1 && a->n_groups.x == 1) {
        dx_check_error(a->device->CreateBuffer(&rw_buffer_desc, NULL, &a->buf_sin), "Create GPU Sync Buffer ");
        dx_check_error(a->device->CreateBuffer(&rw_buffer_desc, NULL, &a->buf_sout), "Create GPU Sync Buffer ");
        dx_check_error(a->device->CreateUnorderedAccessView(a->buf_sin, &uav_desc, &a->buf_sin_uav), "Create GPU Out UnorderedAccessView");
        dx_check_error(a->device->CreateUnorderedAccessView(a->buf_sout, &uav_desc, &a->buf_sout_uav), "Create GPU Out UnorderedAccessView");
    }

    if (in != NULL) {
        a->context->UpdateSubresource(a->buf_input, 0, nullptr, &in[0], 0, 0);
    }
}

static __inline void dx_setup_2d(dx_args* a, cpx* in, const int n)
{
    LPCWSTR cs_file = L"Platforms/DirectX/dx_cs.hlsl";
    LPCWSTR cs_file_transpose = L"Platforms/DirectX/dx_cs_transpose.hlsl";
    dx_set_dim_2d(cs_file, n);
    dx_set_dim_trans(cs_file_transpose, n);

    a->n_groups.x = n;
    int n_half = n >> 1;
    a->n_groups.y = n_half > MAX_BLOCK_SIZE ? n_half / MAX_BLOCK_SIZE : 1;

    D3D_FEATURE_LEVEL featureLevel;
    D3D11_BUFFER_DESC rw_buffer_desc = get_output_buffer_description(n * n);
    D3D11_BUFFER_DESC staging_buffer_desc = get_staging_buffer_description(n * n);    
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = get_unordered_access_view_description(n * n);
    D3D11_BUFFER_DESC constant_buffer_desc = get_constant_buffer_description();
    ID3DBlob* errorBlob = nullptr;

    dx_check_error(D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, NULL, NULL, 0, D3D11_SDK_VERSION, &a->device, &featureLevel, &a->context), "D3D11CreateDevice");

    // GPU read/write accessible buffers.
    dx_check_error(a->device->CreateBuffer(&rw_buffer_desc, NULL, &a->buf_input), "Create GPU In Buffer ");
    dx_check_error(a->device->CreateBuffer(&rw_buffer_desc, NULL, &a->buf_output), "Create GPU Out Buffer ");

    // Create shader resource view for IO buffers.  
    dx_check_error(a->device->CreateShaderResourceView(a->buf_input, NULL, &a->buf_input_srv), "Create CPU Buffer ShaderResourceView");
    dx_check_error(a->device->CreateShaderResourceView(a->buf_output, NULL, &a->buf_output_srv), "Create CPU Buffer ShaderResourceView");

    // Create unordered access view for IO buffers.  
    dx_check_error(a->device->CreateUnorderedAccessView(a->buf_input, &uav_desc, &a->buf_input_uav), "Create GPU In UnorderedAccessView");
    dx_check_error(a->device->CreateUnorderedAccessView(a->buf_output, &uav_desc, &a->buf_output_uav), "Create GPU Out UnorderedAccessView");

    // Create a staging buffer, which will be used to copy back from the GPU out buffer.
    dx_check_error(a->device->CreateBuffer(&staging_buffer_desc, NULL, &a->buf_staging), "Create Staging Buffer");

    // Create a constant buffer (this buffer is used to pass the constant value 'a' to the kernel as cbuffer Constants).
    dx_check_error(a->device->CreateBuffer(&constant_buffer_desc, NULL, &a->buf_constant), "Create Constant Buffer");

    // Attach the constant buffer
    a->context->CSSetConstantBuffers(0, 1, &a->buf_constant);

    // Compile the compute shader into a blob.
    dx_check_error(D3DCompileFromFile(cs_file, NULL, NULL, "dx_2d_local_row", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &a->blob_local, &errorBlob), "D3DCompileFromFile", errorBlob);
    dx_check_error(D3DCompileFromFile(cs_file, NULL, NULL, "dx_2d_local_col", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &a->blob_local_col, &errorBlob), "D3DCompileFromFile", errorBlob);
    dx_check_error(D3DCompileFromFile(cs_file, NULL, NULL, "dx_2d_global", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &a->blob_global, &errorBlob), "D3DCompileFromFile", errorBlob);
    dx_check_error(D3DCompileFromFile(cs_file_transpose, NULL, NULL, "dx_transpose", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &a->blob_transpose, &errorBlob), "D3DCompileFromFile", errorBlob);

    // Create a shader object from the compiled blob.
    dx_check_error(a->device->CreateComputeShader(a->blob_local->GetBufferPointer(), a->blob_local->GetBufferSize(), NULL, &a->cs_local), "CreateComputeShader");
    dx_check_error(a->device->CreateComputeShader(a->blob_local_col->GetBufferPointer(), a->blob_local_col->GetBufferSize(), NULL, &a->cs_local_col), "CreateComputeShader");
    dx_check_error(a->device->CreateComputeShader(a->blob_global->GetBufferPointer(), a->blob_global->GetBufferSize(), NULL, &a->cs_global), "CreateComputeShader");
    dx_check_error(a->device->CreateComputeShader(a->blob_transpose->GetBufferPointer(), a->blob_transpose->GetBufferSize(), NULL, &a->cs_transpose), "CreateComputeShader");
        
    if (in != NULL) {
        a->context->UpdateSubresource(a->buf_input, 0, nullptr, &in[0], 0, 0);
    }
}

static __inline void dx_shakedown(dx_args *a)
{
    ID3D11UnorderedAccessView* nullUAV = nullptr;
    a->context->CSSetUnorderedAccessViews(0, 1, &nullUAV, &a->init_cnts);
    ID3D11ShaderResourceView* nullSRV = nullptr;
    a->context->CSSetShaderResources(0, 1, &nullSRV);
    ID3D11Buffer* nullBuffer = nullptr;
    a->context->CSSetConstantBuffers(0, 1, &nullBuffer);

    if (a->cs_local)
        a->cs_local->Release();
    if (a->cs_local_col)
        a->cs_local_col->Release();
    if (a->cs_global)
        a->cs_global->Release();
    if (a->blob_local)
        a->blob_local->Release();
    if (a->blob_local_col)
        a->blob_local_col->Release();
    if (a->blob_global)
        a->blob_global->Release();
    if (a->buf_constant)
        a->buf_constant->Release();
    if (a->buf_staging)
        a->buf_staging->Release();
    a->buf_output_uav->Release();
    a->buf_output_srv->Release();
    a->buf_output->Release();
    a->buf_input_uav->Release();
    a->buf_input_srv->Release();
    a->buf_input->Release();
    if (a->buf_sin) {
        a->context->CSSetUnorderedAccessViews(1, 1, &nullUAV, &a->init_cnts);
        a->buf_sin->Release();
        a->buf_sin_uav->Release();
    }
    if (a->buf_sout) {
        a->context->CSSetUnorderedAccessViews(2, 1, &nullUAV, &a->init_cnts);
        a->buf_sout->Release();
        a->buf_sout_uav->Release();
    }
    if (a->blob_transpose) {
        a->cs_transpose->Release();
        a->blob_transpose->Release();
    }
    a->context->Release();
    a->device->Release();
}

#endif