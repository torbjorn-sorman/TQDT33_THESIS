#pragma once
#ifndef DX_HELPER_H
#define DX_HELPER_H

#include <d3d11.h>
#include <D3Dcompiler.h>

#include <comdef.h>
#include <vector>
#include <iostream>

#include "../../Definitions.h"
#include "../../Common/mathutil.h"
#include "../../Common/myfile.h"


_COM_SMARTPTR_TYPEDEF(ID3D11Query, __uuidof(ID3D11Query));

struct dx_args
{
    dim3                        n_groups = { 1, 1, 1 };
    int                         number_of_blocks = 1;
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

    ID3D11Buffer*               buf_staging = nullptr;
    ID3D11Buffer*               buf_constant = nullptr;
    ID3DBlob*                   blob_local = nullptr;
    ID3DBlob*                   blob_global = nullptr;
    ID3DBlob*                   blob_transpose = nullptr;
    ID3D11ComputeShader*        cs_local = nullptr;
    ID3D11ComputeShader*        cs_global = nullptr;
    ID3D11ComputeShader*        cs_transpose = nullptr;
    ID3D11ShaderResourceView*   view_nullptr = nullptr;
};

struct dx_cs_args
{
    float           global_angle;
    float           local_angle;
    float           scalar;
    int             steps_left;
    int             leading_bits;
    int             steps_gpu;
    int             number_of_blocks;
    int             block_range;
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

static __inline void dx_start_profiling(ID3D11Device *device, ID3D11DeviceContext *context, profiler_data *p_data)
{
    if (p_data->disjoint_query == NULL)
    {
        D3D11_QUERY_DESC desc;
        desc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
        desc.MiscFlags = 0;
        device->CreateQuery(&desc, &p_data->disjoint_query);
        desc.Query = D3D11_QUERY_TIMESTAMP;
        device->CreateQuery(&desc, &p_data->q_start);
        device->CreateQuery(&desc, &p_data->q_end);
    }
    context->Begin(p_data->disjoint_query);
    context->End(p_data->q_start);
}

static __inline void dx_end_profiling(ID3D11DeviceContext *context, profiler_data *p_data)
{
    context->End(p_data->q_end);
    context->End(p_data->disjoint_query);
}

template<typename T> static __inline void dx_swap(T **a, T **b)
{
    T *c = *a;
    *a = *b;
    *b = c;
}

static __inline void swap_io(dx_args *a)
{
    dx_swap<ID3D11Buffer>(&a->buf_input, &a->buf_output);
    dx_swap<ID3D11ShaderResourceView>(&a->buf_input_srv, &a->buf_output_srv);
    dx_swap<ID3D11UnorderedAccessView>(&a->buf_input_uav, &a->buf_output_uav);
}

template<typename T> static __inline void dx_map_args(ID3D11DeviceContext* context, ID3D11Buffer* arg_buffer, T *params)
{
    D3D11_MAPPED_SUBRESOURCE mapped_resource = { 0 };
    context->Map(arg_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped_resource);
    if (mapped_resource.pData) {
        T* constants = reinterpret_cast<T *>(mapped_resource.pData);
        constants[0] = *params;
        constants = nullptr;
        context->Unmap(arg_buffer, 0);
    }
}

double dx_avg(profiler_data profiler[], dx_args *args);
double dx_avg(profiler_data profiler[], ID3D11DeviceContext *context);
double dx_time_elapsed(profiler_data *p, dx_args *args);
double dx_time_elapsed(profiler_data *p, ID3D11DeviceContext *context);

size_t padded_size(size_t sz, size_t width);
size_t padded_size(size_t sz);

void dx_check_error(HRESULT hr, char *method, ID3DBlob* error_blob);
void dx_check_error(HRESULT hr, char *method);

IDXGIAdapter1 *dx_get_adapter(int vendor);

D3D11_BUFFER_DESC get_output_buffer_description(const int dimension);
D3D11_UNORDERED_ACCESS_VIEW_DESC get_unordered_access_view_description(const int dimension);
D3D11_BUFFER_DESC get_staging_buffer_description(const int dimension);
D3D11_BUFFER_DESC get_constant_buffer_description();

void dx_write_buffer(ID3D11DeviceContext* context, ID3D11Buffer* buffer, cpx* in, const int n);
void dx_read_buffer(dx_args* args, ID3D11Buffer* src, cpx* dst, const int n);
void dx_setup(dx_args* a, cpx* in, int group_size, const int n);
void dx_setup_2d(dx_args* a, cpx* in, int group_size, int tile_dim, const int n);
void dx_shakedown(dx_args *a);

#endif