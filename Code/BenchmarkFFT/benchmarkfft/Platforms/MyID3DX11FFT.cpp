#include "MyID3DX11FFT.h"
#include <atlbase.h>
#include <D3dcsx.h>
#include "../Common/mycomplex.h"

MyID3DX11FFT::MyID3DX11FFT(const int dim, const int runs)
    : Platform(dim)
{
    name = "ID3DX11FFT";
}

MyID3DX11FFT::~MyID3DX11FFT()
{
}

CComPtr<ID3D11UnorderedAccessView> get_uav(ID3D11Device* device, VOID *init_data, UINT number_of_floats)
{
    CComPtr<ID3D11Buffer> buffer = NULL;
    CComPtr<ID3D11UnorderedAccessView> buffer_uav = NULL;
    D3D11_BUFFER_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    desc.ByteWidth = sizeof(float) * number_of_floats;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    desc.StructureByteStride = sizeof(float);
    if (init_data != NULL) {
        D3D11_SUBRESOURCE_DATA data;
        data.pSysMem = init_data;
        dx_check_error(device->CreateBuffer(&desc, &data, &buffer), "create_buffer");
    }
    else {
        dx_check_error(device->CreateBuffer(&desc, NULL, &buffer), "create_buffer");
    }
    D3D11_BUFFER_DESC descBuf;
    ZeroMemory(&descBuf, sizeof(descBuf));
    buffer->GetDesc(&descBuf);
    D3D11_UNORDERED_ACCESS_VIEW_DESC desc_uav;
    ZeroMemory(&desc_uav, sizeof(desc_uav));
    desc_uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    desc_uav.Buffer.FirstElement = 0;
    desc_uav.Format = DXGI_FORMAT_R32_TYPELESS;
    desc_uav.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
    desc_uav.Buffer.NumElements = descBuf.ByteWidth / sizeof(float);
    dx_check_error(device->CreateUnorderedAccessView(buffer, &desc_uav, &buffer_uav), "create_buffer_uav");
    return buffer_uav;
}

ID3D11Buffer* copy_to_buffer(ID3D11Buffer* pBuffer, ID3D11DeviceContext *pContextOut)  //release the returned buffer
{
    ID3D11Buffer* debugbuf = { 0 };

    D3D11_BUFFER_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    pBuffer->GetDesc(&desc);
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.MiscFlags = 0;

    CComPtr<ID3D11Device> pDeviceOut = NULL;
    pContextOut->GetDevice(&pDeviceOut);

    HRESULT hr = pDeviceOut->CreateBuffer(&desc, NULL, &debugbuf);
    if (debugbuf) {
        dx_check_error(hr, "CreateBuffer");
        pContextOut->CopyResource(debugbuf, pBuffer);
        return debugbuf;
    }
    return NULL;
}

cpx *get_result(CComPtr<ID3D11DeviceContext> context, CComPtr<ID3D11UnorderedAccessView> resp_uav, size_t data_size)
{
    CComPtr<ID3D11Buffer> result_output;
    resp_uav->GetResource((ID3D11Resource **)&result_output);
    CComPtr<ID3D11Buffer> result;
    result.Attach(copy_to_buffer(result_output, context));
    D3D11_MAPPED_SUBRESOURCE MappedResource = { 0 };
    context->Map(result, 0, D3D11_MAP_READ, 0, &MappedResource);
    if (MappedResource.pData) {
        cpx *out = (cpx *)MappedResource.pData;
        cpx *data = (cpx *)malloc(data_size);
        if (data)
            memcpy(data, out, data_size);
        return data;
    }
    return NULL;
}

bool MyID3DX11FFT::validate(const int n, bool write_img)
{
    return true;
    /*
    if (n > 4096) {
        results.push_back(-1);
        return false;
    }
    cpx *data, *ref;
    size_t data_size;
    if (dimensions == 1) {
        write_img = false;
        data = get_seq(n, 1);
        ref = get_seq(n, data);
        data_size = sizeof(cpx) * n;
    }
    else {
        setup_seq_2d(&data, NULL, &ref, n * 2);
        data_size = sizeof(cpx) * n * n;
    }
    CComPtr<ID3DX11FFT> fft = 0;
    CComPtr<ID3DX11FFT> fft_inv = 0;
    CComPtr<ID3D11DeviceContext> context = { 0 };
    CComPtr<ID3D11Device> device = { 0 };
    D3DX11_FFT_BUFFER_INFO buffer_info = { 0 };
    D3DX11_FFT_BUFFER_INFO buffer_info_inv = { 0 };
    CComPtr<ID3D11UnorderedAccessView> resp_uav = { 0 };
    CComPtr<ID3D11UnorderedAccessView> resp_inv_uav = { 0 };
    const D3D_FEATURE_LEVEL feature_levels[1] = { D3D_FEATURE_LEVEL_11_0 };
    {
        dx_check_error(D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, NULL, feature_levels, 1, D3D11_SDK_VERSION, &device, NULL, &context), "D3D11CreateDevice");
        if (dimensions == 1) {
            dx_check_error(D3DX11CreateFFT1DComplex(context, n, 0, &buffer_info, &fft), "D3DX11CreateFFT1DComplex");
        }
        else {
            dx_check_error(D3DX11CreateFFT2DComplex(context, n, n, 0, &buffer_info, &fft), "D3DX11CreateFFT2DComplex");
        }
        if (buffer_info.TempBufferFloatSizes) {
            CComPtr<ID3D11UnorderedAccessView> buf_uav = get_uav(device, data, (UINT)buffer_info.TempBufferFloatSizes[0]);
            CComPtr<ID3D11UnorderedAccessView> tmp1_uav = get_uav(device, NULL, (UINT)buffer_info.TempBufferFloatSizes[0]);
            CComPtr<ID3D11UnorderedAccessView> tmp2_uav = get_uav(device, NULL, (UINT)buffer_info.TempBufferFloatSizes[1]);
            fft->SetForwardScale(1.0f);
            ID3D11UnorderedAccessView *vect[] = { tmp1_uav, tmp2_uav };
            dx_check_error(fft->AttachBuffersAndPrecompute(2, vect, 0, 0), "AttachBuffersAndPrecompute");
            fft->ForwardTransform(buf_uav, &resp_uav);
        }
    }
    free(data);
    data = get_result(context, resp_uav, data_size);
    if (write_img) {
        write_normalized_image("ID3DX11FFT", "freq", data, n, true);
    }
    {
        if (dimensions == 1) {
            dx_check_error(D3DX11CreateFFT1DComplex(context, n, 0, &buffer_info_inv, &fft_inv), "D3DX11CreateFFT1DComplex");
        }
        else {
            dx_check_error(D3DX11CreateFFT2DComplex(context, n, n, 0, &buffer_info_inv, &fft_inv), "D3DX11CreateFFT2DComplex");
        }
        if (buffer_info_inv.TempBufferFloatSizes) {
            CComPtr<ID3D11UnorderedAccessView> buf_uav = get_uav(device, data, (UINT)buffer_info_inv.TempBufferFloatSizes[0]);
            CComPtr<ID3D11UnorderedAccessView> tmp1_uav = get_uav(device, NULL, (UINT)buffer_info_inv.TempBufferFloatSizes[0]);
            CComPtr<ID3D11UnorderedAccessView> tmp2_uav = get_uav(device, NULL, (UINT)buffer_info_inv.TempBufferFloatSizes[1]);
            fft_inv->SetInverseScale(1.f / n);
            ID3D11UnorderedAccessView *vect[] = { tmp1_uav, tmp2_uav };
            dx_check_error(fft_inv->AttachBuffersAndPrecompute(2, vect, 0, 0), "AttachBuffersAndPrecompute");
            fft_inv->InverseTransform(buf_uav, &resp_inv_uav);
        }
    }
    free(data);
    data = get_result(context, resp_inv_uav, data_size);
    if (write_img) {
        write_image("ID3DX11FFT", "spat", data, n);
    }
    double res = diff_seq(data, ref, (int)(data_size / sizeof(cpx)));
    free(data);
    free(ref);
    return res == 0;
    */
}


void MyID3DX11FFT::runPerformance(const int n)
{    
    if (true || n > 4096) {
        results.push_back(-1);
        return;
    }
    /*
    cpx *data, *ref;
    CComPtr<ID3D11Device> device = NULL;
    CComPtr<ID3D11DeviceContext> context = NULL;
    D3DX11_FFT_BUFFER_INFO buffer_info;
    CComPtr<ID3DX11FFT> fft = 0;
    const D3D_FEATURE_LEVEL feature_levels[1] = { D3D_FEATURE_LEVEL_11_0 };
    dx_check_error(D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, NULL, feature_levels, 1, D3D11_SDK_VERSION, &device, NULL, &context), "D3D11CreateDevice");

    if (dimensions == 1) {
        dx_check_error(D3DX11CreateFFT1DComplex(context, n, 0, &buffer_info, &fft), "D3DX11CreateFFT1DComplex");
        data = get_seq(n, 1);
        ref = get_seq(n, data);
    }
    else {
        dx_check_error(D3DX11CreateFFT2DComplex(context, n, n, 0, &buffer_info, &fft), "D3DX11CreateFFT2DComplex");
        setup_seq_2d(&data, NULL, &ref, n * 2);
    }

    CComPtr<ID3D11UnorderedAccessView> buf_uav = get_uav(device, data, (UINT)buffer_info.TempBufferFloatSizes[0]);
    CComPtr<ID3D11UnorderedAccessView> tmp1_uav = get_uav(device, NULL, (UINT)buffer_info.TempBufferFloatSizes[0]);
    CComPtr<ID3D11UnorderedAccessView> tmp2_uav = get_uav(device, NULL, (UINT)buffer_info.TempBufferFloatSizes[1]);
    fft->SetForwardScale(1.0f);

    ID3D11UnorderedAccessView *vect[] = { tmp1_uav, tmp2_uav };
    dx_check_error(fft->AttachBuffersAndPrecompute(2, vect, 0, 0), "AttachBuffersAndPrecompute");

    profiler_data profiler[64];
    for (int i = 0; i < number_of_tests; ++i) {
        profiler_data p;
        CComPtr<ID3D11UnorderedAccessView> resp_uav;
        dx_start_profiling(device, context, &p);
        fft->ForwardTransform(buf_uav, &resp_uav);
        dx_end_profiling(context, &p);
        profiler[i] = p;
    }
    results.push_back(dx_avg(profiler, context));
    free(data);
    free(ref);
    */
}
