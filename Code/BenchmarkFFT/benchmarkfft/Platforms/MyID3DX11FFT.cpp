#include "MyID3DX11FFT.h"
#include <atlbase.h>

MyID3DX11FFT::MyID3DX11FFT(const int dim, const int runs)
    : Platform(dim)
{
    name = "ID3DX11FFT";
}

MyID3DX11FFT::~MyID3DX11FFT()
{
}

bool MyID3DX11FFT::validate(const int n, bool write_img)
{
    return true;
}

HRESULT CreateByteOrderBufferOnGPU(ID3D11Device* pDevice, UINT uElementSize, UINT uCount, ID3D11Buffer** ppBufOut)
{
    *ppBufOut = NULL;
    D3D11_BUFFER_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    desc.ByteWidth = uElementSize * uCount;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    desc.StructureByteStride = uElementSize;
    return pDevice->CreateBuffer(&desc, NULL, ppBufOut);
}

HRESULT CreateBufferUAV(ID3D11Device* pDevice, ID3D11Buffer* pBuffer, ID3D11UnorderedAccessView** ppUAVOut)
{
    D3D11_BUFFER_DESC descBuf;
    ZeroMemory(&descBuf, sizeof(descBuf));
    pBuffer->GetDesc(&descBuf);

    D3D11_UNORDERED_ACCESS_VIEW_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    desc.Buffer.FirstElement = 0;

    if (descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS)
    {
        // This is a Raw Buffer

        desc.Format = DXGI_FORMAT_R32_TYPELESS; // Format must be DXGI_FORMAT_R32_TYPELESS, when creating Raw Unordered Access View
        desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        desc.Buffer.NumElements = descBuf.ByteWidth / 4;
    }
    else
        if (descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED)
        {
            // This is a Structured Buffer

            desc.Format = DXGI_FORMAT_UNKNOWN;      // Format must be must be DXGI_FORMAT_UNKNOWN, when creating a View of a Structured Buffer
            desc.Buffer.NumElements = descBuf.ByteWidth / descBuf.StructureByteStride;
        }
        else
        {
            return E_INVALIDARG;
        }

    return pDevice->CreateUnorderedAccessView(pBuffer, &desc, ppUAVOut);
}

CComPtr<ID3D11UnorderedAccessView> get_uav(ID3D11Device* device, UINT float_size)
{
    CComPtr<ID3D11Buffer> buf = NULL;
    CComPtr<ID3D11UnorderedAccessView> buf_uav = NULL;
    dx_check_error(CreateByteOrderBufferOnGPU(device, sizeof(float), float_size, &buf), "CreateByteOrderBufferOnGPU");
    dx_check_error(CreateBufferUAV(device, buf, &buf_uav), "CreateBufferUAV");
    return buf_uav;
}

void MyID3DX11FFT::runPerformance(const int n)
{
    CComPtr<ID3D11Device> device = NULL;
    CComPtr<ID3D11DeviceContext> context = NULL;
    D3DX11_FFT_BUFFER_INFO buffer_info;
    CComPtr<ID3DX11FFT> fft = 0;
    const D3D_FEATURE_LEVEL feature_levels[1] = { D3D_FEATURE_LEVEL_11_0 };
    dx_check_error(D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, NULL, feature_levels, 1, D3D11_SDK_VERSION, &device, NULL, &context), "D3D11CreateDevice");

    if (dimensions == 1)
        dx_check_error(D3DX11CreateFFT1DComplex(context, n, 0, &buffer_info, &fft), "D3DX11CreateFFT1DComplex");
    else
        dx_check_error(D3DX11CreateFFT2DComplex(context, n, n, 0, &buffer_info, &fft), "D3DX11CreateFFT2DComplex");

    CComPtr<ID3D11UnorderedAccessView> buf_uav = get_uav(device, (UINT)buffer_info.TempBufferFloatSizes[0]);
    CComPtr<ID3D11UnorderedAccessView> tmp1_uav = get_uav(device, (UINT)buffer_info.TempBufferFloatSizes[0]);
    CComPtr<ID3D11UnorderedAccessView> tmp2_uav = get_uav(device, (UINT)buffer_info.TempBufferFloatSizes[1]);
    fft->SetForwardScale(1.0f);

    ID3D11UnorderedAccessView *vect[] = { tmp1_uav, tmp2_uav };
    dx_check_error(fft->AttachBuffersAndPrecompute(2, vect, 0, 0), "AttachBuffersAndPrecompute");

    profiler_data profiler[NUM_TESTS];
    for (int i = 0; i < NUM_TESTS; ++i) {
        profiler_data p;
        CComPtr<ID3D11UnorderedAccessView> resp_uav;
        dx_start_profiling(device, context, &p);
        fft->ForwardTransform(buf_uav, &resp_uav);
        dx_end_profiling(context, &p);
        profiler[i] = p;
    }
    results.push_back(dx_avg(profiler, context));
}
