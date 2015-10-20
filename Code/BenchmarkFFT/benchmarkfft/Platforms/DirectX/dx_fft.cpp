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
    write_normalized_image("C_C++", "freq", in, n, true);
    dx_fft_2d(FFT_INVERSE, &in, &buf, n);
    write_image("C_C++", "spat", in, n);

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

HRESULT CompileComputeShader(_In_ LPCWSTR srcFile, _In_ LPCSTR entryPoint,
    _In_ ID3D11Device* device, _Outptr_ ID3DBlob** blob)
{
    if (!srcFile || !entryPoint || !device || !blob)
        return E_INVALIDARG;

    *blob = nullptr;

    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined( DEBUG ) || defined( _DEBUG )
    flags |= D3DCOMPILE_DEBUG;
#endif

    // We generally prefer to use the higher CS shader profile when possible as CS 5.0 is better performance on 11-class hardware
    LPCSTR profile = (device->GetFeatureLevel() >= D3D_FEATURE_LEVEL_11_0) ? "cs_5_0" : "cs_4_0";

    const D3D_SHADER_MACRO defines[] =
    {
        "EXAMPLE_DEFINE", "1",
        NULL, NULL
    };

    ID3DBlob* shaderBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;
    HRESULT hr = D3DCompileFromFile(srcFile, defines, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entryPoint, profile,
        flags, 0, &shaderBlob, &errorBlob);
    if (FAILED(hr))
    {
        if (errorBlob)
        {
            OutputDebugStringA((char*)errorBlob->GetBufferPointer());
            errorBlob->Release();
        }

        if (shaderBlob)
            shaderBlob->Release();

        return hr;
    }

    *blob = shaderBlob;

    return hr;
}

__inline void dx_fft(fftDir dir, cpx **in, cpx **out, const int n)
{
    return;
}

__inline void dx_fft_2d(fftDir dir, cpx **in, cpx **out, const int n)
{
    return;
}

void dx_fft()
{
    // Create Device
    const D3D_FEATURE_LEVEL lvl[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };

    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    ID3D11Device* device = nullptr;
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags, lvl, _countof(lvl), D3D11_SDK_VERSION, &device, nullptr, nullptr);
    if (hr == E_INVALIDARG) {
        // DirectX 11.0 Runtime doesn't recognize D3D_FEATURE_LEVEL_11_1 as a valid value
        hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &lvl[1], _countof(lvl) - 1, D3D11_SDK_VERSION, &device, nullptr, nullptr);
    }

    if (FAILED(hr)) {
        printf("Failed creating Direct3D 11 device %08X\n", hr);
        return;
    }

    // Compile shader
    ID3DBlob *csBlob = nullptr;
    hr = CompileComputeShader(L"Platforms/DirectX/dx_compute_shader.hlsl", "CSMain", device, &csBlob);
    if (FAILED(hr)) {
        device->Release();
        printf("Failed compiling shader %08X\n", hr);
        return;
    }

    // Create shader
    ID3D11ComputeShader* computeShader = nullptr;
    hr = device->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(), nullptr, &computeShader);
    csBlob->Release();
    if (FAILED(hr)) {
        device->Release();
    }
    printf("Success\n");

    // Clean up
    computeShader->Release();
    device->Release();
}