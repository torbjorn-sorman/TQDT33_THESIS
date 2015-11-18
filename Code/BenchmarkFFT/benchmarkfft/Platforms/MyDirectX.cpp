#include "MyDirectX.h"

MyDirectX::MyDirectX(const int dim, const int runs)
    : Platform(dim)
{
    name = "DirectX";
    const D3D_FEATURE_LEVEL feature_levels[2] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    ID3D11Device *device;
    ID3D11DeviceContext *context;
    D3D_FEATURE_LEVEL feature_level;
    IDXGIAdapter1 *adapter = dx_get_adapter();
    HRESULT hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, NULL, NULL, feature_levels, 2, D3D11_SDK_VERSION, &device, &feature_level, &context);
    if (hr == S_OK) {
        if (feature_level == D3D_FEATURE_LEVEL_11_0)
            printf("DirectCompute D3D_FEATURE_LEVEL_11_0\n");
        else if (feature_level == D3D_FEATURE_LEVEL_11_1)
            printf("DirectCompute D3D_FEATURE_LEVEL_11_1\n");
    }
    else {
        printf("DirectCompute Failed...");
        getchar();
        exit(1);
    }
    context->Release();
    device->Release();    
    adapter->Release();
}

MyDirectX::~MyDirectX()
{
}

bool MyDirectX::validate(const int n, bool write_img)
{
    if (dimensions == 1)
        return dx_validate(n) == 1;
    return dx_2d_validate(n, write_img) == 1;
}

void MyDirectX::runPerformance(const int n)
{
    double time = (dimensions == 1) ? dx_performance(n) : dx_2d_performance(n);
    results.push_back(time);
}
