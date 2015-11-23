#include "MyDirectX.h"

MyDirectX::MyDirectX(const int dim, const int runs)
    : Platform(dim)
{
    name = "DirectX";
    const D3D_FEATURE_LEVEL feature_levels[2] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    ID3D11DeviceContext *context;
    D3D_FEATURE_LEVEL feature_level;
    IDXGIAdapter1 *adapter = dx_get_adapter(vendor_gpu);
    HRESULT hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, NULL, NULL, feature_levels, 2, D3D11_SDK_VERSION, NULL, &feature_level, &context);
    if (hr == S_OK) {
        switch (feature_level)
        {
        case D3D_FEATURE_LEVEL_9_1:
            break;
        case D3D_FEATURE_LEVEL_9_2:
            break;
        case D3D_FEATURE_LEVEL_9_3:
            break;
        case D3D_FEATURE_LEVEL_10_0:
            break;
        case D3D_FEATURE_LEVEL_10_1:
            break;
        case D3D_FEATURE_LEVEL_11_0:
            printf("DirectCompute\tD3D_FEATURE_LEVEL_11_0 (GTX670)\n");
            break;
        case D3D_FEATURE_LEVEL_11_1:
            printf("DirectCompute\tD3D_FEATURE_LEVEL_11_0 (R7 260X)\n");
            break;
        default:
            break;
        }
    }
    else {
        printf("DirectCompute Failed...");
        getchar();
        exit(1);
    }
    context->Release();
    adapter->Release();
}

MyDirectX::~MyDirectX()
{
}

bool MyDirectX::validate(const int n, bool write_img)
{
    if (dimensions == 1) {
#if defined(_AMD)
        if (n > power2(24))
            return false;    
#endif
        return dx_validate(n) == 1;
    }
    else {
#if defined(_AMD)
        if (n > power2(13))
            return false;
#endif
        return dx_2d_validate(n, write_img) == 1;
    }
}

void MyDirectX::runPerformance(const int n)
{
    double time;
    if (dimensions == 1) {
#if defined(_AMD)
        if (n > power2(24)){
            results.push_back(-1);
            return;
        }
#endif
        time = dx_performance(n);
    }
    else {
#if defined(_AMD)
        if (n > power2(13)){
            results.push_back(-1);
            return;
        }
#endif
        time = dx_2d_performance(n);
    }
    results.push_back(time);
}
