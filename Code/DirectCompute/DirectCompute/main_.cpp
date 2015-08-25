#define _WIN32_WINNT 0x600
#include <stdio.h>

#include <d3d11.h>
#include <d3dcompiler.h>

// The number of elements in a buffer to be tested
const UINT NUM_ELEMENTS = 1024;


//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------
HRESULT CreateComputeDevice(_Outptr_ ID3D11Device** ppDeviceOut, _Outptr_ ID3D11DeviceContext** ppContextOut, _In_ bool bForceRef);
HRESULT CreateComputeShader(_In_z_ LPCWSTR pSrcFile, _In_z_ LPCSTR pFunctionName,
    _In_ ID3D11Device* pDevice, _Outptr_ ID3D11ComputeShader** ppShaderOut);
HRESULT CreateStructuredBuffer(_In_ ID3D11Device* pDevice, _In_ UINT uElementSize, _In_ UINT uCount,
    _In_reads_(uElementSize*uCount) void* pInitData,
    _Outptr_ ID3D11Buffer** ppBufOut);
HRESULT CreateRawBuffer(_In_ ID3D11Device* pDevice, _In_ UINT uSize, _In_reads_(uSize) void* pInitData, _Outptr_ ID3D11Buffer** ppBufOut);
HRESULT CreateBufferSRV(_In_ ID3D11Device* pDevice, _In_ ID3D11Buffer* pBuffer, _Outptr_ ID3D11ShaderResourceView** ppSRVOut);
HRESULT CreateBufferUAV(_In_ ID3D11Device* pDevice, _In_ ID3D11Buffer* pBuffer, _Outptr_ ID3D11UnorderedAccessView** pUAVOut);
ID3D11Buffer* CreateAndCopyToDebugBuf(_In_ ID3D11Device* pDevice, _In_ ID3D11DeviceContext* pd3dImmediateContext, _In_ ID3D11Buffer* pBuffer);
void RunComputeShader(_In_ ID3D11DeviceContext* pd3dImmediateContext,
    _In_ ID3D11ComputeShader* pComputeShader,
    _In_ UINT nNumViews, _In_reads_(nNumViews) ID3D11ShaderResourceView** pShaderResourceViews,
    _In_opt_ ID3D11Buffer* pCBCS, _In_reads_opt_(dwNumDataBytes) void* pCSData, _In_ DWORD dwNumDataBytes,
    _In_ ID3D11UnorderedAccessView* pUnorderedAccessView,
    _In_ UINT X, _In_ UINT Y, _In_ UINT Z);
HRESULT FindDXSDKShaderFileCch(_Out_writes_(cchDest) WCHAR* strDestPath,
    _In_ int cchDest,
    _In_z_ LPCWSTR strFilename);

//--------------------------------------------------------------------------------------
// Running DEBUG in VS, wait for key
//--------------------------------------------------------------------------------------
inline int returnDBG(int val)
{
#ifdef _DEBUG 
    getchar();
#endif
    return val;
}

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
ID3D11Device*               g_pDevice = nullptr;
ID3D11DeviceContext*        g_pContext = nullptr;
ID3D11ComputeShader*        g_pCS = nullptr;

ID3D11Buffer*               g_pBuf0 = nullptr;
ID3D11Buffer*               g_pBuf1 = nullptr;
ID3D11Buffer*               g_pBufResult = nullptr;

ID3D11ShaderResourceView*   g_pBuf0SRV = nullptr;
ID3D11ShaderResourceView*   g_pBuf1SRV = nullptr;
ID3D11UnorderedAccessView*  g_pBufResultUAV = nullptr;

struct BufType
{
    int i;
    float f;
#ifdef TEST_DOUBLE
    double d;
#endif
};
BufType g_vBuf0[NUM_ELEMENTS];
BufType g_vBuf1[NUM_ELEMENTS];


HRESULT CompileComputeShader(_In_ LPCWSTR srcFile, _In_ LPCSTR entryPoint, _In_ ID3D11Device* device, _Outptr_ ID3DBlob** blob)
{
    if (!srcFile || !entryPoint || !device || !blob)
        return E_INVALIDARG;

    *blob = nullptr;

    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined( DEBUG ) || defined( _DEBUG )
    flags |= D3DCOMPILE_DEBUG;
#endif

    const D3D_SHADER_MACRO defines[] =
    {
        "EXAMPLE_DEFINE", "1", NULL, NULL
    };

    ID3DBlob* shaderBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;
    HRESULT hr = D3DCompileFromFile(srcFile, defines, D3D_COMPILE_STANDARD_FILE_INCLUDE, entryPoint, "cs_5_0", flags, 0, &shaderBlob, &errorBlob);
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

int main()
{
    // Create Device
    const D3D_FEATURE_LEVEL lvl[] = { D3D_FEATURE_LEVEL_11_1 };

    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    /* ================================== */
    printf("Creating device...");
    ID3D11Device* device = nullptr;
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags, lvl, _countof(lvl), D3D11_SDK_VERSION, &device, nullptr, nullptr);
    if (FAILED(hr))
    {
        printf("Failed creating Direct3D 11 device %08X\n", hr);
        return returnDBG(-1);
    }

    /* ================================== */
    printf("Compile Compute Shader...");
    ID3DBlob *csBlob = nullptr;
    hr = CompileComputeShader(L"ComputeShader.hlsl", "main", device, &csBlob);
    if (FAILED(hr))
    {
        device->Release();
        printf("Failed compiling shader %08X\n", hr);
        return returnDBG(-1);
    }

    /* ================================== */
    printf("Create Compute Shader...");
    ID3D11ComputeShader* computeShader = nullptr;
    hr = device->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(), nullptr, &computeShader);
            
    /* ================================== */
    printf("Creating buffers and filling them with initial data...");
    for (int i = 0; i < NUM_ELEMENTS; ++i)
    {
        g_vBuf0[i].i = i;
        g_vBuf0[i].f = (float)i;

        g_vBuf1[i].i = i;
        g_vBuf1[i].f = (float)i;
    }

    CreateRawBuffer(g_pDevice, NUM_ELEMENTS * sizeof(BufType), &g_vBuf0[0], &g_pBuf0);
    CreateRawBuffer(g_pDevice, NUM_ELEMENTS * sizeof(BufType), &g_vBuf1[0], &g_pBuf1);
    CreateRawBuffer(g_pDevice, NUM_ELEMENTS * sizeof(BufType), nullptr, &g_pBufResult);

    /* ================================== */
    printf("Running Compute Shader...");
    ID3D11ShaderResourceView* aRViews[2] = { g_pBuf0SRV, g_pBuf1SRV };
    RunComputeShader(g_pContext, g_pCS, 2, aRViews, nullptr, nullptr, 0, g_pBufResultUAV, NUM_ELEMENTS, 1, 1);
    printf("done\n");
    
    
    
    csBlob->Release();

    if (FAILED(hr))
    {
        device->Release();
    }

    printf("Success\n");



    // Clean up
    computeShader->Release();

    device->Release();

    return returnDBG(0);
}
