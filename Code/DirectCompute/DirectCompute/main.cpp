#define _WIN32_WINNT 0x600

#include <Windows.h>
#include <stdio.h>

#include <d3d11.h>
#include <d3dcompiler.h>

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)      { if (p) { (p)->Release(); (p)=nullptr; } }
#endif

typedef long long int64; typedef unsigned long long uint64;

// The number of elements in a buffer to be tested
const UINT NUM_ELEMENTS = 1024 * 96;//1024 * 64;
const UINT NUM_LOOPS = 4096;

//--------------------------------------------------------------------------------------
// Get time
//--------------------------------------------------------------------------------------
uint64 GetTimeMs64()
{
    FILETIME ft;
    LARGE_INTEGER li;

    /* Get the amount of 100 nano seconds intervals elapsed since January 1, 1601 (UTC) and copy it
    * to a LARGE_INTEGER structure. */
    GetSystemTimeAsFileTime(&ft);
    li.LowPart = ft.dwLowDateTime;
    li.HighPart = ft.dwHighDateTime;

    uint64 ret = li.QuadPart;
    ret -= 116444736000000000LL; /* Convert from file time to UNIX epoch time. */
    ret /= 10000; /* From 100 nano seconds (10^-7) to 1 millisecond (10^-3) intervals */

    return ret;
}

//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------
HRESULT CreateComputeDevice(_Outptr_ ID3D11Device** ppDeviceOut, _Outptr_ ID3D11DeviceContext** ppContextOut);
HRESULT CreateComputeShader(_In_z_ LPCWSTR pSrcFile, _In_z_ LPCSTR pFunctionName,
    _In_ ID3D11Device* pDevice, _Outptr_ ID3D11ComputeShader** ppShaderOut);
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

// DEBUG
inline int pauseOnDBG(int ret)
{
#ifdef _DEBUG
    getchar();
#endif
    return ret;
}

void RunCPU(BufType *out)
{
    int ir;
    float fr;
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        ir = 0; fr = 0;

        for (int j = 0; j < NUM_LOOPS / 8; ++j)
        {
            ir += g_vBuf0[i].i + g_vBuf1[i].i;
            fr += g_vBuf0[i].f + g_vBuf1[i].f;
            ir += g_vBuf0[i].i + g_vBuf1[i].i;
            fr += g_vBuf0[i].f + g_vBuf1[i].f;
            ir += g_vBuf0[i].i + g_vBuf1[i].i;
            fr += g_vBuf0[i].f + g_vBuf1[i].f;
            ir += g_vBuf0[i].i + g_vBuf1[i].i;
            fr += g_vBuf0[i].f + g_vBuf1[i].f;
            ir += g_vBuf0[i].i + g_vBuf1[i].i;
            fr += g_vBuf0[i].f + g_vBuf1[i].f;
            ir += g_vBuf0[i].i + g_vBuf1[i].i;
            fr += g_vBuf0[i].f + g_vBuf1[i].f;
            ir += g_vBuf0[i].i + g_vBuf1[i].i;
            fr += g_vBuf0[i].f + g_vBuf1[i].f;
            ir += g_vBuf0[i].i + g_vBuf1[i].i;
            fr += g_vBuf0[i].f + g_vBuf1[i].f;
        }
        out[i].i = ir;
        out[i].f = fr;
    }
}

//--------------------------------------------------------------------------------------
// Entry point to the program
//--------------------------------------------------------------------------------------
int main()
{
    LARGE_INTEGER freq;        // ticks per second
    LARGE_INTEGER t1, t2;           // ticks
    double tGPU, tCPU;

    // get ticks per second
    QueryPerformanceFrequency(&freq);

    /* =========================== */
    printf("Creating device...");
    if (FAILED(CreateComputeDevice(&g_pDevice, &g_pContext)))
        return pauseOnDBG(1);
    printf("done\n");

    /* =========================== */
    printf("Creating Compute Shader...");
    if (FAILED(CreateComputeShader(L"ComputeShader.hlsl", "main", g_pDevice, &g_pCS)))
        return pauseOnDBG(1);
    printf("done\n");

    /* =========================== */
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
    printf("done\n");

    /* =========================== */
    printf("Creating buffer views...");
    CreateBufferSRV(g_pDevice, g_pBuf0, &g_pBuf0SRV);
    CreateBufferSRV(g_pDevice, g_pBuf1, &g_pBuf1SRV);
    CreateBufferUAV(g_pDevice, g_pBufResult, &g_pBufResultUAV);
    printf("done\n");

    /* =========================== */
    printf("Running Compute Shader...");
    ID3D11ShaderResourceView* aRViews[2] = { g_pBuf0SRV, g_pBuf1SRV };
    QueryPerformanceCounter(&t1);
    RunComputeShader(g_pContext, g_pCS, 2, aRViews, nullptr, nullptr, 0, g_pBufResultUAV, NUM_ELEMENTS, 1, 1);
    QueryPerformanceCounter(&t2);
    tGPU = (t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart;

    printf("done\n");

    /* =========================== */
    printf("Running on CPU...");
    BufType res[NUM_ELEMENTS];

    RunCPU(res);

    QueryPerformanceCounter(&t1);
    RunCPU(res);
    QueryPerformanceCounter(&t2);
    tCPU = (t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart;

    printf("done\n");

    /* =========================== */
    // Read back the result from GPU, verify its correctness against result computed by CPU
    {
        ID3D11Buffer* debugbuf = CreateAndCopyToDebugBuf(g_pDevice, g_pContext, g_pBufResult);
        D3D11_MAPPED_SUBRESOURCE MappedResource;
        BufType *p;
        g_pContext->Map(debugbuf, 0, D3D11_MAP_READ, 0, &MappedResource);

        // Set a break point here and put down the expression "p, 1024" in your watch window to see what has been written out by our CS
        // This is also a common trick to debug CS programs.
        p = (BufType*)MappedResource.pData;

        // Verify that if Compute Shader has done right
        printf("Verifying against CPU result...");
        bool bSuccess = true;
        for (int i = 0; i < NUM_ELEMENTS; ++i) {
            if ((p[i].i != res[i].i) || (p[i].f != res[i].f))
            {
                printf("failure\n");
                bSuccess = false;
                break;
            }
        }
        if (bSuccess)
            printf("succeeded\n");

        g_pContext->Unmap(debugbuf, 0);

        SAFE_RELEASE(debugbuf);
    }

    printf("\nCompare time GPU: %f, CPU: %f\n\n", tGPU, tCPU);

    /* =========================== */
    printf("Cleaning up...\n");
    SAFE_RELEASE(g_pBuf0SRV);
    SAFE_RELEASE(g_pBuf1SRV);
    SAFE_RELEASE(g_pBufResultUAV);
    SAFE_RELEASE(g_pBuf0);
    SAFE_RELEASE(g_pBuf1);
    SAFE_RELEASE(g_pBufResult);
    SAFE_RELEASE(g_pCS);
    SAFE_RELEASE(g_pContext);
    SAFE_RELEASE(g_pDevice);

    return pauseOnDBG(0);
}


//--------------------------------------------------------------------------------------
// Create the D3D device and device context suitable for running Compute Shaders(CS)
//--------------------------------------------------------------------------------------
_Use_decl_annotations_
HRESULT CreateComputeDevice(ID3D11Device** ppDeviceOut, ID3D11DeviceContext** ppContextOut)
{
    *ppDeviceOut = nullptr;
    *ppContextOut = nullptr;

    HRESULT hr = S_OK;
    UINT uCreationFlags = 0;
#ifdef _DEBUG
    uCreationFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL flOut;
    static const D3D_FEATURE_LEVEL flvl[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };

    return D3D11CreateDevice(nullptr,  // Use default graphics card
        D3D_DRIVER_TYPE_HARDWARE,    // Try to create a hardware accelerated device
        nullptr,                     // Do not use external software rasterizer module
        uCreationFlags,              // Device creation flags
        flvl,
        sizeof(flvl) / sizeof(D3D_FEATURE_LEVEL),
        D3D11_SDK_VERSION,           // SDK version
        ppDeviceOut,                 // Device out
        &flOut,                      // Actual feature level created
        ppContextOut);               // Context out
}

//--------------------------------------------------------------------------------------
// Compile and create the CS
//--------------------------------------------------------------------------------------
_Use_decl_annotations_
HRESULT CreateComputeShader(LPCWSTR pSrcFile, LPCSTR pFunctionName, ID3D11Device* pDevice, ID3D11ComputeShader** ppShaderOut)
{
    if (!pDevice || !ppShaderOut)
        return E_INVALIDARG;

    DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
    // Set the D3DCOMPILE_DEBUG flag to embed debug information in the shaders.
    // Setting this flag improves the shader debugging experience, but still allows 
    // the shaders to be optimized and to run exactly the way they will run in 
    // the release configuration of this program.
    dwShaderFlags |= D3DCOMPILE_DEBUG;

    // Disable optimizations to further improve shader debugging
    dwShaderFlags |= D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

    const D3D_SHADER_MACRO defines[] = { nullptr, nullptr };

    // We generally prefer to use the higher CS shader profile when possible as CS 5.0 is better performance on 11-class hardware
    LPCSTR pProfile = "cs_5_0";

    ID3DBlob* pErrorBlob = nullptr;
    ID3DBlob* pBlob = nullptr;

    HRESULT hr = D3DCompileFromFile(pSrcFile, defines, D3D_COMPILE_STANDARD_FILE_INCLUDE, pFunctionName, pProfile, dwShaderFlags, 0, &pBlob, &pErrorBlob);

    if (FAILED(hr))
    {
        if (pErrorBlob)
            OutputDebugStringA((char*)pErrorBlob->GetBufferPointer());

        SAFE_RELEASE(pErrorBlob);
        SAFE_RELEASE(pBlob);

        return hr;
    }

    hr = pDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), nullptr, ppShaderOut);

    SAFE_RELEASE(pErrorBlob);
    SAFE_RELEASE(pBlob);

    return hr;
}

//--------------------------------------------------------------------------------------
// Create Raw Buffer
//--------------------------------------------------------------------------------------
_Use_decl_annotations_
HRESULT CreateRawBuffer(ID3D11Device* pDevice, UINT uSize, void* pInitData, ID3D11Buffer** ppBufOut)
{
    *ppBufOut = nullptr;

    D3D11_BUFFER_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_INDEX_BUFFER | D3D11_BIND_VERTEX_BUFFER;
    desc.ByteWidth = uSize;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;

    if (pInitData)
    {
        D3D11_SUBRESOURCE_DATA InitData;
        InitData.pSysMem = pInitData;
        return pDevice->CreateBuffer(&desc, &InitData, ppBufOut);
    }
    else
        return pDevice->CreateBuffer(&desc, nullptr, ppBufOut);
}

//--------------------------------------------------------------------------------------
// Create Shader Resource View for Structured or Raw Buffers
//--------------------------------------------------------------------------------------
_Use_decl_annotations_
HRESULT CreateBufferSRV(ID3D11Device* pDevice, ID3D11Buffer* pBuffer, ID3D11ShaderResourceView** ppSRVOut)
{
    D3D11_BUFFER_DESC descBuf;
    ZeroMemory(&descBuf, sizeof(descBuf));
    pBuffer->GetDesc(&descBuf);

    D3D11_SHADER_RESOURCE_VIEW_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    desc.BufferEx.FirstElement = 0;

    if (descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS)
    {
        // This is a Raw Buffer

        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        desc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
        desc.BufferEx.NumElements = descBuf.ByteWidth / 4;
    }
    else
        if (descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED)
        {
            // This is a Structured Buffer

            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.BufferEx.NumElements = descBuf.ByteWidth / descBuf.StructureByteStride;
        }
        else
        {
            return E_INVALIDARG;
        }

    return pDevice->CreateShaderResourceView(pBuffer, &desc, ppSRVOut);
}

//--------------------------------------------------------------------------------------
// Create Unordered Access View for Structured or Raw Buffers
//-------------------------------------------------------------------------------------- 
_Use_decl_annotations_
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

//--------------------------------------------------------------------------------------
// Create a CPU accessible buffer and download the content of a GPU buffer into it
// This function is very useful for debugging CS programs
//-------------------------------------------------------------------------------------- 
_Use_decl_annotations_
ID3D11Buffer* CreateAndCopyToDebugBuf(ID3D11Device* pDevice, ID3D11DeviceContext* pd3dImmediateContext, ID3D11Buffer* pBuffer)
{
    ID3D11Buffer* debugbuf = nullptr;

    D3D11_BUFFER_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    pBuffer->GetDesc(&desc);
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.MiscFlags = 0;
    if (SUCCEEDED(pDevice->CreateBuffer(&desc, nullptr, &debugbuf)))
        pd3dImmediateContext->CopyResource(debugbuf, pBuffer);

    return debugbuf;
}

//--------------------------------------------------------------------------------------
// Run CS
//-------------------------------------------------------------------------------------- 
_Use_decl_annotations_
void RunComputeShader(ID3D11DeviceContext* pd3dImmediateContext,
ID3D11ComputeShader* pComputeShader,
UINT nNumViews, ID3D11ShaderResourceView** pShaderResourceViews,
ID3D11Buffer* pCBCS, void* pCSData, DWORD dwNumDataBytes,
ID3D11UnorderedAccessView* pUnorderedAccessView,
UINT X, UINT Y, UINT Z)
{
    pd3dImmediateContext->CSSetShader(pComputeShader, nullptr, 0);
    pd3dImmediateContext->CSSetShaderResources(0, nNumViews, pShaderResourceViews);
    pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, &pUnorderedAccessView, nullptr);
    if (pCBCS && pCSData)
    {
        D3D11_MAPPED_SUBRESOURCE MappedResource;
        pd3dImmediateContext->Map(pCBCS, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource);
        memcpy(MappedResource.pData, pCSData, dwNumDataBytes);
        pd3dImmediateContext->Unmap(pCBCS, 0);
        ID3D11Buffer* ppCB[1] = { pCBCS };
        pd3dImmediateContext->CSSetConstantBuffers(0, 1, ppCB);
    }

    pd3dImmediateContext->Dispatch(X, Y, Z);

    pd3dImmediateContext->CSSetShader(nullptr, nullptr, 0);

    ID3D11UnorderedAccessView* ppUAViewnullptr[1] = { nullptr };
    pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewnullptr, nullptr);

    ID3D11ShaderResourceView* ppSRVnullptr[2] = { nullptr, nullptr };
    pd3dImmediateContext->CSSetShaderResources(0, 2, ppSRVnullptr);

    ID3D11Buffer* ppCBnullptr[1] = { nullptr };
    pd3dImmediateContext->CSSetConstantBuffers(0, 1, ppCBnullptr);
}