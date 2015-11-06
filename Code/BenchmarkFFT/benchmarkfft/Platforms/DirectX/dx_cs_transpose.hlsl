// Auto update, do not alter
#define WIDTH 8192
#define DX_TILE_DIM 32

// Tweak here!
#define DX_BLOCK_DIM 16

struct cpx
{
    float x;
    float y;
};

StructuredBuffer<cpx> input : register(t0);
RWStructuredBuffer<cpx> rw_buf : register(u0);

// Likely to result in banking issues... DX does not allow for more then 32K (48K is available)
// Alternative is to use less shared memory. (32 x 32 instead of 64 x 64)
groupshared cpx tile[DX_TILE_DIM][DX_TILE_DIM + 1];

[numthreads(DX_BLOCK_DIM, DX_BLOCK_DIM, 1)]
void dx_transpose(uint3 threadIDInGroup : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uint groupIndex : SV_GroupIndex,
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int i, j;
    int x = groupID.x * DX_TILE_DIM + threadIDInGroup.x;
    int y = groupID.y * DX_TILE_DIM + threadIDInGroup.y;    
    for (j = 0; j < DX_TILE_DIM; j += DX_BLOCK_DIM)
        for (i = 0; i < DX_TILE_DIM; i += DX_BLOCK_DIM)
            tile[threadIDInGroup.y + j][threadIDInGroup.x + i] = input[(y + j) * WIDTH + (x + i)];

    GroupMemoryBarrierWithGroupSync();

    x = groupID.y * DX_TILE_DIM + threadIDInGroup.x;
    y = groupID.x * DX_TILE_DIM + threadIDInGroup.y;
    for (j = 0; j < DX_TILE_DIM; j += DX_BLOCK_DIM)
        for (i = 0; i < DX_TILE_DIM; i += DX_BLOCK_DIM)
            rw_buf[(y + j) * WIDTH + (x + i)] = tile[threadIDInGroup.x + i][threadIDInGroup.y + j];
}