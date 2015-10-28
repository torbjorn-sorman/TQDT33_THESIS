#define WIDTH 16384
#define TILE_DIM 64
#define THREAD_TILE_DIM 32  

struct cpx
{
    float x;
    float y;
};

StructuredBuffer<cpx> input : register(t0);
RWStructuredBuffer<cpx> rw_buf : register(u0);

// Likely to result in banking issues... DX does not allow for more then 32K (48K is available)
// Alternative is to use less shared memory. (32 x 32 instead of 64 x 64)
groupshared cpx tile[TILE_DIM][TILE_DIM];

[numthreads(THREAD_TILE_DIM, THREAD_TILE_DIM, 1)]
void dx_transpose(uint3 threadIDInGroup : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uint groupIndex : SV_GroupIndex,
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int x = groupID.x * TILE_DIM + threadIDInGroup.x;
    int y = groupID.y * TILE_DIM + threadIDInGroup.y;
    [unroll(TILE_DIM / THREAD_TILE_DIM)] for (int j = 0; j < TILE_DIM; j += THREAD_TILE_DIM)
    {
        [unroll(TILE_DIM / THREAD_TILE_DIM)] for (int i = 0; i < TILE_DIM; i += THREAD_TILE_DIM)
        {
            tile[threadIDInGroup.y + j][threadIDInGroup.x + i] = input[(y + j) * WIDTH + (x + i)];
        }
    }

    GroupMemoryBarrierWithGroupSync();

    x = groupID.y * TILE_DIM + threadIDInGroup.x;
    y = groupID.x * TILE_DIM + threadIDInGroup.y;
    [unroll(TILE_DIM / THREAD_TILE_DIM)] for (int j = 0; j < TILE_DIM; j += THREAD_TILE_DIM)
    {
        [unroll(TILE_DIM / THREAD_TILE_DIM)] for (int i = 0; i < TILE_DIM; i += THREAD_TILE_DIM)
        {
            rw_buf[(y + j) * WIDTH + (x + i)] = tile[threadIDInGroup.x + i][threadIDInGroup.y + j];
        }
    }
}