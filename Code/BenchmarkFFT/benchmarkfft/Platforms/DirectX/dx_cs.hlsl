#define GROUP_SIZE_X 1024
#define GRID_DIM_X 8192
struct cpx
{
    float x;
    float y;
};

cbuffer Constants : register(b0)
{
    float           angle;
    float           local_angle;
    float           scalar;
    int             steps_left;
    int             leading_bits;
    int             steps_gpu;
    int             number_of_blocks;
    int             block_range_half;
    int             steps;
    unsigned int    lmask;
    int             dist;
};

StructuredBuffer<cpx> input : register(t0);
RWStructuredBuffer<cpx> rw_buf : register(u0);

groupshared cpx shared_buf[GROUP_SIZE_X << 1];

void add_sub_mul(in cpx low, in cpx high, out cpx o_low, out cpx o_high, in cpx w)
{
    float x = low.x - high.x;
    float y = low.y - high.y;
    o_low.x = low.x + high.x;
    o_low.y = low.y + high.y;
    o_high.x = (w.x * x) - (w.y * y);
    o_high.y = (w.y * x) + (w.x * y);
}

[numthreads(GROUP_SIZE_X, 1, 1)]
void dx_global(uint3 threadIDInGroup : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uint groupIndex : SV_GroupIndex,
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = groupID.x * GROUP_SIZE_X + threadIDInGroup.x;
    int in_low = tid + (tid & lmask);
    int in_high = in_low + dist;
    cpx w;
    sincos(angle * ((tid << steps) & ((dist - 1) << steps)), w.y, w.x);
    add_sub_mul(input[in_low], input[in_high], rw_buf[in_low], rw_buf[in_high], w);
}

[numthreads(GROUP_SIZE_X, 1, 1)]
void dx_2d_global(uint3 threadIDInGroup : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uint groupIndex : SV_GroupIndex,
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    cpx w;
    int col_id = groupID.y * GROUP_SIZE_X + threadIDInGroup.x;
    int in_low = (col_id + (col_id & lmask)) + groupID.x * GRID_DIM_X;
    int in_high = in_low + dist;
    sincos(angle * ((col_id << steps) & ((dist - 1) << steps)), w.y, w.x);
    add_sub_mul(input[in_low], input[in_high], rw_buf[in_low], rw_buf[in_high], w);
}

void dx_algorithm_local(in int in_low, in int in_high)
{
    float x, y;
    cpx w, in_lower, in_upper;
    int out_i = (in_low << 1);
    int out_ii = out_i + 1;
    for (int steps = 0; steps < steps_left; ++steps)
    {
        sincos(local_angle * (in_low & (0xFFFFFFFF << steps)), w.y, w.x);
        in_lower = shared_buf[in_low];
        in_upper = shared_buf[in_high];
        GroupMemoryBarrierWithGroupSync();
        add_sub_mul(in_lower, in_upper, shared_buf[out_i], shared_buf[out_ii], w);
        GroupMemoryBarrierWithGroupSync();
    }
}

[numthreads(GROUP_SIZE_X, 1, 1)]
void dx_local(uint3 threadIDInGroup : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uint groupIndex : SV_GroupIndex,
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int in_low = threadIDInGroup.x;
    int in_high = in_low + block_range_half;
    int offset = (groupID.x * GROUP_SIZE_X) << 1;

    shared_buf[in_low] = input[in_low + offset];
    shared_buf[in_high] = input[in_high + offset];

    dx_algorithm_local(in_low, in_high);

    cpx a = { shared_buf[in_low].x * scalar, shared_buf[in_low].y * scalar };
    cpx b = { shared_buf[in_high].x * scalar, shared_buf[in_high].y * scalar };
    rw_buf[(reversebits((uint)(in_low + offset)) >> leading_bits)] = a;
    rw_buf[(reversebits((uint)(in_high + offset)) >> leading_bits)] = b;
}

[numthreads(GROUP_SIZE_X, 1, 1)]
void dx_2d_local_row(uint3 threadIDInGroup : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uint groupIndex : SV_GroupIndex,
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int in_low = threadIDInGroup.x;
    int in_high = block_range_half + in_low;
    int row_start = GRID_DIM_X * groupID.x;
    int row_offset = (groupID.y * GROUP_SIZE_X) << 1;

    shared_buf[in_low] = input[in_low + row_start + row_offset];
    shared_buf[in_high] = input[in_high + row_start + row_offset];

    dx_algorithm_local(in_low, in_high);

    cpx a = { shared_buf[in_low].x * scalar, shared_buf[in_low].y *scalar };
    cpx b = { shared_buf[in_high].x * scalar, shared_buf[in_high].y *scalar };
    rw_buf[(reversebits(in_low + row_offset) >> leading_bits) + row_start] = a;
    rw_buf[(reversebits(in_high + row_offset) >> leading_bits) + row_start] = b;
}