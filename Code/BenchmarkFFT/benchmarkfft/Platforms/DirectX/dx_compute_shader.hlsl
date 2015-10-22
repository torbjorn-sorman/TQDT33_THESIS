#define GROUP_SIZE_X 1024
#define NUMBER_OF_BLOCKS 2

struct cpx
{
    float x;
    float y;
};

cbuffer Constants
{
    float   angle;
    float   local_angle;
    float   scalar;
    int     steps_left;
    int     leading_bits;
    int     steps_gpu;
    int     number_of_blocks;
    int     block_range_half;
};

static int sync_in[GROUP_SIZE_X];
static int sync_out[GROUP_SIZE_X];

StructuredBuffer<cpx> input;
RWStructuredBuffer<cpx> dev_in;
RWStructuredBuffer<cpx> dev_out;
groupshared cpx shared_buf[GROUP_SIZE_X << 1];

int tab32[32] =
{
    0, 9, 1, 10, 13, 21, 2, 29,
    11, 14, 16, 18, 22, 25, 3, 30,
    8, 12, 20, 28, 15, 17, 24, 7,
    19, 27, 23, 6, 26, 5, 4, 31
};

int log2(in int value)
{
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return tab32[(unsigned int)(value * 0x07C4ACDD) >> 27];
}

void dx_group_sync(in int tid, in int gid, uniform int goal)
{
    if (tid == 0)
    {
        sync_in[gid] = goal;
    }
    if (gid == 1)
    {
        if (tid < number_of_blocks)
        {
            while (sync_in[tid] != goal){}
        }
        AllMemoryBarrierWithGroupSync();
        if (tid < number_of_blocks)
        {
            sync_out[tid] = goal;
        }
    }
    if (tid == 0)
    {
        while (sync_out[gid] != goal) {}
    }
    AllMemoryBarrierWithGroupSync();
}

void dx_algorithm_global_sync(in int tid, in int thread_id, in int group_id, in int bit_start, in int steps_gpu, uniform float angle, uniform int n_half, out int bit_out)
{
    int dist = n_half;
    int steps = 0;
    if (tid < number_of_blocks)
    {
        sync_in[tid] = 0;
        sync_out[tid] = 0;
    }
    int in_low = tid + (tid & (0xFFFFFFFF << bit_start));
    int in_high = in_low + dist;
    cpx w;
    sincos(angle * ((tid << steps) & ((dist - 1) << steps)), w.y, w.x);
    cpx in_lower = dev_in[in_low];
    cpx in_upper = dev_in[in_high];
    float x = in_lower.x - in_upper.x;
    float y = in_lower.y - in_upper.y;
    dev_out[in_low].x = in_lower.x + in_upper.x;
    dev_out[in_low].y = in_lower.y + in_upper.y;
    dev_out[in_high].x = (w.x * x) - (w.y * y);
    dev_out[in_high].y = (w.y * x) + (w.x * y);
    dx_group_sync(thread_id, group_id, number_of_blocks + steps);
    for (int bit = bit_start - 1; bit > steps_gpu; --bit)
    {
        dist >>= 1;
        ++steps;
        in_low = tid + (tid & (0xFFFFFFFF << bit));
        in_high = in_low + dist;
        sincos(angle * ((tid << steps) & ((dist - 1) << steps)), w.y, w.x);
        in_lower = dev_out[in_low];
        in_upper = dev_out[in_high];
        x = in_lower.x - in_upper.x;
        y = in_lower.y - in_upper.y;
        dev_out[in_low].x = in_lower.x + in_upper.x;
        dev_out[in_low].y = in_lower.y + in_upper.y;
        dev_out[in_high].x = (w.x * x) - (w.y * y);
        dev_out[in_high].y = (w.y * x) + (w.x * y);
        dx_group_sync(thread_id, group_id, number_of_blocks + steps);
    }
    bit_out = steps_gpu + 1;
}

void dx_algorithm_local(in int in_low, in int in_high, uniform int bit)
{
    float x, y;
    cpx w, in_lower, in_upper;
    int out_i = (in_low << 1);
    int out_ii = out_i + 1;
    for (int steps = 0; steps < bit; ++steps)
    {
        sincos(local_angle * (in_low & (0xFFFFFFFF << steps)), w.y, w.x);
        in_lower = shared_buf[in_low];
        in_upper = shared_buf[in_high];
        AllMemoryBarrierWithGroupSync();
        x = in_lower.x - in_upper.x;
        y = in_lower.y - in_upper.y;
        shared_buf[out_i].x = in_lower.x + in_upper.x;
        shared_buf[out_i].y = in_lower.y + in_upper.y;
        shared_buf[out_ii].x = (w.x * x) - (w.y * y);
        shared_buf[out_ii].y = (w.y * x) + (w.x * y);
        AllMemoryBarrierWithGroupSync();
    }
}

[numthreads(GROUP_SIZE_X, 1, 1)]
void dx_fft(uint3 threadIDInGroup : SV_GroupThreadID, uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int bit = steps_left;
    int in_low = threadIDInGroup.x;
    int in_high = block_range_half;
    uint tid = groupID.x * GROUP_SIZE_X + threadIDInGroup.x;
    int offset = (groupID.x * GROUP_SIZE_X) << 1;
    dev_in[tid] = input[tid];
    dev_in[GROUP_SIZE_X + tid] = input[GROUP_SIZE_X + tid];
    AllMemoryBarrierWithGroupSync();

    if (NUMBER_OF_BLOCKS > 1)
    {
        dx_algorithm_global_sync(tid, threadIDInGroup.x, groupID.x, steps_left - 1, steps_gpu, angle, in_high, bit);
        in_high = (in_high >> log2(number_of_blocks)) + in_low;
        shared_buf[in_low] = dev_out[in_low + offset];
        shared_buf[in_high] = dev_out[in_high + offset];
    }
    else
    {
        in_high += in_low;
        shared_buf[in_low] = dev_in[in_low + offset];
        shared_buf[in_high] = dev_in[in_high + offset];
    }

    AllMemoryBarrierWithGroupSync();
    dx_algorithm_local(in_low, in_high, bit);
    cpx a = { shared_buf[in_low].x * scalar, shared_buf[in_low].y * scalar };
    cpx b = { shared_buf[in_high].x * scalar, shared_buf[in_high].y * scalar };
    dev_out[(reversebits((uint)(in_low + offset)) >> leading_bits)] = a;
    dev_out[(reversebits((uint)(in_high + offset)) >> leading_bits)] = b;
}