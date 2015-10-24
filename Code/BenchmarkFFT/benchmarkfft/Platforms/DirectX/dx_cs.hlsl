#define GROUP_SIZE_X 4
#define NUMBER_OF_BLOCKS 1

//#define BARRIER AllMemoryBarrierWithGroupSync()
//#define BARRIER DeviceMemoryBarrierWithGroupSync()
#define BARRIER GroupMemoryBarrierWithGroupSync()

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
    bool            load_input;
    int             steps;
    unsigned int    lmask;
    int             dist;
};

StructuredBuffer<cpx> input : register(t0);

RWStructuredBuffer<cpx> rwbuf_in : register(u0);
RWStructuredBuffer<cpx> rwbuf_out : register(u1);
/*
RWStructuredBuffer<int> sync_in : register(u2);
RWStructuredBuffer<int> sync_out : register(u3);
*/
groupshared cpx shared_buf[GROUP_SIZE_X << 1];
/*
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
        BARRIER;
        if (tid < number_of_blocks)
        {
            sync_out[tid] = goal;
        }
    }
    if (tid == 0)
    {
        while (sync_out[gid] != goal) {}
    }
    BARRIER;
}

void add_cub_mul(in cpx low, in cpx high, out cpx o_low, out cpx o_high, in cpx w)
{
    float x = low.x - high.x;
    float y = low.y - high.y;
    o_low.x = low.x + high.x;
    o_low.y = low.y + high.y;
    o_high.x = (w.x * x) - (w.y * y);
    o_high.y = (w.y * x) + (w.x * y);
}

void add_cub_mul_global(inout cpx low, inout cpx high, in cpx w)
{
    float x = low.x - high.x;
    float y = low.y - high.y;
    low.x = low.x + high.x;
    low.y = low.y + high.y;
    high.x = (w.x * x) - (w.y * y);
    high.y = (w.y * x) + (w.x * y);
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
    //add_cub_mul_global(rwbuf_in[in_low], rwbuf_in[in_high], w);
    
    cpx in_lower = rwbuf_in[in_low];
    cpx in_upper = rwbuf_in[in_high];
    float x = in_lower.x - in_upper.x;
    float y = in_lower.y - in_upper.y;
    rwbuf_in[in_low].x = in_lower.x + in_upper.x;
    rwbuf_in[in_low].y = in_lower.y + in_upper.y;
    rwbuf_in[in_high].x = (w.x * x) - (w.y * y);
    rwbuf_in[in_high].y = (w.y * x) + (w.x * y);
    
    dx_group_sync(thread_id, group_id, number_of_blocks + steps);
    for (int bit = bit_start - 1; bit > steps_gpu; --bit)
    {
        dist >>= 1;
        ++steps;
        in_low = tid + (tid & (0xFFFFFFFF << bit));
        in_high = in_low + dist;
        sincos(angle * ((tid << steps) & ((dist - 1) << steps)), w.y, w.x);
        //add_cub_mul_global(rwbuf_in[in_low], rwbuf_in[in_high], w);
        
        in_lower = rwbuf_in[in_low];
        in_upper = rwbuf_in[in_high];
        x = in_lower.x - in_upper.x;
        y = in_lower.y - in_upper.y;
        rwbuf_in[in_low].x = in_lower.x + in_upper.x;
        rwbuf_in[in_low].y = in_lower.y + in_upper.y;
        rwbuf_in[in_high].x = (w.x * x) - (w.y * y);
        rwbuf_in[in_high].y = (w.y * x) + (w.x * y);
        
        dx_group_sync(thread_id, group_id, number_of_blocks + steps);
    }
    bit_out = steps_gpu + 1;
}

[numthreads(GROUP_SIZE_X, 1, 1)]
void dx_global(uint3 threadIDInGroup : SV_GroupThreadID, uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 dispatchThreadID : SV_DispatchThreadID)
{
int tid = groupID.x * GROUP_SIZE_X + threadIDInGroup.x;
int in_low = tid + (tid & lmask);
int in_high = in_low + dist;
cpx w;
sincos(angle * ((tid << steps) & ((dist - 1) << steps)), w.y, w.x);
cpx in_lower, in_upper;
if (load_input)
{
in_lower = input[in_low];
in_upper = input[in_high];
}
else
{
in_lower = rwbuf_in[in_low];
in_upper = rwbuf_in[in_high];
}
float x = in_lower.x - in_upper.x;
float y = in_lower.y - in_upper.y;
rwbuf_in[in_low].x = in_lower.x + in_upper.x;
rwbuf_in[in_low].y = in_lower.y + in_upper.y;
rwbuf_in[in_high].x = (w.x * x) - (w.y * y);
rwbuf_in[in_high].y = (w.y * x) + (w.x * y);
}
*/
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
        BARRIER;
        x = in_lower.x - in_upper.x;
        y = in_lower.y - in_upper.y;
        shared_buf[out_i].x = in_lower.x + in_upper.x;
        shared_buf[out_i].y = in_lower.y + in_upper.y;
        shared_buf[out_ii].x = (w.x * x) - (w.y * y);
        shared_buf[out_ii].y = (w.y * x) + (w.x * y);
        BARRIER;
    }
}

[numthreads(GROUP_SIZE_X, 1, 1)]
void dx_local(uint3 threadIDInGroup : SV_GroupThreadID, uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int bit = steps_left;
    int in_low = threadIDInGroup.x;
    int in_high = block_range_half;
    uint tid = groupID.x * GROUP_SIZE_X + threadIDInGroup.x;
    int offset = (groupID.x * GROUP_SIZE_X) << 1;
    
    if (load_input) {
        rwbuf_in[tid] = input[tid];
        rwbuf_in[block_range_half + tid] = input[block_range_half + tid];
        BARRIER;
    }
    /*
    if (NUMBER_OF_BLOCKS > 1)
    {
        dx_algorithm_global_sync(tid, threadIDInGroup.x, groupID.x, steps_left - 1, steps_gpu, angle, in_high, bit);
        in_high = (in_high >> (firstbitlow(number_of_blocks) - 1));
    }
    */
    in_high += in_low;
    shared_buf[in_low] = rwbuf_in[in_low + offset];
    shared_buf[in_high] = rwbuf_in[in_high + offset];

    dx_algorithm_local(in_low, in_high, bit);
    cpx a = { shared_buf[in_low].x * scalar, shared_buf[in_low].y * scalar };
    cpx b = { shared_buf[in_high].x * scalar, shared_buf[in_high].y * scalar };
    rwbuf_out[(reversebits((uint)(in_low + offset)) >> leading_bits)] = a;
    rwbuf_out[(reversebits((uint)(in_high + offset)) >> leading_bits)] = b;
}