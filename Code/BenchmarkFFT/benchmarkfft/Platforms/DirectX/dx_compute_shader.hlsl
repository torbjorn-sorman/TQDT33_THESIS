#define GROUP_SIZE_X 1024

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

int sync_array_in[GROUP_SIZE_X];
int sync_array_out[GROUP_SIZE_X];

StructuredBuffer<cpx> input;
RWStructuredBuffer<cpx> dev_in;
RWStructuredBuffer<cpx> dev_out;
groupshared cpx shared_buf[GROUP_SIZE_X << 1];

/*
void dx_inner_kernel(in float angle, in int steps, in unsigned int lmask, in int dist)
{
    cpx w;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int in_low = tid + (tid & lmask);
    in += in_low;
    out += in_low;
    SIN_COS_F(angle * ((tid << steps) & ((dist - 1) << steps)), &w.y, &w.x);
    cpx_add_sub_mul(in, in + dist, out, out + dist, &w);
}

int dx_algorithm_global_sync(in int bit_start, in int steps_gpu, uniform float angle, uniform int number_of_blocks, uniform int n_half)
{
    int dist = n_half;
    int steps = 0;
    cuda_block_sync_init(sync_array_in, sync_array_out, (groupID.x * blockDim.x + threadIDInGroup.x), number_of_blocks);

    dx_inner_kernel(dev_in, dev_out, angle, steps, 0xFFFFFFFF << bit_start, dist);

    cuda_block_sync(sync_array_in, sync_array_out, number_of_blocks + steps);
    for (int bit = bit_start - 1; bit > steps_gpu; --bit) {
        dist >>= 1;
        ++steps;
        cuda_inner_kernel(dev_out, dev_out, angle, steps, 0xFFFFFFFF << bit, dist);
        cuda_block_sync(sync_array_in, sync_array_out, number_of_blocks + steps);
    }
    return steps_gpu + 1;
}
*/
void dx_algorithm_local(in int in_low, in int in_high, uniform float angle, uniform int bit)
{
    float x, y;
    cpx w, in_lower, in_upper;
    int out_i = (in_low << 1);
    int out_ii = out_i + 1;
    for (int steps = 0; steps < bit; ++steps) {
        sincos(angle * (in_low & (0xFFFFFFFF << steps)), w.y, w.x);
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

    /*
    if (number_of_blocks > 1) {
    bit = dx_algorithm_global_sync(steps_left - 1, steps_gpu, angle, number_of_blocks, in_high);
    in_high >>= log2(number_of_blocks);
    dev_in = dev_out;
    }
    */

    int offset = (groupID.x * GROUP_SIZE_X) << 1;

    uint n = groupID.x * GROUP_SIZE_X + threadIDInGroup.x;
    dev_in[n] = input[n];
    dev_in[GROUP_SIZE_X + n] = input[GROUP_SIZE_X + n];
    AllMemoryBarrierWithGroupSync();

    in_high += in_low;

    shared_buf[in_low] = dev_in[in_low + offset];
    shared_buf[in_high] = dev_in[in_high + offset];
    AllMemoryBarrierWithGroupSync();

    dx_algorithm_local(in_low, in_high, local_angle, bit);

    cpx a = { shared_buf[in_low].x * scalar, shared_buf[in_low].y * scalar };
    cpx b = { shared_buf[in_high].x * scalar, shared_buf[in_high].y * scalar };

    dev_out[(reversebits((uint)(in_low + offset)) >> leading_bits)] = a;
    dev_out[(reversebits((uint)(in_high + offset)) >> leading_bits)] = b;
}