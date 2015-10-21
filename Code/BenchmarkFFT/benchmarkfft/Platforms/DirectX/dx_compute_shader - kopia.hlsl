#define GROUP_SIZE_X  8

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
    int     n_half;
};

StructuredBuffer<cpx> input;
RWStructuredBuffer<cpx> dev_in;
RWStructuredBuffer<cpx> dev_out;
groupshared cpx shared_buf[GROUP_SIZE_X * 2];

/*
int cuda_algorithm_global_sync(cpx dev_in[], cpx dev_out[], int bit_start, int steps_gpu, float angle, int number_of_blocks, int n_half)
{
int dist = n_half;
int steps = 0;
cuda_block_sync_init(sync_array_in, sync_array_out, (groupID.x * blockDim.x + threadIDInGroup.x), number_of_blocks);
cuda_inner_kernel(dev_in, dev_out, angle, steps, 0xFFFFFFFF << bit_start, dist);
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
    int in_high = n_half;
    /*
    if (number_of_blocks > 1) {
    bit = cuda_algorithm_global_sync(dev_in, dev_out, steps_left - 1, steps_gpu, angle, number_of_blocks, in_high);
    in_high >>= log2(number_of_blocks);
    dev_in = dev_out;
    }
    */
    int offset = groupID.x * GROUP_SIZE_X * 2;
    uint n = groupID.x * GROUP_SIZE_X + threadIDInGroup.x;

    dev_in[n] = input[n];
    dev_in[n << 1] = input[n << 1];
    AllMemoryBarrierWithGroupSync();

    in_high += in_low;

    shared_buf[in_low] = dev_in[in_low + offset];
    shared_buf[in_high] = dev_in[in_high + offset];
    AllMemoryBarrierWithGroupSync();

    dx_algorithm_local(in_low, in_high, local_angle, bit);

    cpx a = { shared_buf[in_low].x * scalar, shared_buf[in_low].y * scalar };
    cpx b = { shared_buf[in_high].x * scalar, shared_buf[in_high].y * scalar };

    int out_low = (reversebits((uint)(in_low + offset)) >> leading_bits);
    int out_high = (reversebits((uint)(in_high + offset)) >> leading_bits);

    dev_out[out_low] = a;
    dev_out[out_high] = b;

    dev_out[out_low].x = threadIDInGroup.x;
    dev_out[out_high].x = threadIDInGroup.x;
}
