#version 450
#define GROUP_SIZE_X 1024
#define GROUP_SIZE_X_2 2048

struct cpx {
    float x;
    float y;
};

layout(std430, binding = 5) buffer bbs{ int bs[]; };
layout(local_size_x = GROUP_SIZE_X) in;

shared cpx shared_buf[GROUP_SIZE_X_2];

void add_sub_mul(cpx low, cpx high, out cpx o_low, out cpx o_high, cpx w)
{
    float x = low.x - high.x;
    float y = low.y - high.y;
    o_low.x = low.x + high.x;
    o_low.y = low.y + high.y;
    o_high.x = (w.x * x) - (w.y * y);
    o_high.y = (w.y * x) + (w.x * y);
}

void dx_algorithm_local(uint in_low, uint in_high)
{
    float x, y;
    cpx w, in_lower, in_upper;
    uint out_i = (in_low << 1);
    uint out_ii = out_i + 1;
    for (int steps = 0; steps < steps_left; ++steps)
    {
        sincos(local_angle * (in_low & (0xFFFFFFFF << steps)), w.y, w.x);
        in_lower = shared_buf[in_low];
        in_upper = shared_buf[in_high];
        barrier();
        add_sub_mul(in_lower, in_upper, shared_buf[out_i], shared_buf[out_ii], w);
        barrier();
    }
}

void main()
{
    uint in_low = gl_LocalInvocationID.x;
    uint in_high = in_low + block_range_half;
    uint offset = (gl_WorkGroupID.x * GROUP_SIZE_X) << 1;

    shared_buf[in_low] = input[in_low + offset];
    shared_buf[in_high] = input[in_high + offset];

    dx_algorithm_local(in_low, in_high);

    cpx a = { shared_buf[in_low].x * scalar, shared_buf[in_low].y * scalar };
    cpx b = { shared_buf[in_high].x * scalar, shared_buf[in_high].y * scalar };
    rw_buf[(bitfieldReverse(in_low + offset) >> leading_bits)] = a;
    rw_buf[(bitfieldReverse(in_high + offset) >> leading_bits)] = b;
}