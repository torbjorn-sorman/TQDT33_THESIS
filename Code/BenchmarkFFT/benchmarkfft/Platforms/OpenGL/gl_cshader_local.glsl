#version 430
#define LOCAL_DIM_X 1024
#define SHARED_MEM_SIZE 2048

struct cpx {
    float x;
    float y;
};

layout(local_size_x = LOCAL_DIM_X) in;

layout(std430, binding = 0) buffer ssbo0 { cpx input_data[]; };
layout(std430, binding = 1) buffer ssbo1 { cpx output_data[]; };

shared cpx shared_buf[SHARED_MEM_SIZE];

uniform float local_angle;
uniform float scalar;
uniform uint leading_bits;
uniform uint steps_left;
uniform uint block_range;

void add_sub_mul(cpx low, cpx high, out cpx o_low, out cpx o_high, cpx w);
void dx_algorithm_local(uint in_low, uint in_high);

void main()
{
    uint in_low = (gl_LocalInvocationID.x);
    uint in_high = (in_low + block_range);
    uint offset = ((gl_WorkGroupID.x * LOCAL_DIM_X) * 2);
    shared_buf[in_low] = input_data[(in_low + offset)];
    shared_buf[in_high] = input_data[(in_high + offset)];

    dx_algorithm_local(in_low, in_high);

    output_data[(bitfieldReverse((in_low + offset)) >> leading_bits)] = cpx((shared_buf[in_low].x * scalar), (shared_buf[in_low].y * scalar));
    output_data[(bitfieldReverse((in_high + offset)) >> leading_bits)] = cpx((shared_buf[in_high].x * scalar), (shared_buf[in_high].y * scalar));
}

void dx_algorithm_local(uint in_low, uint in_high)
{
    float angle, x, y;
    cpx w, in_lower, in_upper;
    uint out_i = (in_low * 2);
    uint out_ii = (out_i + 1);
    for (uint steps = 0; steps < steps_left; ++steps)
    {
        angle = (local_angle * (int(in_low & (0xFFFFFFFF << steps))));
        w.x = cos(angle);
        w.y = sin(angle);
        in_lower = shared_buf[in_low];
        in_upper = shared_buf[in_high];
        barrier();
        add_sub_mul(in_lower, in_upper, shared_buf[out_i], shared_buf[out_ii], w);
        barrier();
    }
}

void add_sub_mul(cpx low, cpx high, out cpx o_low, out cpx o_high, cpx w)
{
    float x = low.x - high.x;
    float y = low.y - high.y;
    o_low.x = low.x + high.x;
    o_low.y = low.y + high.y;
    o_high.x = (w.x * x) - (w.y * y);
    o_high.y = (w.y * x) + (w.x * y);
}