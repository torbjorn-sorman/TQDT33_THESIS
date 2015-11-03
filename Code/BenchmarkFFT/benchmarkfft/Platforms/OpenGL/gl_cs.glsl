#version 430
#define LOCAL_DIM_X 1024
#define SHARED_MEM_SIZE 2048

struct cpx {
    float x;
    float y;
};

layout(local_size_x = LOCAL_DIM_X) in;

layout(std430, binding = 0) buffer ssbo0{ cpx input_data[]; };
layout(std430, binding = 1) buffer ssbo1{ cpx output_data[]; };

shared cpx shared_buf[SHARED_MEM_SIZE];

uniform float local_angle;
uniform float scalar;
uniform uint leading_bits;
uniform uint steps_left;
uniform uint block_range_half;

void add_sub_mul(cpx low, cpx high, out cpx o_low, out cpx o_high, cpx w);
void dx_algorithm_local(uint in_low, uint in_high);

void main()
{
    uint in_low = gl_LocalInvocationID.x;
    uint in_high = in_low + block_range_half;
    uint offset = (gl_WorkGroupID.x * LOCAL_DIM_X) << 1;
    shared_buf[in_low] = input_data[in_low + offset];
    shared_buf[in_high] = input_data[in_high + offset];

    dx_algorithm_local(in_low, in_high);

    cpx a = cpx(shared_buf[in_low].x * scalar, shared_buf[in_low].y * scalar);
    cpx b = cpx(shared_buf[in_high].x * scalar, shared_buf[in_high].y * scalar);
    output_data[(bitfieldReverse(in_low + offset) >> leading_bits)] = a;
    output_data[(bitfieldReverse(in_high + offset) >> leading_bits)] = b;
}

void dx_algorithm_local(uint in_low, uint in_high)
{
    float a, x, y;
    cpx w, in_lower, in_upper;
    uint out_i = (in_low << 1);
    uint out_ii = out_i + 1;
    for (int steps = 0; steps < steps_left; ++steps)
    {
        a = local_angle * (in_low & (0xFFFFFFFF << steps));
        w.x = cos(a);
        w.y = sin(a);
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