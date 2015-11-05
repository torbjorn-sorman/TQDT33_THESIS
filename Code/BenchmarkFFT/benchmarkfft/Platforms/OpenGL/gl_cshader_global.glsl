#version 450
#define LOCAL_DIM_X 1024

struct cpx {
    float x;
    float y;
};

layout(local_size_x = LOCAL_DIM_X) in;

layout(std430, binding = 0) buffer ssbo0 { cpx data[]; };

uniform float global_angle;
uniform uint lmask;
uniform uint dist;
uniform uint steps;

void add_sub_mul(inout cpx low, inout cpx high, cpx w);

void main()
{
    //uint tid = gl_LocalInvocationID.x + gl_WorkGroupID.x * LOCAL_DIM_X;
    uint tid = gl_GlobalInvocationID.x;
    uint in_low = tid + (tid & lmask);
    uint in_high = in_low + dist;
    float a = global_angle * float((tid << steps) & ((dist - 1U) << steps));
    cpx w;
    w.x = cos(a);
    w.y = sin(a);
    add_sub_mul(data[in_low], data[in_high], w);
}

void add_sub_mul(inout cpx low, inout cpx high, cpx w)
{
    float x = low.x - high.x;
    float y = low.y - high.y;
    low.x = low.x + high.x;
    low.y = low.y + high.y;
    high.x = (w.x * x) - (w.y * y);
    high.y = (w.y * x) + (w.x * y);
}