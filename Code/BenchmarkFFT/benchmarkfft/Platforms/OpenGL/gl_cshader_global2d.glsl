#version 450
#define LOCAL_DIM_X 1024

struct cpx {
    float x;
    float y;
};

layout(local_size_x = LOCAL_DIM_X) in;

layout(std430, binding = 0) buffer ssbo0{ cpx data[]; };
layout(std430, binding = 1) buffer ssbo1{ cpx data_tmp[]; };

uniform float global_angle;
uniform uint lmask;
uniform uint dist;
uniform uint steps;

void main()
{
    uint col = gl_WorkGroupID.y * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint in_low = col + (col & lmask) + gl_WorkGroupID.x * gl_NumWorkGroups.x;
    uint in_high = in_low + dist;
    float a = global_angle * float((col << steps) & ((dist - 1) << steps));
    cpx w;
    w.x = cos(a);
    w.y = sin(a);
    cpx low = data[in_low];
    cpx high = data[in_high];
    float x = low.x - high.x;
    float y = low.y - high.y;
    data[in_low].x = low.x + high.x;
    data[in_low].y = low.y + high.y;
    data[in_high].x = (w.x * x) - (w.y * y);
    data[in_high].y = (w.y * x) + (w.x * y);
}