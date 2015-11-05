#version 450
#define LOCAL_DIM_X 1024

struct cpx {
    float x;
    float y;
};

layout(local_size_x = LOCAL_DIM_X) in;

layout(std430, binding = 0) buffer ssbo0{ cpx data[]; };
layout(std430, binding = 0) buffer ssbo1{ cpx data_tmp[]; };

uniform float global_angle;
uniform uint lmask;
uniform uint dist;
uniform uint steps;

//void add_sub_mul(inout cpx low, inout cpx high, cpx w);

void main()
{
    uint tid = gl_GlobalInvocationID.x;
    uint in_low = tid + (tid & lmask);
    uint in_high = in_low + dist;        
    float a = global_angle * ((tid << steps) & ((dist - 1U) << steps));
    cpx w;
    w.x = cos(a);
    w.y = sin(a);
    cpx low = data[in_low];
    cpx high = data[in_high];
    float x = low.x - high.x;
    float y = low.y - high.y;
    /*
    if (gl_LocalInvocationID.x == 0 && gl_WorkGroupID.x == 0) {
        data[0].x = in_low;
        data[0].y = in_high;
        data[1].x = lmask;        
        data[2].x = dist;
        data[3].x = steps;
        data[4].x = global_angle;
        data[1].y = a;
        data[2].y = w.x;
        data[3].y = w.y;
        data[4].y = 0;
    }


    */
    data[in_low].x = low.x + high.x;
    data[in_low].y = low.y + high.y;
    data[in_high].x = (w.x * x) - (w.y * y);
    data[in_high].y = (w.y * x) + (w.x * y);
    //*/
    //add_sub_mul(data[in_low], data[in_high], w);
}
/*
void add_sub_mul(inout cpx low, inout cpx high, cpx w)
{
    float x = low.x - high.x;
    float y = low.y - high.y;
    low.x = low.x + high.x;
    low.y = low.y + high.y;
    high.x = (w.x * x) - (w.y * y);
    high.y = (w.y * x) + (w.x * y);
}
*/