
#include "kernelHelper.cl"

__kernel void kernelGPUSync(__local cpx *shared, __global cpx *in, __global cpx *out, const float angle, const int depth, const int lead, const cpx scale, const int n2)
{   
    const int idx = get_local_id(0);
    const int i = (idx << 1);
    const int ii = i + 1;
    const int offset = get_group_id(0) * get_local_size(0) * 2;  
    const int in_high = n2 + idx;
    float w_x, w_y;
    shared[idx] = in[idx + offset];
    shared[in_high] = in[in_high + offset];
    for (int steps = 0; steps < depth; ++steps) {
        cpx in_lower = shared[idx];
        cpx in_upper = shared[in_high];
        barrier(0);
        w_y = sincos(angle * ((idx & (0xFFFFFFFF << steps))), &w_x);
        shared[i] = cpxAdd(in_lower, in_upper);
        shared[ii] = cpxMul(cpxSub(in_lower, in_upper), make_cpx(w_x, w_y));
        barrier(0);
    }    
    out[(reverse(idx + offset) >> lead)] = cpxMul(shared[idx], scale);
    out[(reverse(in_high + offset) >> lead)] = cpxMul(shared[in_high], scale);    
}