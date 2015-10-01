
#include "kernelHelper.cl"

void inner_kernel(__global cpx *in, __global cpx *out, float angle, int steps, cUInt lmask, cUInt pmask, cInt dist)
{
    cpx w;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int in_low = tid + (tid & lmask);
    int in_high = in_low + dist;
    cpx in_lower = in[in_low];
    cpx in_upper = in[in_high];
    SIN_COS_F(angle * ((tid << steps) & pmask), &w.y, &w.x);
    cpx_add_sub_mul(&(out[in_low]), &(out[in_high]), in_lower, in_upper, w);
}

__kernel void kernelPartSync(__local cpx *shared, __global cpx *in, __global cpx *out, __global int *sync_in, __global int *sync_out, const float angle, const float bAngle, const int depth, const int breakSize, const cpx scale, const int blocks, const int n2)
{   
    const int lead = 32 - depth;
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
        w_y = sincos(bAngle * ((idx & (0xFFFFFFFF << steps))), &w_x);
        shared[i] = cpxAdd(in_lower, in_upper);
        shared[ii] = cpxMul(cpxSub(in_lower, in_upper), make_cpx(w_x, w_y));
        barrier(0);
    }    
    out[(reverse(idx + offset) >> lead)] = cpxMul(shared[idx], scale);
    out[(reverse(in_high + offset) >> lead)] = cpxMul(shared[in_high], scale);    
}