
#include "helper.cl"

__kernel void kernelGPUSync(local cpx *shared, global cpx *in, global cpx *out, const float angle, const float bAngle, const int depth, const int breakSize, const cpx scale, const int nBlocks, const int n2)
{
    float w_x;
    float w_y;
    int lead = 32 - depth;
    int bit = depth;
    int in_high = n2;
    int idx = get_local_id(0);
    const int i = (idx << 1);
    const int ii = i + 1;

    // To do, run over blocks!

    int offset = get_group_id(0) * get_local_size(0) * 2;            
    in_high += idx;
    shared[idx] = in[idx + offset];
    shared[in_high] = in[in_high + offset];    
    for (int steps = 0; steps < bit; ++steps) {
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


/*
if (get_global_id(0) == 0)
{
printf("local id: %d/%d\t%d/%d\t%d/%d\n", get_local_id(0), get_local_size(0), get_local_id(1), get_local_size(1), get_local_id(2), get_local_size(2));
printf("group id: %d/%d\t%d/%d\t%d/%d\n", get_group_id(0), get_num_groups(0), get_group_id(1), get_num_groups(1), get_group_id(2), get_num_groups(2));
}
*/