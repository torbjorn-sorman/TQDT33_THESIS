
#include "helper.cl"

__kernel void kernelGPUSync(local cpx *shared, global cpx *in, global cpx *out, global int *sync_in, global int *sync_out, const float angle, const float bAngle, const int depth, const int breakSize, const cpx scale, const int nBlocks, const int n2)
{
    float w_x;
    float w_y;
    int lead = 32 - depth;
    int bit = depth;
    int in_high = n2;
    int idx = get_local_id(0);
    const int i = (idx << 1);
    const int ii = i + 1;    
    
    if (nBlocks > 1) {   
        const int gid = get_global_id(0);
        int dist = n2;
        int steps = 0;
        int bit_start = depth - 1;
        cpx w;

        int gid = get_global_id(0);
        if (gid < blocks) {
            sync_in[gid] = 0;
            sync_out[gid] = 0;
        }

        unsigned int lmask = 0xFFFFFFFF << bit_start;
        unsigned int pmask = (dist - 1) << steps;
        int in_low = gid + (gid & lmask);
        int in_high = in_low + dist;
        cpx in_lower = in[in_low];
        cpx in_upper = in[in_high];
        SIN_COS_F(angle * ((gid << steps) & pmask), &w.y, &w.x);
        cpx_add_sub_mul(&(out[in_low]), &(out[in_high]), in_lower, in_upper, w);
        
        int goal = nBlocks + steps;
        int bid = get_group_id(0);
        if (idx == 0) { sync_in[bid] = goal; }
        if (bid == 1) { // Use bid == 1, if only one block this part will not run.
        if (idx < nBlocks) { while (sync_in[idx] != goal){} }
        barrier(0);
        if (idx < nBlocks) { sync_out[idx] = goal; }
        }
        if (idx == 0) { while (sync_out[bid] != goal) {} }
        barrier(0);


        for (int bit = bit_start - 1; bit > breakSize; --bit) {
            dist = dist >> 1;
            ++steps;
            lmask = 0xFFFFFFFF << bit;
            pmask = (dist - 1) << steps;
            in_low = gid + (gid & lmask);
            in_high = in_low + dist;
            in_lower = in[in_low];
            in_upper = in[in_high];
            SIN_COS_F(angle * ((gid << steps) & pmask), &w.y, &w.x);
            cpx_add_sub_mul(&(out[in_low]), &(out[in_high]), in_lower, in_upper, w);
            

            goal = nBlocks + steps;
            int bid = get_group_id(0);
            if (idx == 0) { sync_in[bid] = goal; }
            if (bid == 1) { // Use bid == 1, if only one block this part will not run.
            if (idx < nBlocks) { while (sync_in[idx] != goal){} }
            barrier(0);
            if (idx < nBlocks) { sync_out[idx] = goal; }
            }
            if (idx == 0) { while (sync_out[bid] != goal) {} }
            barrier(0);

        }
        bit = breakSize + 1;
        in_high >>= log2(nBlocks);
        in = out;
    }

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