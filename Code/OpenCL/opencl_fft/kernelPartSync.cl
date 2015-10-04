
#include "kernelHelper.cl"

void group_sync_init(syn_buf *s_in, syn_buf *s_out)
{
    if (get_global_id(0) < get_num_groups(0)) {
        s_in[get_global_id(0)] = 0;
        s_out[get_global_id(0)] = 0;
    }
}

// This is currently a "hack" to get it to work! Refer to CUDA that actually works (presumably). 
// Groups do not run i parallel even if hardware would allow it, this is not guaranteed by the program model but is a nice feature to have.
// By escaping the while loops the syncronization is not guaranteed but have not failed during test-execution.
// My guess is that groups do not run the same way as a CUDA block. In CUDA one might expect blocks to run in parallel if possible (GTX670 have 7 cores).
// Perhaps there is another way to use physical cores where scheduling does not put groups in parallel the same way.
void group_sync(syn_buf *s_in, syn_buf *s_out, const int goal)
{
    int failSafe = 0;
    if (get_local_id(0) == 0) { s_in[get_group_id(0)] = goal; }
    if (get_group_id(0) == 1) { // Use get_group_id(0) == 1, if only one block this part will not run.
        if (get_local_id(0) < get_num_groups(0)) { while (s_in[get_local_id(0)] != goal && failSafe < goal) { ++failSafe; } }
        barrier(0);
        if (get_local_id(0) < get_num_groups(0)) { s_out[get_local_id(0)] = goal; }
    }
    failSafe = 0;
    if (get_local_id(0) == 0) { while (s_out[get_group_id(0)] != goal && failSafe < goal) { ++failSafe; } }
    barrier(0);
}

void inner_kernel(dev_buf *in, dev_buf *out, float angle, int steps, unsigned int lmask, int dist)
{
    int in_low = get_global_id(0) + (get_global_id(0) & lmask);
    int in_high = in_low + dist;
    cpx in_lower = in[in_low];
    cpx in_upper = in[in_high];
    butterflyDev(out, in_lower, in_upper, in_low, in_high, angle * ((get_global_id(0) << steps) & ((dist - 1) << steps)));
}

int algorithm_cross_group(dev_buf *in, dev_buf *out, syn_buf *sync_in, syn_buf *sync_out, int bit_start, int breakSize, float angle, int nBlocks, int n2)
{    
    int dist = n2;
    int steps = 0;
    group_sync_init(sync_in, sync_out);
    inner_kernel(in, out, angle, steps, 0xFFFFFFFF << bit_start, dist);
    group_sync(sync_in, sync_out, nBlocks + steps);
    
    for (int bit = bit_start - 1; bit > breakSize; --bit) {
        dist = dist >> 1;
        ++steps;
        inner_kernel(out, out, angle, steps, 0xFFFFFFFF << bit, dist);
        group_sync(sync_in, sync_out, nBlocks + steps);
    }
    return breakSize + 1;
}

void algorithm_partial(grp_buf *shared, int in_high, float angle, int bit)
{
    cpx in_lower, in_upper;
    int i = (get_local_id(0) << 1);
    int ii = i + 1;
    for (int steps = 0; steps < bit; ++steps) {
        in_lower = shared[get_local_id(0)];
        in_upper = shared[in_high];
        barrier(0);
        butterflyGrp(shared, in_lower, in_upper, i, ii, angle * ((get_local_id(0) & (0xFFFFFFFF << steps))));
        barrier(0);
    }
}

// GPU takes care of overall syncronization
__kernel void kernelCPU(dev_buf *in, dev_buf *out, float angle, unsigned int lmask, int steps, int dist)
{
    inner_kernel(in, out, angle, steps, lmask, dist);
}

// CPU takes care of overall syncronization, limited in problem sizes that can be solved.
// Can be combined with kernelCPU in a manner that the kernelCPU is run until problem can be split into smaller parts.
__kernel void kernelGPU(dev_buf *in, dev_buf *out, syn_buf *sync_in, syn_buf *sync_out, grp_buf *shared, float angle, float bAngle, int depth, int lead, int breakSize, cpx scale, int nBlocks, const int n2)
{    
    int bit = depth;
    int in_high = n2;
    if (nBlocks > 1) {
        bit = algorithm_cross_group(in, out, sync_in, sync_out, depth - 1, breakSize, angle, nBlocks, in_high);
        in_high >>= log2_32(nBlocks);
        in = out;
    }
    int offset = get_group_id(0) * get_local_size(0) * 2;
    in_high += get_local_id(0);
    mem_gtos(get_local_id(0), in_high, offset, shared, in);
    algorithm_partial(shared, in_high, bAngle, bit);
    mem_stog_db(get_local_id(0), in_high, offset, lead, scale, shared, out);
}