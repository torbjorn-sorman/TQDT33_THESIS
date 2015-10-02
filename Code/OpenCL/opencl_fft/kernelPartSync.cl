
#include "kernelHelper.cl"

void inner_kernel(dev_buf *in, dev_buf *out, float angle, int steps, unsigned int lmask, unsigned int pmask, int dist)
{
    int in_low = get_global_id(0) + (get_global_id(0) & lmask);
    int in_high = in_low + dist;
    cpx in_lower = in[in_low];
    cpx in_upper = in[in_high];
    butterflyDev(out, in_lower, in_upper, in_low, in_high, angle * ((get_global_id(0) << steps) & pmask));
}

int algorithm_cross_group(dev_buf *in, dev_buf *out, syn_buf *sync_in, syn_buf *sync_out, int bit_start, int breakSize, float angle, int nBlocks, int n2)
{    
    int dist = n2;
    int steps = 0;
    group_sync_init(sync_in, sync_out);
    inner_kernel(in, out, angle, steps, 0xFFFFFFFF << bit_start, (dist - 1) << steps, dist);
    group_sync(sync_in, sync_out, nBlocks + steps);
    for (int bit = bit_start - 1; bit > breakSize; --bit) {
        dist = dist >> 1;
        ++steps;
        inner_kernel(out, out, angle, steps, 0xFFFFFFFF << bit, (dist - 1) << steps, dist);
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
__kernel void kernelCPU(dev_buf *in, dev_buf *out, float angle, unsigned int lmask, unsigned int pmask, int steps, int dist)
{
    printf("kernelCPU Global id: %d\n", get_global_id(0));
    inner_kernel(in, out, angle, steps, lmask, pmask, dist);
}

// CPU takes care of overall syncronization, limited in problem sizes that can be solved.
// Can be combined with kernelCPU in a manner that the kernelCPU is run until problem can be split into smaller parts.
__kernel void kernelGPU(dev_buf *in, dev_buf *out, syn_buf *sync_in, syn_buf *sync_out, grp_buf *shared, float angle, float bAngle, int depth, int lead, int breakSize, cpx scale, int nBlocks, const int n2)
{
    printf("kernelGPU Global id: %d\n", get_global_id(0));
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