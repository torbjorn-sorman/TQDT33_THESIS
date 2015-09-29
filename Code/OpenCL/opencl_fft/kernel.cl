
#include "helper.cl"
/*
static __inline int algorithm_complete(cpx *in, cpx *out, const int bit_start, const int breakSize, const float angle, const int nBlocks, const int n2)
{
    int tid = get_global_id(0);
    int dist = n2;
    int steps = 0;
    init_sync(tid, nBlocks);
    inner_kernel(in, out, angle, steps, tid, 0xFFFFFFFF << bit_start, (dist - 1) << steps, dist, nBlocks);
    for (int bit = bit_start - 1; bit > breakSize; --bit) {
        dist = dist >> 1;
        ++steps;
        inner_kernel(out, out, angle, steps, tid, 0xFFFFFFFF << bit, (dist - 1) << steps, dist, nBlocks);
    }
    return breakSize + 1;
}
*/
static __inline void algorithm_partial(__local cpx *shared, const int in_high, const float angle, const int bit)
{
    cpx w, in_lower, in_upper;
    int idx = get_local_id(0);
    const int i = (idx << 1);
    const int ii = i + 1;
    for (int steps = 0; steps < bit; ++steps) {
        in_lower = shared[idx];
        in_upper = shared[in_high];
        barrier(0);
        w.y = sincos(angle * ((idx & (0xFFFFFFFF << steps))), &w.x);
        cpx_add_sub_mul(&(shared[i]), &(shared[ii]), in_lower, in_upper, w);
        barrier(0);
    }
}

static __inline void mem_gtos(int low, int high, int offset, __local cpx *shared, __global cpx *out)
{
    shared[low] = out[low + offset];
    shared[high] = out[high + offset];
}
static __inline void mem_stog_db(int low, int high, int offset, unsigned int lead, cpx scale, cpx *shared, __global cpx *out)
{
    out[reverseBitOrder(low + offset, lead)] = cpxMul(shared[low], scale);
    out[reverseBitOrder(high + offset, lead)] = cpxMul(shared[high], scale);
}

__kernel void kernelGPUSync(__global cpx *in, __global cpx *out, const float angle, const float bAngle, const int depth, const int breakSize, const cpx scale, const int nBlocks, const int n2)
{
    __local cpx shared[2048];
    int bit = depth;
    int in_high = n2;
    /*
    if (nBlocks > 1) {
        bit = algorithm_complete(in, out, depth - 1, breakSize, angle, nBlocks, in_high);
        in_high >>= log2(nBlocks);
        in = out;
    }
    */
    int offset = get_group_id(0) * get_local_size(0) * 2;
    in_high += get_local_id(0);
    mem_gtos(get_local_id(0), in_high, offset, shared, in);
    algorithm_partial(shared, in_high, bAngle, bit);
    mem_stog_db(get_local_id(0), in_high, offset, 32 - depth, scale, shared, out);
    return;
}