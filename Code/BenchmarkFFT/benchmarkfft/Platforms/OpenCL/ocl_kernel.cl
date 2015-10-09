#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define get_global_id(a) 1
#define get_group_id(a) 1
#define get_local_id(a) 1
#define get_num_groups(a) 1
#define get_local_size(a) 1
#define barrier(a)
#define sincos(a,x) 1
#endif

typedef struct {
    float x;
    float y;
} cpx;

int log2_32(unsigned int value)
{
    const int tab32[32] = {
        0, 9, 1, 10, 13, 21, 2, 29,
        11, 14, 16, 18, 22, 25, 3, 30,
        8, 12, 20, 28, 15, 17, 24, 7,
        19, 27, 23, 6, 26, 5, 4, 31 };
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return tab32[(unsigned int)(value * 0x07C4ACDD) >> 27];
}

cpx make_cpx(const float x, const float y)
{
    cpx a;
    a.x = x;
    a.y = y;
    return a;
}

cpx cpxAdd(const cpx a, const cpx b)
{
    return make_cpx(
        a.x + b.x,
        a.y + b.y
        );
}

cpx cpxSub(const cpx a, const cpx b)
{
    return make_cpx(
        a.x - b.x,
        a.y - b.y
        );
}

cpx cpxMul(const cpx a, const cpx b)
{
    return make_cpx(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
        );
}

void cpxAddSubMul(__global cpx* in, int l, int u, __global cpx *outL, __global cpx *outU, cpx *W)
{
    float x = in[l].x - in[u].x;
    float y = in[l].y - in[u].y;
    outL->x = in[l].x + in[u].x;
    outL->y = in[l].y + in[u].y;
    outU->x = (W->x * x) - (W->y * y);
    outU->y = (W->y * x) + (W->x * y);
}
/*
void cpxAddSubMulRef(__global cpx* inL, __global cpx* inU, __global cpx *outL, __global cpx *outU, cpx *W)
{
    float x = inL->x - inU->x;
    float y = inL->y - inU->y;
    outL->x = inL->x + inU->x;
    outL->y = inL->y + inU->y;
    outU->x = (W->x * x) - (W->y * y);
    outU->y = (W->y * x) + (W->x * y);
}
*/
void cpxAddSubMulR(cpx* inL, cpx* inU, __local cpx *outL, __local cpx *outU, cpx *W)
{
    float x = inL->x - inU->x;
    float y = inL->y - inU->y;
    outL->x = inL->x + inU->x;
    outL->y = inL->y + inU->y;
    outU->x = (W->x * x) - (W->y * y);
    outU->y = (W->y * x) + (W->x * y);
}
/*
void cpxAddSubMulSync(cpx *inL, cpx *inU, __local cpx *out, cpx *W)
{
    float x = inL->x - inU->x;
    float y = inL->y - inU->y;
    out->x = inL->x + inU->x;
    out->y = inL->y + inU->y;
    (++out)->x = (W->x * x) - (W->y * y);
    out->y = (W->y * x) + (W->x * y);
}
*/
unsigned int reverse(register unsigned int x)
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return((x >> 16) | (x << 16));
}

void mem_gtos(int low, int high, int offset, __local cpx *shared, __global cpx *device)
{
    shared[low] = device[low + offset];
    shared[high] = device[high + offset];
}

void mem_stog_db(int low, int high, int offset, unsigned int lead, cpx scale, __local cpx *shared, __global cpx *device)
{
    device[(reverse(low + offset) >> lead)] = cpxMul(shared[low], scale);
    device[(reverse(high + offset) >> lead)] = cpxMul(shared[high], scale);
}

void group_sync_init(__global int *s_in, __global int *s_out)
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
void group_sync(__global int *s_in, __global int *s_out, const int goal)
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

void inner_kernel(__global cpx *in, __global cpx *out, float angle, int steps, unsigned int lmask, int dist)
{
    int in_low = get_global_id(0) + (get_global_id(0) & lmask);
    int in_high = in_low + dist;
    cpx w;
    w.y = sincos(angle * ((get_global_id(0) << steps) & ((dist - 1) << steps)), &w.x);
    cpxAddSubMul(in, in_low, in_high, &out[in_low], &out[in_high], &w);    
}

int algorithm_cross_group(__global cpx *in, __global cpx *out, __global int *sync_in, __global int *sync_out, int bit_start, int breakSize, float angle, int nBlocks, int n2)
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

void algorithm_partial(__local cpx *shared, int in_high, float angle, int bit)
{
    float x, y;
    cpx in_lower, in_upper;
    cpx w;
    __local cpx *out_i = &shared[(get_local_id(0) << 1)];
    __local cpx *out_ii = out_i + 1;
    __local cpx *in_l = &shared[get_local_id(0)];
    __local cpx *in_u = &shared[in_high];
    for (int steps = 0; steps < bit; ++steps) {
        in_lower = *in_l;
        in_upper = *in_u;
        barrier(0);
        w.y = sincos(angle * (get_local_id(0) & (0xFFFFFFFF << steps)), &w.x);
        cpxAddSubMulR(&in_lower, &in_upper, out_i, out_ii, &w);
        barrier(0);
    }
}

// GPU takes care of overall syncronization
__kernel void kernelCPU(__global cpx *in, __global cpx *out, float angle, unsigned int lmask, int steps, int dist)
{
    inner_kernel(in, out, angle, steps, lmask, dist);
}

// CPU takes care of overall syncronization, limited in problem sizes that can be solved.
// Can be combined with kernelCPU in a manner that the kernelCPU is run until problem can be split into smaller parts.
__kernel void kernelGPU(__global cpx *in, __global cpx *out, __global int *sync_in, __global int *sync_out, __local cpx *shared, float angle, float bAngle, int depth, int lead, int breakSize, cpx scale, int nBlocks, const int n2)
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

__kernel void kernelCPU2D(__global cpx *in, __global cpx *out, float angle, unsigned int lmask, int steps, int dist)
{
    cpx w;
    int col_id = get_group_id(1) * get_local_size(0) + get_local_id(0);
    int in_low = (col_id + (col_id & lmask)) + get_group_id(0) * get_num_groups(0);
    int in_high = in_low + dist;
    w.y = sincos(angle * ((col_id << steps) & ((dist - 1) << steps)), &w.x);
    cpxAddSubMul(in, in_low, in_high, &out[in_low], &out[in_low], &w);
}

__kernel void kernelGPU2D(__global cpx *in, __global cpx *out, __local cpx *shared, float angle, float bAngle, int depth, cpx scale, int nBlock)
{
    int rowStart = get_num_groups(0) * get_group_id(0);
    int in_high = (nBlock >> 1) + get_local_id(0);
    int rowOffset = get_group_id(1) * get_local_size(0) * 2;
    mem_gtos(get_local_id(0), in_high, rowOffset, shared, &(in[rowStart]));
    algorithm_partial(shared, in_high, bAngle, depth);
    mem_stog_db(get_local_id(0), in_high, rowOffset, (32 - log2_32((int)get_num_groups(0))), scale, shared, &(out[rowStart]));
}

#define TILE_DIM 64
#define THREAD_TILE_DIM 32

__kernel void kernelTranspose(__global cpx *in, __global cpx *out, __local cpx *tile, int n)
{
    // Write to shared from Global (in)
    int x = get_group_id(0) * TILE_DIM + get_local_id(0);
    int y = get_group_id(1) * TILE_DIM + get_local_id(1);
    for (int j = 0; j < TILE_DIM; j += THREAD_TILE_DIM)
        for (int i = 0; i < TILE_DIM; i += THREAD_TILE_DIM)
            tile[(get_local_id(1) + j) * TILE_DIM + (get_local_id(0) + i)] = in[(y + j) * n + (x + i)];

    barrier(0);
    // Write to global
    x = get_group_id(1) * TILE_DIM + get_local_id(0);
    y = get_group_id(0) * TILE_DIM + get_local_id(1);
    for (int j = 0; j < TILE_DIM; j += THREAD_TILE_DIM)
        for (int i = 0; i < TILE_DIM; i += THREAD_TILE_DIM)
            out[(y + j) * n + (x + i)] = tile[(get_local_id(0) + i) * TILE_DIM + (get_local_id(1) + j)];
}