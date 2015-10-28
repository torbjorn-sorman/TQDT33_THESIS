#define UNROLL_FACTOR 4096

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
#define printf(...) 1
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

void add_sub_mul_global(__global cpx* inL, __global cpx* inU, __global cpx *outL, __global cpx *outU, cpx *W)
{
    float x = inL->x - inU->x;
    float y = inL->y - inU->y;
    outL->x = inL->x + inU->x;
    outL->y = inL->y + inU->y;
    outU->x = (W->x * x) - (W->y * y);
    outU->y = (W->y * x) + (W->x * y);
}

void add_sub_mul_local(cpx* inL, cpx* inU, __local cpx *outL, __local cpx *outU, const cpx *W)
{
    float x = inL->x - inU->x;
    float y = inL->y - inU->y;
    outL->x = inL->x + inU->x;
    outL->y = inL->y + inU->y;
    outU->x = (W->x * x) - (W->y * y);
    outU->y = (W->y * x) + (W->x * y);
}

unsigned int reverse(register unsigned int x)
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return((x >> 16) | (x << 16));
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
    int local_id = get_local_id(0);
    if (local_id == 0) { s_in[get_group_id(0)] = goal; }
    if (get_group_id(0) == 1) { // Use get_group_id(0) == 1, if only one block this part will not run.
        if (local_id < get_num_groups(0)) { while (s_in[local_id] != goal && failSafe < goal) { ++failSafe; } }
        barrier(0);
        if (local_id < get_num_groups(0)) { s_out[local_id] = goal; }
    }
    failSafe = 0;
    if (local_id == 0) { while (s_out[get_group_id(0)] != goal && failSafe < goal) { ++failSafe; } }
    barrier(0);
}

void inner_kernel(__global cpx *in, __global cpx *out, float angle, int steps, unsigned int lmask, int dist)
{
    cpx w;
    int tid = get_global_id(0);
    int in_low = tid + (tid & lmask);
    in += in_low;
    out += in_low;
    w.y = sincos(angle * ((tid << steps) & ((dist - 1) << steps)), &w.x);
    add_sub_mul_global(in, in + dist, out, out + dist, &w);
}

int algorithm_cross_group(__global cpx *in, __global cpx *out, __global int *sync_in, __global int *sync_out, int bit_start, int steps_gpu, float angle, int number_of_blocks, int n_half)
{    
    int dist = n_half;
    int steps = 0;
    group_sync_init(sync_in, sync_out);
    inner_kernel(in, out, angle, steps, 0xFFFFFFFF << bit_start, dist);
    group_sync(sync_in, sync_out, number_of_blocks + steps); 
#pragma unroll UNROLL_FACTOR      
    for (int bit = bit_start - 1; bit > steps_gpu; --bit) {
        dist >>= 1;
        ++steps;
        inner_kernel(out, out, angle, steps, 0xFFFFFFFF << bit, dist);
        group_sync(sync_in, sync_out, number_of_blocks + steps);
    }
    return steps_gpu + 1;
}

void algorithm_partial(__local cpx *shared, int in_high, float angle, int bit)
{
    cpx w, in_lower, in_upper;
    int local_id = get_local_id(0);
    __local cpx *out_i = shared + (local_id << 1);
    __local cpx *out_ii = out_i + 1;
    __local cpx *in_l = shared + local_id;
    __local cpx *in_u = shared + in_high;    
#pragma unroll UNROLL_FACTOR
    for (int steps = 0; steps < bit; ++steps) {
        w.y = sincos(angle * (local_id & (0xFFFFFFFF << steps)), &w.x);
        in_lower = *in_l;
        in_upper = *in_u;
        barrier(0);
        add_sub_mul_local(&in_lower, &in_upper, out_i, out_ii, &w);
        barrier(0);
    }
}

// GPU takes care of overall syncronization
__kernel void opencl_kernel_global(__global cpx *in, __global cpx *out, float angle, unsigned int lmask, int steps, int dist)
{
    inner_kernel(in, out, angle, steps, lmask, dist);
}

// CPU takes care of overall syncronization, limited in problem sizes that can be solved.
// Can be combined with opencl_kernel_global in a manner that the opencl_kernel_global is run until problem can be split into smaller parts.
__kernel void opencl_kernel_local(__global cpx *in, __global cpx *out, __global int *sync_in, __global int *sync_out, __local cpx *shared, float angle, float local_angle, int steps_left, int leading_bits, int steps_gpu, float scalar, int number_of_blocks, const int n_half)
{    
    
    int bit = steps_left;
    int in_high = n_half;
    if (number_of_blocks > 1) {
        bit = algorithm_cross_group(in, out, sync_in, sync_out, steps_left - 1, steps_gpu, angle, number_of_blocks, in_high);
        in_high >>= log2_32(number_of_blocks);
        in = out;
    }
    int in_low = get_local_id(0);
    int offset = get_group_id(0) * get_local_size(0) * 2;
    in_high += in_low;    
    in += offset;
    shared[in_low] = in[in_low];
    shared[in_high] = in[in_high];
    barrier(0);
    algorithm_partial(shared, in_high, local_angle, bit);
    cpx src_low = { shared[in_low].x * scalar, shared[in_low].y * scalar };
    cpx src_high = { shared[in_high].x * scalar, shared[in_high].y * scalar };
    out[(reverse(in_low + offset) >> leading_bits)] = src_low;
    out[(reverse(in_high + offset) >> leading_bits)] = src_high;
}

__kernel void opencl_kernel_global_row(__global cpx *in, __global cpx *out, float angle, unsigned int lmask, int steps, int dist)
{
    cpx w;
    int col_id = get_group_id(1) * get_local_size(0) + get_local_id(0);
    int in_low = (col_id + (col_id & lmask)) + get_group_id(0) * get_num_groups(0);
    in += in_low;
    out += in_low;
    w.y = sincos(angle * ((col_id << steps) & ((dist - 1) << steps)), &w.x);
    add_sub_mul_global(in, in + dist, out, out + dist, &w);
}

__kernel void opencl_kernel_local_row(__global cpx *in, __global cpx *out, __local cpx shared[], float local_angle, int steps_left, float scalar, int n_per_block)
{
    int leading_bits = (32 - log2_32((int)get_num_groups(0)));
    int in_low = get_local_id(0);
    int in_high = (n_per_block >> 1) + in_low;
    int row_start = get_num_groups(0) * get_group_id(0);
    int row_offset = get_group_id(1) * get_local_size(0) * 2;
    in += row_start + row_offset;
    out += row_start;
    shared[in_low]  = in[in_low];
    shared[in_high] = in[in_high];
    algorithm_partial(shared, in_high, local_angle, steps_left);
    cpx src_low = { shared[in_low].x * scalar, shared[in_low].y * scalar };
    cpx src_high = { shared[in_high].x * scalar, shared[in_high].y * scalar };
    out[(reverse(in_low + row_offset) >> leading_bits)] = src_low;
    out[(reverse(in_high + row_offset) >> leading_bits)] = src_high;
}

__kernel void opencl_kernel_local_col(__global cpx *in, __global cpx *out, __local cpx shared[], float local_angle, int steps_left, float scalar, int n)
{

    int in_low = get_local_id(0);
    int in_high = (n >> 1) + in_low;
    int colOffset = get_group_id(1) * get_local_size(0) * 2;
    in += (in_low + colOffset) * n + get_group_id(0);
    out += get_group_id(0);
    shared[in_low] = *in;
    shared[in_high] = *(in + ((n >> 1) * n));
    barrier(0);
    algorithm_partial(shared, in_high, local_angle, steps_left);
    int leading_bits = 32 - log2_32((int)get_num_groups(0));
    cpx src_low = { shared[in_low].x * scalar, shared[in_low].y * scalar };
    cpx src_high = { shared[in_high].x * scalar, shared[in_high].y * scalar };
    out[(reverse(in_low + colOffset) >> leading_bits) * n] = src_low;
    out[(reverse(in_high + colOffset) >> leading_bits) * n] = src_high;
}

#define TILE_DIM 64
#define THREAD_TILE_DIM 32

__kernel void opencl_transpose_kernel(__global cpx *in, __global cpx *out, __local cpx tile[TILE_DIM][TILE_DIM + 1], int n)
{
    // Write to shared from Global (in)
    int x = get_group_id(0) * TILE_DIM + get_local_id(0);
    int y = get_group_id(1) * TILE_DIM + get_local_id(1);    
#pragma unroll UNROLL_FACTOR    
    for (int j = 0; j < TILE_DIM; j += THREAD_TILE_DIM) {
#pragma unroll UNROLL_FACTOR
        for (int i = 0; i < TILE_DIM; i += THREAD_TILE_DIM) {
            tile[get_local_id(1) + j][get_local_id(0) + i] = in[(y + j) * n + (x + i)];            
        }
    }
    barrier(0);
    // Write to global
    x = get_group_id(1) * TILE_DIM + get_local_id(0);
    y = get_group_id(0) * TILE_DIM + get_local_id(1);
#pragma unroll UNROLL_FACTOR
    for (int j = 0; j < TILE_DIM; j += THREAD_TILE_DIM){ 
#pragma unroll UNROLL_FACTOR
        for (int i = 0; i < TILE_DIM; i += THREAD_TILE_DIM) {
            out[(y + j) * n + (x + i)] = tile[get_local_id(0) + i][get_local_id(1) + j];
        }
    }
}

__kernel void opencl_timestamp_kernel()
{
    // There be none here!
}