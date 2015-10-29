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

void add_sub_mul_global(__global cpx* low, __global cpx* high, cpx* w)
{
    float x = low->x - high->x;
    float y = low->y - high->y;
    low->x = low->x + high->x;
    low->y = low->y + high->y;
    high->x = (w->x * x) - (w->y * y);
    high->y = (w->y * x) + (w->x * y);
}

void add_sub_mul_local(cpx* inL, cpx* inU, __local cpx *outL, __local cpx *outU, const cpx *w)
{
    float x = inL->x - inU->x;
    float y = inL->y - inU->y;
    outL->x = inL->x + inU->x;
    outL->y = inL->y + inU->y;
    outU->x = (w->x * x) - (w->y * y);
    outU->y = (w->y * x) + (w->x * y);
}

unsigned int reverse(register unsigned int x)
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return((x >> 16) | (x << 16));
}

void algorithm_partial(__local cpx *shared, int in_high, float angle, int bit)
{
    cpx w, in_lower, in_upper;
    int local_id = get_local_id(0);
    __local cpx *out_i = shared + (local_id << 1);
    __local cpx *out_ii = out_i + 1;
    __local cpx *in_l = shared + local_id;
    __local cpx *in_u = shared + in_high;
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
__kernel void opencl_kernel_global(__global cpx *in, float angle, unsigned int lmask, int steps, int dist)
{
    cpx w;
    int tid = get_global_id(0);
    in += tid + (tid & lmask);
    w.y = sincos(angle * ((tid << steps) & ((dist - 1) << steps)), &w.x);
    add_sub_mul_global(in, in + dist, &w);
}

// CPU takes care of overall syncronization, limited in problem sizes that can be solved.
// Can be combined with opencl_kernel_global in a manner that the opencl_kernel_global is run until problem can be split into smaller parts.
__kernel void opencl_kernel_local(__global cpx *in, __global cpx *out, __local cpx *shared, float local_angle, int steps_left, int leading_bits, float scalar, const int n_half)
{      
    int in_low = get_local_id(0);
    int in_high = n_half + in_low;
    int offset = get_group_id(0) * get_local_size(0) * 2;          
    in += offset;
    shared[in_low] = in[in_low];
    shared[in_high] = in[in_high];
    algorithm_partial(shared, in_high, local_angle, steps_left);
    cpx src_low = { shared[in_low].x * scalar, shared[in_low].y * scalar };
    cpx src_high = { shared[in_high].x * scalar, shared[in_high].y * scalar };
    out[(reverse(in_low + offset) >> leading_bits)] = src_low;
    out[(reverse(in_high + offset) >> leading_bits)] = src_high;
}

__kernel void opencl_kernel_global_row(__global cpx *in, float angle, unsigned int lmask, int steps, int dist)
{
    cpx w;
    int col_id = get_group_id(1) * get_local_size(0) + get_local_id(0);
    in += (col_id + (col_id & lmask)) + get_group_id(0) * get_num_groups(0);
    w.y = sincos(angle * ((col_id << steps) & ((dist - 1) << steps)), &w.x);
    add_sub_mul_global(in, in + dist, &w);
}

__kernel void opencl_kernel_local_row(__global cpx *in, __global cpx *out, __local cpx shared[], float local_angle, int steps_left, int leading_bits, float scalar, int n_per_block)
{
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

#define TILE_DIM 64
#define THREAD_TILE_DIM 32

__kernel void opencl_transpose_kernel(__global cpx *in, __global cpx *out, __local cpx tile[TILE_DIM][TILE_DIM + 1], int n)
{
    // Write to shared from Global (in)
    int x = get_group_id(0) * TILE_DIM + get_local_id(0);
    int y = get_group_id(1) * TILE_DIM + get_local_id(1);    
#pragma unroll    
    for (int j = 0; j < TILE_DIM; j += THREAD_TILE_DIM) {
#pragma unroll
        for (int i = 0; i < TILE_DIM; i += THREAD_TILE_DIM) {
            tile[get_local_id(1) + j][get_local_id(0) + i] = in[(y + j) * n + (x + i)];            
        }
    }
    barrier(0);
    // Write to global
    x = get_group_id(1) * TILE_DIM + get_local_id(0);
    y = get_group_id(0) * TILE_DIM + get_local_id(1);
#pragma unroll
    for (int j = 0; j < TILE_DIM; j += THREAD_TILE_DIM){ 
#pragma unroll
        for (int i = 0; i < TILE_DIM; i += THREAD_TILE_DIM) {
            out[(y + j) * n + (x + i)] = tile[get_local_id(0) + i][get_local_id(1) + j];
        }
    }
}

__kernel void opencl_timestamp_kernel()
{
    // There be none here!
    // Kernel only used as event trigger to get timestamps!
}