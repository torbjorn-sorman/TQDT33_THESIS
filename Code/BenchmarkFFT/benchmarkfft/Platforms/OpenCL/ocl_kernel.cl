#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define get_global_id(a) 1
#define get_group_id(a) 1
#define get_local_id(a) 1
#define get_num_groups(a) 1
#define get_local_size(a) 1
#define get_global_size(a) 1
#define barrier(a)
#define sincos(a,x) 1
#define printf(...) 1
#endif

#define OCL_TILE_DIM 64
#define OCL_BLOCK_DIM 16

typedef struct {
    float x;
    float y;
} cpx;

#define OCL_BATCH_ID (get_group_id(0))
#define OCL_N_POINTS ((get_num_groups(1) * get_local_size(1)) << 1)
#define OCL_THREAD_ID (get_group_id(1) * get_local_size(1) + get_local_id(1))
#define OCL_BLOCK_OFFSET (get_group_id(1) * (get_local_size(1) << 1))

#define OCL_OCL_BATCH_ID_2D (blockIdx.z)
#define OCL_OCL_N_POINTS_2D (gridDim.x * gridDim.x)
#define OCL_OFFSET_2D ((get_group_id(0) + blockIdx.z * gridDim.x) * gridDim.x)
#define OCL_COL_ID (get_group_id(1) * get_local_size(0) + get_local_id(0))

#define OCL_IMG_DIST (blockIdx.z * gridDim.x * gridDim.x * OCL_TILE_DIM * OCL_TILE_DIM)

#define OCL_IN_OFFSET (OCL_BLOCK_OFFSET + OCL_BATCH_ID * OCL_N_POINTS)

#define OCL_N_PER_BLOCK (get_local_size(1) << 1)
#define OCL_O4 (get_group_id(1) * OCL_N_PER_BLOCK)
#define OCL_O1 (get_num_groups(1) * OCL_N_PER_BLOCK * get_group_id(0))
#define OCL_O0 (OCL_O1 + OCL_O4)
#define OCL_O3 (get_local_id(1) + get_local_size(1))

unsigned int reverse(register unsigned int x)
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return((x >> 16) | (x << 16));
}

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
        barrier(CLK_LOCAL_MEM_FENCE);
        add_sub_mul_local(&in_lower, &in_upper, out_i, out_ii, &w);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


void ocl_global(__global cpx *in, int tid, float angle, int steps, int dist)
{
    cpx w;    
    w.y = sincos(angle * ((tid << steps) & ((dist - 1) << steps)), &w.x);
    cpx l = *in;
    cpx h = in[dist];
    float x = l.x - h.x;
    float y = l.y - h.y;
    cpx a = { l.x + h.x, l.y + h.y };
    cpx b = { (w.x * x) - (w.y * y), (w.y * x) + (w.x * y) };
    *in = a;
    in[dist] = b;
}

__kernel void ocl_kernel_global(__global cpx *in, float angle, unsigned int lmask, int steps, int dist)
{
    ocl_global(in + get_global_id(1) + (get_global_id(1) & lmask) + get_global_id(0) * get_global_size(1), get_global_id(1), angle, steps, dist);
}

/*
__global__ void cuda_kernel_global(cpx *in, float angle, unsigned int lmask, int steps, int dist)
{
    cu_global(in + OCL_THREAD_ID + (OCL_THREAD_ID & lmask) + OCL_BATCH_ID * OCL_N_POINTS, OCL_THREAD_ID, angle, steps, dist);
}

__global__ void cuda_kernel_global_row(cpx *in, float angle, unsigned int lmask, int steps, int dist)
{
    cu_global(in + (OCL_COL_ID + (OCL_COL_ID & lmask)) + OCL_OFFSET_2D, OCL_COL_ID, angle, steps, dist);
}
*/
/*
// GPU takes care of overall syncronization
__kernel void ocl_kernel_global(__global cpx *in, float angle, unsigned int lmask, int steps, int dist)
{
    cpx w;
    int tid = get_global_id(0);
    in += tid + (tid & lmask);
    w.y = sincos(angle * ((tid << steps) & ((dist - 1) << steps)), &w.x);
    add_sub_mul_global(in, in + dist, &w);
}
*/
__kernel void ocl_kernel_global_row(__global cpx *in, float angle, unsigned int lmask, int steps, int dist)
{
    cpx w;
    int col_id = get_group_id(1) * get_local_size(0) + get_local_id(0);
    in += (col_id + (col_id & lmask)) + get_group_id(0) * get_num_groups(0);
    w.y = sincos(angle * ((col_id << steps) & ((dist - 1) << steps)), &w.x);
    add_sub_mul_global(in, in + dist, &w);
}

// CPU takes care of overall syncronization, limited in problem sizes that can be solved.
// Can be combined with ocl_kernel_global in a manner that the ocl_kernel_global is run until problem can be split into smaller parts.
/*
__kernel void ocl_kernel_local(__global cpx *in, __global cpx *out, __local cpx *shared, float local_angle, int steps_left, int leading_bits, float scalar, const int n_half)
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
*/

__kernel void ocl_kernel_local_row(__global cpx *in, __global cpx *out, __local cpx shared[], float local_angle, int steps_left, int leading_bits, float scalar, int block_range)
{
    int in_low = get_local_id(0);
    int in_high = block_range + in_low;
    int row_start = get_num_groups(0) * get_group_id(0);
    int row_offset = get_group_id(1) * get_local_size(0) * 2;
    in += row_start + row_offset;
    out += row_start;
    shared[in_low] = in[in_low];
    shared[in_high] = in[in_high];
    algorithm_partial(shared, in_high, local_angle, steps_left);
    cpx src_low = { shared[in_low].x * scalar, shared[in_low].y * scalar };
    cpx src_high = { shared[in_high].x * scalar, shared[in_high].y * scalar };
    out[(reverse(in_low + row_offset) >> leading_bits)] = src_low;
    out[(reverse(in_high + row_offset) >> leading_bits)] = src_high;
}


__kernel void ocl_contant_geometry(__local cpx *shared, __local cpx *in_l, __local cpx *in_h, float angle, int steps_limit)
{
    cpx w, l, h;
    int tid = get_local_id(1);
    __local cpx *out_i = shared + (tid << 1),
                *out_ii = out_i + 1;
    float x, y;
    for (int steps = 0; steps < steps_limit; ++steps) {
        l = *in_l;
        h = *in_h;
        x = l.x - h.x;
        y = l.y - h.y;
        w.y = sincos(angle * (tid & (0xFFFFFFFF << steps)), &w.x);
        barrier(CLK_LOCAL_MEM_FENCE);
        cpx a = { l.x + h.x, l.y + h.y };
        cpx b = { (w.x * x) - (w.y * y), (w.y * x) + (w.x * y) };
        *out_i = a;
        *out_ii = b;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void ocl_partial(__global cpx *in, __global cpx *out, __local cpx *shared, unsigned int in_high, unsigned int offset, float local_angle, int steps_left, int leading_bits, float scalar)
{
    int in_low = get_local_id(1);
    __local cpx *in_l = shared + in_low,
                *in_u = shared + in_high;
    *in_l = in[in_low];
    *in_u = in[in_high];
    ocl_contant_geometry(shared, in_l, in_u, local_angle, steps_left);
    cpx a = { in_l->x * scalar, in_l->y * scalar };
    cpx b = { in_u->x * scalar, in_u->y * scalar };
    out[reverse(in_low + offset) >> leading_bits] = a;
    out[reverse(in_high + offset) >> leading_bits] = b;
}


__kernel void ocl_kernel_local(__global cpx *in, __global cpx *out, __local cpx shared[], float local_angle, int steps_left, int leading_bits, float scalar)
{
    ocl_partial(in + OCL_O0, out + OCL_O1, shared, OCL_O3, OCL_O4, local_angle, steps_left, leading_bits, scalar);
}

__kernel void ocl_transpose_kernel(__global cpx *in, __global cpx *out, __local cpx tile[OCL_TILE_DIM][OCL_TILE_DIM], int n)
{
    int i, j;
    // Write to shared from Global (in)
    int bx = get_group_id(0),
        by = get_group_id(1),
        ix = get_local_id(0),
        iy = get_local_id(1);
    int x = bx * OCL_TILE_DIM + ix;
    int y = by * OCL_TILE_DIM + iy;

#pragma unroll OCL_TILE_DIM / OCL_BLOCK_DIM
    for (j = 0; j < OCL_TILE_DIM; j += OCL_BLOCK_DIM) {
#pragma unroll OCL_TILE_DIM / OCL_BLOCK_DIM
        for (i = 0; i < OCL_TILE_DIM; i += OCL_BLOCK_DIM) {
            tile[iy + j][ix + i] = in[(y + j) * n + (x + i)];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x = by * OCL_TILE_DIM + ix;
    y = bx * OCL_TILE_DIM + iy;
#pragma unroll OCL_TILE_DIM / OCL_BLOCK_DIM
    for (j = 0; j < OCL_TILE_DIM; j += OCL_BLOCK_DIM) {
#pragma unroll OCL_TILE_DIM / OCL_BLOCK_DIM
        for (i = 0; i < OCL_TILE_DIM; i += OCL_BLOCK_DIM) {
            out[(y + j) * n + (x + i)] = tile[ix + i][iy + j];
        }
    }
}

__kernel void ocl_timestamp_kernel()
{
    // There be none here!
    // Kernel only used as event trigger to get timestamps! This kernel is enqueued first and last.
}