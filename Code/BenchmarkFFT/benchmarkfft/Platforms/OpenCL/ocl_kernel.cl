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

unsigned int reverse(register unsigned int x)
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return((x >> 16) | (x << 16));
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
    int arg1 = (get_group_id(1) * get_local_size(1) + get_local_id(1)),
        arg0 = (arg1 + (arg1 & lmask) + get_group_id(0) * ((get_num_groups(1) * get_local_size(1)) << 1));
    ocl_global(in + arg0, arg1, angle, steps, dist);
}

__kernel void ocl_kernel_global_row(__global cpx *in, float angle, unsigned int lmask, int steps, int dist)
{
    int arg1 = (get_group_id(1) * get_local_size(0) + get_local_id(0)),
        arg0 = (arg1 + (arg1 & lmask) + (get_group_id(0) + get_group_id(2) * get_num_groups(0)) * get_num_groups(0));
    ocl_global(in + arg0, arg1, angle, steps, dist);
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
    int n_block = (get_local_size(1) << 1),
        arg4 = (get_group_id(1) * n_block),
        arg1 = (get_num_groups(1) * n_block * get_group_id(0)),
        arg0 = (arg1 + arg4),
        arg3 = (get_local_id(1) + get_local_size(1));
    ocl_partial(in + arg0, out + arg1, shared, arg3, arg4, local_angle, steps_left, leading_bits, scalar);
}

__kernel void ocl_kernel_local_row(__global cpx *in, __global cpx *out, __local cpx shared[], float local_angle, int steps_left, int leading_bits, float scalar)
{
    int arg4 = ((get_group_id(1) * get_local_size(0)) << 1),
        arg1 = ((get_group_id(0) + get_group_id(2) * get_num_groups(0)) * get_num_groups(0)),
        arg0 = (arg1 + arg4),
        arg3 = (get_local_id(0) + get_local_size(0));
    ocl_partial(in + arg0, out + arg1, shared, arg3, arg4, local_angle, steps_left, leading_bits, scalar);
}

__kernel void ocl_transpose_kernel(__global cpx *in, __global cpx *out, __local cpx tile[OCL_TILE_DIM][OCL_TILE_DIM], int n)
{
    int i, j;
    int bx = get_group_id(0),
        by = get_group_id(1),
        ix = get_local_id(0),
        iy = get_local_id(1);
    int x = bx * OCL_TILE_DIM + ix;
    int y = by * OCL_TILE_DIM + iy;

    int offset = get_num_groups(0) * OCL_TILE_DIM;
    offset = (get_group_id(2) * offset * offset);
    in += offset;
    out += offset;

    for (j = 0; j < OCL_TILE_DIM; j += OCL_BLOCK_DIM) {
        for (i = 0; i < OCL_TILE_DIM; i += OCL_BLOCK_DIM) {
            tile[iy + j][ix + i] = in[(y + j) * n + (x + i)];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x = by * OCL_TILE_DIM + ix;
    y = bx * OCL_TILE_DIM + iy;
    for (j = 0; j < OCL_TILE_DIM; j += OCL_BLOCK_DIM) {
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