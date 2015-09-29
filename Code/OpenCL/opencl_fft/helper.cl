#ifndef HELPER_CL
#define HELPER_CL
/*
#define WHAT

#ifdef WHAT
#define __kernel
#define __global
#define __local
#define get_local_id(i) 1
#define get_group_id(i) 1
#define get_num_groups(i) 1
#define get_local_size(i) 1
#define barrier(f) 1
#define sincos(a,x) 1
#define log2(v) 1
#else
#define __kernel __kernel
#define __global __global
#define __local __local
#define get_local_id(i) get_local_id((i))
#define get_group_id(i) get_group_id((i))
#define get_num_groups(i) get_num_groups((i))
#define get_local_size(i) get_local_size((i))
#define barrier(f) barrier((f))
#define sincos(a,x) sincos((a), (x))
#define log2(v) log2((v))
#endif
*/
//#define M_2_PI 6.28318530718f
//#define M_PI 3.14159265359f

typedef struct {
    float x;
    float y;
} cpx;

static __inline cpx make_cpx(const float x, const float y)
{
    cpx a;
    a.x = x;
    a.y = y;
    return a;
}

static __inline cpx cpxAdd(const cpx a, const cpx b)
{
    return make_cpx(
        a.x + b.x,
        a.y + b.y
        );
}

static __inline cpx cpxSub(const cpx a, const cpx b)
{
    return make_cpx(
        a.x - b.x,
        a.y - b.y
        );
}

static __inline cpx cpxMul(const cpx a, const cpx b)
{
    return make_cpx(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
        );
}

static __inline void cpx_add_sub_mul(__local cpx *out_lower,__local cpx *out_upper, cpx in_lower, cpx in_upper, cpx w)
{
    (*out_lower) = cpxAdd(in_lower, in_upper);
    (*out_upper) = cpxMul(cpxSub(in_lower, in_upper), w);
}

static __inline unsigned int reverseBitOrder(int x, const int l)
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return((x >> 16) | (x << 16)) >> l;
}

#endif