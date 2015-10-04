#ifndef KERNELHELPER_CL
#define KERNELHELPER_CL

typedef struct {
    float x;
    float y;
} cpx;

typedef __global cpx dev_buf;
typedef __local cpx grp_buf;
typedef __global int syn_buf;

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

unsigned int reverse(register unsigned int x)
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return((x >> 16) | (x << 16));
}

void mem_gtos(int low, int high, int offset, grp_buf *shared, dev_buf *device)
{
    shared[low] = device[low + offset];
    shared[high] = device[high + offset];
}

void mem_stog_db(int low, int high, int offset, unsigned int lead, cpx scale, grp_buf *shared, dev_buf *device)
{
    device[(reverse(low + offset) >> lead)] = cpxMul(shared[low], scale);
    device[(reverse(high + offset) >> lead)] = cpxMul(shared[high], scale);
}

void butterflyDev(dev_buf *out, cpx in_lower, cpx in_upper, int index_low, int index_high, float angle)
{
    float w_x, w_y;
    w_y = sincos(angle, &w_x);
    out[index_low] = cpxAdd(in_lower, in_upper);
    out[index_high] = cpxMul(cpxSub(in_lower, in_upper), make_cpx(w_x, w_y));
}

void butterflyGrp(grp_buf *out, cpx in_lower, cpx in_upper, int index_low, int index_high, float angle)
{
    float w_x, w_y;
    w_y = sincos(angle, &w_x);
    out[index_low] = cpxAdd(in_lower, in_upper);
    out[index_high] = cpxMul(cpxSub(in_lower, in_upper), make_cpx(w_x, w_y));
}

#endif