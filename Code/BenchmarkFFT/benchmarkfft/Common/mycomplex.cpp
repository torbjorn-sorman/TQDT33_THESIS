#include "mycomplex.h"

cpx *get_seq(int n, int sinus)
{
    int i;
    cpx *seq;
    seq = (cpx *)malloc(sizeof(cpx) * n);
    for (i = 0; i < n; ++i) {
        seq[i].x = sinus == 0 ? 0.f : (float)sin(M_2_PI * (((double)i) / n));
        seq[i].y = 0.f;
    }
    return seq;
}

cpx *get_seq(int n)
{
    return get_seq(n, 0);
}

cpx *get_seq(int n, cpx *src)
{
    int i;
    cpx *seq;
    seq = (cpx *)malloc(sizeof(cpx) * n);
    for (i = 0; i < n; ++i) {
        seq[i].x = src[i].x;
        seq[i].y = src[i].y;
    }
    return seq;
}

void setup_seq2D(cpx **in, cpx **buf, cpx **ref, int n)
{
    char input_file[40];
    sprintf_s(input_file, 40, "Images/%u.ppm", n);
    int sz;
    *in = (cpx *)malloc(sizeof(cpx) * n * n);
    if (buf != NULL)
        *buf = (cpx *)malloc(sizeof(cpx) * n * n);
    *ref = (cpx *)malloc(sizeof(cpx) * n * n);
    read_image(*in, input_file, &sz);
    memcpy(*ref, *in, sizeof(cpx) * n * n);
}

cpx **get_seq2D(const int n, const int type)
{
    cpx **seq;
    seq = (cpx **)malloc(sizeof(cpx *) * n);
    if (type == 0 || type == 1) {
        for (int i = 0; i < n; ++i) {
            seq[i] = get_seq(n, type);
        }
    }
    else {
        for (int y = 0; y < n; ++y) {
            seq[y] = (cpx *)malloc(sizeof(cpx) * n);
            for (int x = 0; x < n; ++x) {
                seq[y][x] = { (float)x, (float)y };
            }
        }
    }
    return seq;
}

cpx **get_seq2D(const int n, cpx **src)
{
    cpx **seq = (cpx **)malloc(sizeof(cpx *) * n);
    for (int i = 0; i < n; ++i)
        seq[i] = get_seq(n, src[i]);
    return seq;
}

cpx **get_seq2D(const int n)
{
    return get_seq2D(n, 0);
}

void free_seq2D(cpx **seq, const int n)
{
    for (int i = 0; i < n; ++i)
        free(seq[i]);
    free(seq);
}

#define maxf(a, b) (((a)>(b))?(a):(b))

bool invalidCpx(cpx c)
{
    return std::isnan(c.x) || std::isinf(c.x) || std::isnan(c.y) || std::isinf(c.y);
}

double diff_seq(cpx *seq, cpx *ref, float scalar, const int n)
{
    double mDiff = 0.0;
    double mVal = -1;
    cpx rScale = make_cuFloatComplex(scalar, 0);
    for (int i = 0; i < n; ++i) {
        cpx norm = cuCmulf(seq[i], rScale);
        if (invalidCpx(norm))
            return 1.0;
        mVal = maxf(mVal, maxf(cuCabsf(norm), cuCabsf(ref[i])));
        double tmp = cuCabsf(cuCsubf(norm, ref[i]));
        mDiff = tmp > mDiff ? tmp : mDiff;
    }
    return (mDiff / mVal);
}

double diff_seq(cpx *seq, cpx *ref, const int n)
{
    return diff_seq(seq, ref, 1.f, n);
}

double diff_seq(cpx **seq, cpx **ref, const int n)
{
    double diff = 0.0;
    for (int i = 0; i < n; ++i)
        diff = maxf(diff, diff_seq(seq[i], ref[i], 1.f, n));
    return diff;
}

double diff_forward_sinus(cpx *seq, const int n)
{
    if (n < 8) {
        return 0.0;
    }
    const int n2 = (n >> 1);
    cpx *neg = seq + 1;
    cpx *pos = seq + n - 1;
    double diff = cuCabsf(seq[0]);
    diff = maxf(diff, neg->x);
    diff = maxf(diff, pos->x);
    for (int i = 2; i < n - 3; ++i) {
        diff = maxf(diff, cuCabsf(seq[i]));
    }
    diff = maxf(diff, abs(abs(neg->y) - n2));
    diff = maxf(diff, abs(abs(pos->y) - n2));
    return diff / n2;
}