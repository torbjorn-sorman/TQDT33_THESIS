#include "mycomplex.h"

cpx *get_seq(int n, int batches, int sinus)
{
    int i;
    cpx *seq;
    seq = (cpx *)malloc(sizeof(cpx) * n * batches);
    if (seq) {
        for (int b = 0; b < batches; ++b) {
            for (i = 0; i < n; ++i) {
                seq[i + b * n].x = sinus == 0 ? 0.f : (float)sin(M_2_PI * (((double)i) / n));
                seq[i + b * n].y = 0.f;
            }
        }
    }
    return seq;
}

cpx *get_seq(int n, int sinus)
{
    return get_seq(n, 1, sinus);
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
    if (seq) {
        for (i = 0; i < n; ++i) {
            seq[i].x = src[i].x;
            seq[i].y = src[i].y;
        }
    }
    return seq;
}

void setup_seq_2d(cpx **in, cpx **buf, cpx **ref, int batches, int n)
{
    char input_file[40];
    sprintf_s(input_file, 40, "Images/%u.ppm", n);
    int sz;
    size_t total_size = sizeof(cpx) * n * n * batches;
    cpx *_in = (cpx *)malloc(total_size);
    if (_in) {
        if (buf != NULL)
            *buf = (cpx *)malloc(total_size);
        *ref = (cpx *)malloc(total_size);
        read_image(_in, input_file, &sz);
        for (int i = 0; i < batches; ++i)
            memcpy(_in + i * n * n, _in, sizeof(cpx) * n * n);
        memcpy(*ref, _in, total_size);
        *in = _in;
    }
}

void setup_seq_2d(cpx **in, cpx **buf, cpx **ref, int n)
{
    setup_seq_2d(in, buf, ref, 1, n);
}

cpx **get_seq_2d(const int n, const int type)
{
    cpx **seq;
    seq = (cpx **)malloc(sizeof(cpx *) * n);
    if (seq) {
        if (type == 0 || type == 1) {
            for (int i = 0; i < n; ++i) {
                seq[i] = get_seq(n, type);
            }
        }
        else {
            for (int y = 0; y < n; ++y) {
                seq[y] = (cpx *)malloc(sizeof(cpx) * n);
                if (seq[y]) {
                    for (int x = 0; x < n; ++x) {
                        seq[y][x] = { (float)x, (float)y };
                    }
                }
            }
        }
    }
    return seq;
}

cpx **get_seq_2d(const int n, cpx **src)
{
    cpx **seq = (cpx **)malloc(sizeof(cpx *) * n);
    if (seq) {
        for (int i = 0; i < n; ++i)
            seq[i] = get_seq(n, src[i]);
    }
    return seq;
}

cpx **get_seq_2d(const int n)
{
    return get_seq_2d(n, 0);
}

void free_seq_2d(cpx **seq, const int n)
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
    if (seq == NULL)
        return 1;
    double mDiff = 0.0;
    double mVal = -1;
    cpx rScale{scalar, 0};
    for (int i = 0; i < n; ++i) {
        cpx norm = cpx_mul(seq + i, &rScale);
        if (invalidCpx(norm))
            return 1.0;
#if defined(_NVIDIA)
        mVal = maxf(mVal, maxf(cuCabsf(norm), cuCabsf(ref[i])));
        double tmp = cuCabsf(cuCsubf(norm, ref[i]));
#else
        mVal = maxf(mVal, maxf(cpx_abs(norm), cpx_abs(ref[i])));
        double tmp = cpx_abs(cpx_sub(&norm, ref + i));
#endif
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
    if (seq == NULL)
        return 1;
    if (n < 4) {
        return 1.0;
    }
    const int n2 = (n >> 1);
    cpx *neg = seq + 1;
    cpx *pos = seq + n - 1;
    double diff = cpx_abs(seq[0]);
    diff = maxf(diff, neg->x);
    diff = maxf(diff, pos->x);
    for (int i = 2; i < n - 3; ++i) {
        diff = maxf(diff, cpx_abs(seq[i]));
    }
    diff = maxf(diff, abs(abs(neg->y) - n2));
    diff = maxf(diff, abs(abs(pos->y) - n2));
    return diff / n2;
}

double diff_forward_sinus(cpx *seq, int batches, const int n)
{
    double d = 0.0;
    cpx *end = seq + batches * n;
    for (; seq < end; seq += n)
        d = maxf(diff_forward_sinus(seq, n), d);
    return d;
}
    