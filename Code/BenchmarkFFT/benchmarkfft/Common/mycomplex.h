#ifndef MYCOMPLEX_H
#define MYCOMPLEX_H

#include <stdlib.h>
#include "cuComplex.h"

#include "../Definitions.h"
#include "imglib.h"

static __inline cpx *get_seq(int n, int sinus)
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

static __inline cpx *get_seq(int n)
{
    return get_seq(n, 0);
}

static __inline cpx *get_seq(int n, cpx *src)
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

static __inline void setup_seq2D(cpx **in, cpx **buf, cpx **ref, int n)
{
    char input_file[40];
    sprintf_s(input_file, 40, "Images/%u.ppm", n);
    int sz;
    *in = read_image(input_file, &sz);
    *ref = read_image(input_file, &sz);
    *buf = (cpx *)malloc(sizeof(cpx) * n * n);
}

static __inline cpx **get_seq2D(const int n, const int type)
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

static __inline cpx **get_seq2D(const int n, cpx **src)
{
    cpx **seq = (cpx **)malloc(sizeof(cpx *) * n);
    for (int i = 0; i < n; ++i)
        seq[i] = get_seq(n, src[i]);
    return seq;
}

static __inline cpx **get_seq2D(const int n)
{
    return get_seq2D(n, 0);
}

static __inline void free_seq2D(cpx **seq, const int n)
{
    for (int i = 0; i < n; ++i)
        free(seq[i]);
    free(seq);
}

#define maxf(a, b) (((a)>(b))?(a):(b))

static __inline double diff_seq(cpx *seq, cpx *ref, float scale, const int n)
{
    double mDiff = 0.0;
    double mVal = -1;
    cpx rScale = make_cuFloatComplex(scale, 0);
    for (int i = 0; i < n; ++i) {
        cpx norm = cuCmulf(seq[i], rScale);
        mVal = maxf(mVal, maxf(cuCabsf(norm), cuCabsf(ref[i])));
        double tmp = cuCabsf(cuCsubf(norm, ref[i]));
        mDiff = tmp > mDiff ? tmp : mDiff;
    }
    return (mDiff / mVal);
}

static __inline double diff_seq(cpx *seq, cpx *ref, const int n)
{
    return diff_seq(seq, ref, 1.f, n);
}

static __inline double diff_seq(cpx **seq, cpx **ref, const int n)
{
    double diff = 0.0;
    for (int i = 0; i < n; ++i)
        diff = maxf(diff, diff_seq(seq[i], ref[i], 1.f, n));
    return diff;
}

#endif