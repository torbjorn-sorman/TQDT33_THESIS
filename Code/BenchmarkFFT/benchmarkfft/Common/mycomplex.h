#pragma once
#ifndef MYCOMPLEX_H
#define MYCOMPLEX_H

#include <cmath>
#if defined(_NVIDIA)
#include "cuComplex.h"
#endif
#include "../Definitions.h"
#include "imglib.h"

#define maxf(a, b) (((a)>(b))?(a):(b))

template<typename T> static __inline void swap(T *a, T *b)
{
    T c = *a;
    *a = *b;
    *b = c;
}

cpx *get_seq(int n, int batches, int sinus);
cpx *get_seq(int n, int sinus);
cpx *get_seq(int n);
cpx *get_seq(int n, cpx *src);

cpx **get_seq_2d(const int n, const int type);
cpx **get_seq_2d(const int n, cpx **src);
cpx **get_seq_2d(const int n);
void free_seq_2d(cpx **seq, const int n);
void setup_seq_2d(cpx **in, cpx **buf, cpx **ref, int n);

double diff_seq(cpx *seq, cpx *ref, float scalar, const int n);
double diff_seq(cpx *seq, cpx *ref, const int n);
double diff_seq(cpx **seq, cpx **ref, const int n);
double diff_forward_sinus(cpx *seq, const int n);
double diff_forward_sinus(cpx *seq, int batches, const int n);

static __inline cpx cpx_mul(cpx *a, cpx *b)
{
    return{ a->x * b->x - a->y * b->y, a->y * b->x + a->x * b->y };
}

static __inline float cpx_abs(cpx x)
{
    float a = x.x;
    float b = x.y;
    float v, w, t;
    a = fabsf(a);
    b = fabsf(b);
    if (a > b) {
        v = a;
        w = b;
    }
    else {
        v = b;
        w = a;
    }
    t = w / v;
    t = 1.0f + t * t;
    t = v * sqrtf(t);
    if ((v == 0.0f) || (v > 3.402823466e38f) || (w > 3.402823466e38f)) {
        t = v + w;
    }
    return t;
}
#if defined(_AMD)
static __inline cpx cpx_sub(cpx *a, cpx *b)
{
    return {a->x - b->x, a->y - b->y};
}
#endif

template<typename T>
static void free_all(T t)
{
    free(t);
}

template<typename T, typename... Args>
static void free_all(T t, Args... args)
{
    free(t);
    free_all(args...);
}

#endif