#pragma once
#ifndef MYCOMPLEX_H
#define MYCOMPLEX_H

#include <cmath>
#include "cuComplex.h"

#include "../Definitions.h"
#include "imglib.h"

#define maxf(a, b) (((a)>(b))?(a):(b))

template<typename T> static __inline void swap(T *a, T *b)
{
    T c = *a;
    *a = *b;
    *b = c;
}

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

#endif