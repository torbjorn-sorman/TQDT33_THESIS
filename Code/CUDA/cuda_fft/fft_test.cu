#include <Windows.h>
#include <stdio.h>

#include "math.h"

#include "fft_test.cuh"

#define ERROR_MARGIN 0.0001

int checkError(cpx *seq, cpx *ref, const int n, int print)
{
    int j;
    double re, im, i_val, r_val;
    re = im = 0.0;
    for (j = 0; j < n; ++j) {
        r_val = abs(seq[j].x - ref[j].x);
        i_val = abs(seq[j].y - ref[j].y);
        re = re > r_val ? re : r_val;
        im = im > i_val ? im : i_val;
    }
    if (print == 1) printf("Error\tre(e): %f\t im(e): %f\t@%u\n", re, im, n);
    return re > ERROR_MARGIN || im > ERROR_MARGIN;
}

cpx *get_seq(const int n)
{
    return get_seq(n, 0);
}

cpx *get_seq(const int n, const int sinus)
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

cpx *get_seq(const int n, cpx *src)
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

int cmp(const void *x, const void *y)
{
    double xx = *(double*)x, yy = *(double*)y;
    if (xx < yy) return -1;
    if (xx > yy) return  1;
    return 0;
}

double avg(double m[], int n)
{
    int i, cnt, end;
    double sum;
    qsort(m, n, sizeof(double), cmp);
    sum = 0.0;
    cnt = 0;
    end = n < 5 ? n - 1 : 5;
    for (i = 0; i < end; ++i) {
        sum += m[i];
        ++cnt;
    }
    return (sum / (double)cnt);
}