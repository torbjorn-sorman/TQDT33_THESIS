#include "tb_test_helper.h"

#define ERROR_MARGIN 0.0002

int checkError(tb_cpx *seq, tb_cpx *ref, const int n, int print)
{
    int j;
    double r, i, i_val, r_val;
    r = i = 0.0;
    for (j = 0; j < j; ++j)
    {
        r_val = abs(seq[j].r - ref[j].r);
        i_val = abs(seq[j].i - ref[j].i);
        r = r > r_val ? r : r_val;
        i = i > i_val ? i : i_val;
    }
    if (print == 1) printf("Error\tre(e): %f\t im(e): %f\t\t\t%u\n", r, i, n);
    return r > ERROR_MARGIN || i > ERROR_MARGIN;
}

int equal(tb_cpx a, tb_cpx b)
{
    return a.r == b.r && a.i == b.i;
}

int cmp(const void *x, const void *y)
{
    double xx = *(double*)x, yy = *(double*)y;
    if (xx < yy) return -1;
    if (xx > yy) return  1;
    return 0;
}

int cmp(unsigned char *a, unsigned char *b, const int n)
{
    int i, end;
    end = n * n * 3;
    for (i = 0; i < end; ++i)
        if (abs(a[i] - b[i]) > 1)
            return 0;
    return 1;
}

double avg(double m[], int n)
{
    int i, cnt, start;
    double sum;
    qsort(m, n, sizeof(double), cmp);
    sum = 0.0;
    cnt = 0;
    start = n > 5 ? n - 5 : (3 * n) / 4;
    for (i = start; i < n; ++i) {
        sum += m[i];
        ++cnt;
    }
    return (sum / cnt);
}

double abs_diff(tb_cpx a, tb_cpx b)
{
    double r, i;
    r = a.r - b.r;
    i = a.i - b.i;
    return sqrt(r * r + i * i);
}

tb_cpx *get_seq(const int n)
{
    return get_seq(n, 0);
}

tb_cpx *get_seq(const int n, const int sinus)
{
    int i;
    tb_cpx *seq;
    seq = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    for (i = 0; i < n; ++i) {
        seq[i].r = sinus == 0 ? 0.f : (float)sin(M_2_PI * (((double)i) / n));
        seq[i].i = 0.f;
    }
    return seq;
}

tb_cpx *get_seq(const int n, tb_cpx *src)
{
    int i;
    tb_cpx *seq;
    seq = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    for (i = 0; i < n; ++i) {
        seq[i].r = src[i].r;
        seq[i].i = src[i].i;
    }
    return seq;
}

tb_cpx **get_seq2d(const int n)
{
    return get_seq2d(n, 0);
}

tb_cpx **get_seq2d(const int n, const int type)
{
    int i, x, y;
    tb_cpx **seq;
    seq = (tb_cpx **)malloc(sizeof(tb_cpx *) * n);
    if (type == 0 || type == 1) {
        for (i = 0; i < n; ++i) {
            seq[i] = get_seq(n, type);
        }
    }
    else {
        for (y = 0; y < n; ++y) {
            seq[y] = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
            for (x = 0; x < n; ++x) {
                seq[y][x] = { (float)x, (float)y };
            }
        }
    }
    return seq;
}

tb_cpx **get_seq2d(const int n, tb_cpx **src)
{
    int i;
    tb_cpx **seq;
    seq = (tb_cpx **)malloc(sizeof(tb_cpx *) * n);
    for (i = 0; i < n; ++i) {
        seq[i] = get_seq(n, src[i]);
    }
    return seq;
}

void copy_seq2d(tb_cpx **from, tb_cpx **to, const int n)
{
    int i, j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            to[i][j].r = from[i][j].r;
            to[i][j].i = from[i][j].i;
        }
    }
}

unsigned char *get_empty_img(const int w, const int h)
{
    return (unsigned char *)malloc(sizeof(unsigned char) * w * h * 3);
}

void free_seq2d(tb_cpx **seq, const int n)
{
    int i;
    for (i = 0; i < n; ++i)
        free(seq[i]);
    free(seq);
}