#include "tb_print.h"

#include <stdio.h>

void console_print(cpx *seq, const int n)
{
    if (console_no_print != 0)
        return;
    int i;
    for (i = 0; i < n; ++i)
        printf("%f\t%f\n", seq[i].r, seq[i].i);
}

void console_print_cmp(cpx *seq, cpx *ref, const int n)
{
    if (console_no_print != 0)
        return;
    int i;
    for (i = 0; i < n; ++i)
        printf("(%f, %f)\t(%f, %f)\n", seq[i].r, seq[i].i, ref[i].r, ref[i].i);
}

void console_print(const int a, const int b)
{
    if (console_no_print != 0)
        return;
    printf("%u\t%u\n", a, b);
}

void console_newline(const int n)
{
    if (console_no_print != 0)
        return;
    int i;
    for (i = 0; i < n; ++i)
        printf("\n");
}

void console_separator(const int n)
{
    if (console_no_print != 0)
        return;
    int i, w;
    for (i = 0; i < n; ++i) {
        for (w = 0; w < 80; ++w) {
            printf("_");
        }
        printf("\n");
    }
}