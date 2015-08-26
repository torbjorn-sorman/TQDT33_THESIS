#include "tb_print.h"

#include <stdio.h>

void console_print(tb_cpx *seq, uint32_t n)
{
    if (console_no_print != 0)
        return;
    uint32_t i;
    for (i = 0; i < n; ++i)
        printf("%f\t%f\n", seq[i].r, seq[i].i);
}

void console_print_cmp(tb_cpx *seq, tb_cpx *ref, uint32_t n)
{
    if (console_no_print != 0)
        return;
    uint32_t i;
    for (i = 0; i < n; ++i)
        printf("(%f, %f)\t(%f, %f)\n", seq[i].r, seq[i].i, ref[i].r, ref[i].i);
}

void console_print(uint32_t a, uint32_t b)
{
    if (console_no_print != 0)
        return;
    printf("%u\t%u\n", a, b);
}

void console_newline(uint32_t n)
{
    if (console_no_print != 0)
        return;
    uint32_t i;
    for (i = 0; i < n; ++i)
        printf("\n");
}

void console_separator(uint32_t n)
{
    if (console_no_print != 0)
        return;
    uint32_t i, w;
    for (i = 0; i < n; ++i) {
        for (w = 0; w < 80; ++w) {
            printf("_");
        }
        printf("\n");
    }
}