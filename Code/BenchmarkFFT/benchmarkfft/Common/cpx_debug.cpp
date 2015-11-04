#include "cpx_debug.h"
#include <iostream>

void cpx_to_console(cpx *sequence, char *title, int len)
{
    if (sequence == NULL) {
        printf("\tSequence is NULL\n");
        return;
    }
    std::cout << title << ":" << std::endl;
    for (int i = 0; i < len; ++i)
        printf("\t%d: %f\t%f\n", i, sequence[i].x, sequence[i].y);
}

void cpx2d_to_console(cpx *sequence, char *title, int len)
{
    std::cout << title << ":" << std::endl;
    for (int y = 0; y < 7; ++y) {
        for (int x = 0; x < 7; ++x) {
            cpx *c = sequence + y * len + x;
            printf("(%.0f_%.0f)", c->x, c->y);
        }
        std::cout << std::endl;
    }
}

void debug_check_compare(const int n)
{
    if (debug_cuda_out != NULL && debug_dx_out != NULL) {
        printf("Debug Diff: %f\n", diff_seq(debug_dx_out, debug_cuda_out, n));
        free(debug_cuda_out);
        free(debug_dx_out);
    }
}