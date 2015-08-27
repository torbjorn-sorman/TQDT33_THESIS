#ifndef TB_PRINT_H
#define TB_PRINT_H

#include "tb_definitions.h"

void console_print(const tb_cpx *seq, const int n);
void console_print_cmp(const tb_cpx *seq, const tb_cpx *ref, const int n);
void console_print(const int a, const int b);
void console_newline(const int n);
void console_separator(const int n);

static int console_no_print = 0;

#endif