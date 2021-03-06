#ifndef TB_PRINT_H
#define TB_PRINT_H

#include "tb_definitions.h"

void console_print(cpx *seq, const int n);
void console_print_cmp(cpx *seq, cpx *ref, const int n);
void console_print(const int a, const int b);
void console_newline();
void console_newline(const int n);
void console_separator();
void console_separator(const int n);

static int console_no_print = 0;

#endif