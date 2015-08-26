#ifndef TB_PRINT_H
#define TB_PRINT_H

#include "tb_definitions.h"

void console_print(tb_cpx *seq, uint32_t n);
void console_print_cmp(tb_cpx *seq, tb_cpx *ref, uint32_t n);
void console_print(uint32_t a, uint32_t b);
void console_newline(uint32_t n);
void console_separator(uint32_t n);

static int console_no_print = 0;

#endif