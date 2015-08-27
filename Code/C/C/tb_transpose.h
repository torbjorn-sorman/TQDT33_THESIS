#ifndef TB_TRANSPOSE_H
#define TB_TRANSPOSE_H

#include "tb_definitions.h"

void transpose(tb_cpx **seq, const int b, const int n);
void transpose_openmp(tb_cpx **seq, const int b, const int n);
void transpose_block(tb_cpx **seq, const int b, const int n);
void transpose_block_openmp(tb_cpx **seq, const int b, const int n);

#endif