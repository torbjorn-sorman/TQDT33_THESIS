#ifndef TB_TRANSPOSE_H
#define TB_TRANSPOSE_H

#include "tb_definitions.h"

void transpose(cpx **seq, const int b, const int n);
void transpose_block(cpx **seq, const int b, const int n);

#endif