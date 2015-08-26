#ifndef TB_TRANSPOSE_H
#define TB_TRANSPOSE_H

#include "tb_definitions.h"

void transpose(tb_cpx **seq, uint32_t n);
void transpose_block(tb_cpx **seq, const uint32_t size, const uint32_t block_size);

#endif