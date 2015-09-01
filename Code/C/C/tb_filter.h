#ifndef TB_FILTER_H
#define TB_FILTER_H

#include "tb_definitions.h"

void filter_edge(const int val, cpx **seq, const int n);
void filter_blur(const int val, cpx **seq, const int n);

#endif