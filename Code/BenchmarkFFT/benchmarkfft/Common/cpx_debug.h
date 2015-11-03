#pragma once
#ifndef CPX_DEBUG_H
#define CPX_DEBUG_H

#include <iosfwd>

#include "../Definitions.h"
#include "mycomplex.h"

#define SHOW_BLOCKING_DEBUG

void cpx_to_console(cpx *sequence, char *title, int len);
void cpx2d_to_console(cpx *sequence, char *title, int len);
void debug_check_compare(const int n);

static cpx *debug_dx_out = NULL;
static cpx *debug_cuda_out = NULL;

#endif