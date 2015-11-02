#pragma once
#ifndef DX_CSMANIP_H
#define DX_CSMANIP_H

#include <iosfwd>
#include <sstream>
#include <fstream>
#include <regex>
#include "../../Definitions.h"

#include <comdef.h>

void dx_set_dim(LPCWSTR shader_file, int group_size, const int n);
void dx_set_dim_2d(LPCWSTR shader_file, int group_size, const int n);
void dx_set_dim_trans(LPCWSTR shader_file, const int tile_dim, const int n);

#endif