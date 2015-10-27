#ifndef DX_CSMANIP_H
#define DX_CSMANIP_H

#include <stdio.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <regex>
#include "../../Definitions.h"

#include <comdef.h>
#include <comip.h>

static __inline void dx_set_dim(LPCWSTR shader_file, const int n)
{
    std::ifstream in_file(shader_file);
    std::stringstream buffer;
    buffer << in_file.rdbuf();
    std::string new_content = buffer.str();
    in_file.close();

    std::regex e_grp_sz("(#define\\s*GROUP_SIZE_X)\\s*\\d*$");
    std::regex e_num_blk("(#define\\s*GRID_SIZE_X)\\s*\\d*$");

    std::ofstream out_file(shader_file);
    std::stringstream fmt1, fmt2;
    const int n2 = n >> 1;
    if (n2 > MAX_BLOCK_SIZE) {
        fmt1 << "$1 " << std::to_string(MAX_BLOCK_SIZE);
        if ((n2 / MAX_BLOCK_SIZE) > HW_LIMIT)
            fmt2 << "$1 " << std::to_string(1);
        else
            fmt2 << "$1 " << std::to_string((n2 / MAX_BLOCK_SIZE));
    }
    else {
        fmt1 << "$1 " << std::to_string(n2);
        fmt2 << "$1 " << std::to_string(1);
    }
    new_content = std::regex_replace(new_content, e_grp_sz, fmt1.str());
    new_content = std::regex_replace(new_content, e_num_blk, fmt2.str());

    out_file << new_content;
    out_file.close();
}

static __inline void dx_set_dim_2d(LPCWSTR shader_file, const int n)
{
    std::ifstream in_file(shader_file);
    std::stringstream buffer;
    buffer << in_file.rdbuf();
    std::string new_content = buffer.str();
    in_file.close();
    std::regex e_grpx_sz("(#define\\s*GROUP_SIZE_X)\\s*\\d*$");
    std::regex e_gridx_sz("(#define\\s*GRID_DIM_X)\\s*\\d*$");
    std::regex e_gridy_sz("(#define\\s*GRID_DIM_Y)\\s*\\d*$");
    std::ofstream out_file(shader_file);
    std::stringstream fmt_grp, fmt_gridx, fmt_gridy;
    const int n2 = n >> 1;
    fmt_gridx << "$1 " << std::to_string(n);
    if (n2 > MAX_BLOCK_SIZE) {
        fmt_grp << "$1 " << std::to_string(MAX_BLOCK_SIZE);
        fmt_gridy << "$1 " << std::to_string((n2 / MAX_BLOCK_SIZE));
    }
    else {
        fmt_grp << "$1 " << std::to_string(n2);
        fmt_gridy << "$1 " << std::to_string(1);
    }
    new_content = std::regex_replace(new_content, e_grpx_sz, fmt_grp.str());
    new_content = std::regex_replace(new_content, e_gridx_sz, fmt_gridx.str());
    new_content = std::regex_replace(new_content, e_gridy_sz, fmt_gridy.str());
    out_file << new_content;
    out_file.close();
}


static __inline void dx_set_dim_trans(LPCWSTR shader_file, const int n)
{
    std::ifstream in_file(shader_file);
    std::stringstream buffer;
    buffer << in_file.rdbuf();
    std::string new_content = buffer.str();
    in_file.close();    
    std::regex e_w_sz("(#define\\s*WIDTH)\\s*\\d*$");
    std::ofstream out_file(shader_file);
    std::stringstream fmt_w;
    fmt_w << "$1 " << std::to_string(n);
    new_content = std::regex_replace(new_content, e_w_sz, fmt_w.str());
    out_file << new_content;
    out_file.close();
}

#endif