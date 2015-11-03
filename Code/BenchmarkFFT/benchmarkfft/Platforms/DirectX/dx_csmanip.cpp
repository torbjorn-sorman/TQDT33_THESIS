#include "dx_csmanip.h"
#include <iostream>

void dx_set_dim(LPCWSTR shader_file, int group_size, const int n)
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
    if (n2 > group_size) {
        fmt1 << "$1 " << std::to_string(group_size);
        if ((n2 / group_size) > 1)
            fmt2 << "$1 " << std::to_string(1);
        else
            fmt2 << "$1 " << std::to_string((n2 / group_size));
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

void dx_set_dim_2d(LPCWSTR shader_file, int group_size, const int n)
{
    std::ifstream in_file(shader_file);
    std::stringstream buffer;
    buffer << in_file.rdbuf();
    std::string new_content = buffer.str();
    in_file.close();
    std::regex e_grpx_sz("(#define\\s*GROUP_SIZE_X)\\s*\\d*\\s*$");
    std::regex e_gridx_sz("(#define\\s*GRID_DIM_X)\\s*\\d*\\s*$");
    std::ofstream out_file(shader_file);
    std::stringstream fmt_grp, fmt_gridx, fmt_gridy;
    const int n2 = n >> 1;
    fmt_gridx << "$1 " << std::to_string(n);
    if (n2 > group_size) {
        fmt_grp << "$1 " << std::to_string(group_size);
        fmt_gridy << "$1 " << std::to_string((n2 / group_size));
    }
    else {
        fmt_grp << "$1 " << std::to_string(n2);
        fmt_gridy << "$1 " << std::to_string(1);
    }
    new_content = std::regex_replace(new_content, e_grpx_sz, fmt_grp.str());
    new_content = std::regex_replace(new_content, e_gridx_sz, fmt_gridx.str());
    out_file << new_content;
    out_file.close();
}


void dx_set_dim_trans(LPCWSTR shader_file, const int tile_dim, const int n)
{
    std::ifstream in_file(shader_file);
    std::stringstream buffer;
    buffer << in_file.rdbuf();
    std::string new_content = buffer.str();
    in_file.close();    
    std::regex e_w_sz(      "(#define\\s*WIDTH)\\s*\\d*\\s*$");
    std::regex e_tile_dim(  "(#define\\s*DX_TILE_DIM)\\s*\\d*\\s*$");
    std::ofstream out_file(shader_file);
    std::stringstream fmt_w, fmt_t;
    fmt_w << "$1 " << std::to_string(n);
    fmt_t << "$1 " << std::to_string(tile_dim);
    new_content = std::regex_replace(new_content, e_w_sz, fmt_w.str());
    new_content = std::regex_replace(new_content, e_tile_dim, fmt_t.str());
    out_file << new_content;
    out_file.close();
}