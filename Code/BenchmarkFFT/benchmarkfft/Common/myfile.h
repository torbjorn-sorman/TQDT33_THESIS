#pragma once
#ifndef MYFILE_H
#define MYFILE_H

#include <Windows.h>
#include <string>
#include <fstream>

FILE *get_img_file_pntr(char *name, int n, char *type);
FILE *get_txt_file_pntr(std::string name, std::string *fname);
char *get_kernel_src(LPCWSTR file_name, int *length);
char *get_kernel_src(std::string file_name, int *length);
char *get_kernel_src_from_string(std::string contents, int *length);

std::string get_file_content(LPCWSTR shader_file);
std::string get_file_content(std::string shader_file);

void manip_content(std::string* content, LPCWSTR var_name, int value);
void set_file_content(LPCWSTR shader_file, std::string content);

#endif