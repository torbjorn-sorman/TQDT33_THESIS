#pragma once
#ifndef MYFILE_H
#define MYFILE_H

#include <Windows.h>
#include <string>
#include <fstream>

bool dirExists(const std::string& dirName_in);
FILE *getImageFilePointer(char *name, int n, char *type);
FILE *getTextFilePointer(std::string name, std::string *fname);
char *get_kernel_src(LPCWSTR file_name, int *length);

std::string get_file_content(LPCWSTR shader_file);
void manip_content(std::string* content, LPCWSTR var_name, int value);
void set_file_content(LPCWSTR shader_file, std::string content);

#endif